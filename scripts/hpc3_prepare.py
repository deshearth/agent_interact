"""Export a verified, credential-free HPC3 benchmark or stopped-study handoff.

This tool does not stop the Mac collector, connect to HPC3, or submit jobs.
Only an explicit allowlist enters the archive. Original frozen files are copied
byte for byte; their old absolute paths are resolved through a manifest mapping.
"""
from argparse import ArgumentParser
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

REPO = Path(__file__).resolve().parents[1]
STUDY = Path("runs/question_breadth_v1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def relative_path(path, origin):
    """Resolve only within the original repository, rejecting traversal."""
    path, origin = Path(path), Path(origin)
    if path.is_absolute():
        path = path.relative_to(origin)
    if not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe package path: {path}")
    return path


@contextmanager
def idle_collection(root):
    """All three writers must be stopped for an authoritative handoff."""
    with ExitStack() as stack:
        for name in [".background.lock", ".pipeline.lock", ".source.lock"]:
            handle = stack.enter_context((root / name).open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("The Mac collection is still active; benchmark export is safe, handoff is not") from exc
        yield


def fixture_groups(repo):
    """Reference outputs from the same frozen float32 source, never assessment.

    The first two groups exercise A/B and argument generation. The third checks
    separate samples sharing a prefix. Production additionally requires actual
    train-conversation inboxes for both opinion and message generation.
    """
    root = repo / STUDY
    benchmark = read(root / "source_benchmark.json")
    if benchmark["source_code_sha256"] != sha(repo / "scripts/breadth_source.py"):
        raise ValueError("Stale local source benchmark")
    groups = []
    for key, category in [("batched", "opinion_no_inbox"), ("messages", "message_no_inbox"),
                          ("eight_independent_samples", "independent_votes")]:
        record = benchmark[key]
        if record["precision"] != "float32":
            raise ValueError("Reference must use the current float32 source")
        groups.append({"id": key, "category": category, "reference": record,
                       "reference_file": str(STUDY / "source_benchmark.json"),
                       "reference_sha256": sha(root / "source_benchmark.json")})
    selection = root / "selection.json"
    if selection.exists():
        for category, step, phase in [("opinion_with_inbox", 1, "opinions0"),
                                      ("message_with_inbox", 2, "messages0")]:
            for q in read(selection)["questions"]:
                if q["split"] != "train":
                    continue
                identity = f"{q['qid']}__{q['graph_id']}__r00|round{step}|{phase}"
                path = root / "chunks" / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
                if not path.exists():
                    continue
                record = read(path)
                if record["group_id"] != identity or record["protocol_sha256"] != sha(root / "source_protocol.json"):
                    raise ValueError("Unexpected source chunk identity")
                groups.append({"id": identity, "category": category, "reference": record,
                               "reference_file": str(path.relative_to(repo)), "reference_sha256": sha(path)})
                break
    categories = {g["category"] for g in groups}
    return {"groups": groups, "required_categories": ["opinion_no_inbox", "message_no_inbox",
            "independent_votes", "opinion_with_inbox", "message_with_inbox"],
            "full_coverage": {"opinion_with_inbox", "message_with_inbox"} <= categories,
            "source_protocol_sha256": sha(root / "source_protocol.json"),
            "gate": {"max_first_token_probability_difference": 0.001,
                     "require_exact_generated_tokens": True, "require_repeatable_tokens": True},
            "interpretation": "Finite cross-device regression check, not proof of identical distributions on all prompts. A failed gate never permits production collection."}


def export_package(destination, mode="benchmark", repo=REPO):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new export directory: {destination}")
    root = repo / STUDY
    with ExitStack() as stack:
        if mode == "handoff":
            stack.enter_context(idle_collection(root))
            if not (root / "selection.json").exists():
                raise RuntimeError("Finish question screening and freeze selection before handoff")
        protocol = read(root / "source_protocol.json")
        files = {}
        for old_path, digest in protocol["source_files"].items():
            rel = relative_path(old_path, repo)
            if sha(repo / rel) != digest:
                raise ValueError(f"Frozen source changed: {rel}")
            files[str(rel)] = repo / rel
        extra = [STUDY / "source_protocol.json", STUDY / "source_benchmark.json",
                 Path("scripts/hpc3_prepare.py"), Path("scripts/hpc3_breadth.py")]
        extra += [Path("hpc3") / name for name in ["README.md", "environment.json", "requirements-cuda12.txt",
                  "container.sh", "bootstrap.sbatch", "benchmark.sbatch", "collect.sbatch"]
                  if (repo / "hpc3" / name).is_file()]
        for rel in extra:
            if "__pycache__" not in rel.parts:
                files[str(rel)] = repo / rel
        if mode == "handoff":
            # Phase checkpoints are mutable summaries; immutable request chunks
            # reconstruct them exactly and remain valid across resumes.
            for dirname in ["constructed", "screen", "chunks", "episodes"]:
                for path in (root / dirname).rglob("*.json"):
                    files[str(path.relative_to(repo))] = path
            for path in root.glob("selection*.json"):
                files[str(path.relative_to(repo))] = path
            frozen = root / "models_frozen.json"
            if frozen.exists():
                files[str(frozen.relative_to(repo))] = frozen
                for path, digest in read(frozen)["artifacts"].items():
                    rel = relative_path(path, repo)
                    if sha(repo / rel) != digest:
                        raise ValueError(f"Frozen predictor changed: {rel}")
                    files[str(rel)] = repo / rel
        fixtures = fixture_groups(repo)
        destination.mkdir(parents=True)
        for rel, source in sorted(files.items()):
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(source) != sha(target):
                raise RuntimeError(f"File changed during export: {rel}")
        write(destination / "hpc3/fixtures.json", fixtures)
        entries = {str(p.relative_to(destination)): {"sha256": sha(p), "bytes": p.stat().st_size}
                   for p in sorted(destination.rglob("*")) if p.is_file()}
        manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": mode, "original_repository": str(repo), "files": entries,
                    "source_protocol_sha256": sha(root / "source_protocol.json"),
                    "full_reference_coverage": fixtures["full_coverage"],
                    "production_export": mode == "handoff", "model_weights_included": False,
                    "paid_api_calls": 0,
                    "collection_policy": "Benchmark never writes study outcomes. Production needs stopped-Mac handoff, complete reference coverage, and passing CUDA replay. Keep source hardware provenance in every new chunk."}
        write(destination / "hpc3/manifest.json", manifest)
        archive = destination.with_suffix(".tar.gz")
        if archive.exists():
            raise FileExistsError(archive)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(destination, arcname=destination.name)
        result = {"directory": str(destination), "archive": str(archive), "archive_sha256": sha(archive),
                  "archive_bytes": archive.stat().st_size, "files": len(entries), "mode": mode,
                  "full_reference_coverage": fixtures["full_coverage"]}
        write(destination.parent / (destination.name + "_export.json"), result)
        return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["benchmark", "handoff"], default="benchmark")
    args = parser.parse_args()
    print(json.dumps(export_package(args.output, args.mode), indent=2))


if __name__ == "__main__":
    main()
