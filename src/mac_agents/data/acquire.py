"""Download a pinned, checksummed subset of the author's public repository.

Usable with the Python standard library before dependencies are installed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY = "batu-el/physics-of-agents"
REVISION = "62bf0e7ddb94e6be936bed6a35aefea94e8d54a9"
METADATA = [
    "LICENSE", "README.md", "implementation_details.md", "reproducibility_audit.md",
    "res/utils.py", "res/4_prediction.ipynb", "res/4_prediction_generalization.ipynb",
    "lib/datagen/dynamics.py", "lib/datagen/samplers.py", "lib/datagen/collect_energy.py",
    "data/obj/train.jsonl", "data/obj/test.jsonl", "data/obj/personas.json",
    "data/obj/persona_embeddings.json", "data/obj/question_embeddings.json",
    "data/obj/question_embeddings_choices.json", "data/J_matrices.json",
    "data/models/gpt-4o-mini/objective_energy/manifest.json",
]


def get_bytes(url: str) -> bytes:
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "mac-agent-dynamics/0.1"})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def acquire(root: Path, *, metadata_only=False, raw=False, workers=6):
    root = Path(root)
    source = root / "raw" / "upstream"
    source.mkdir(parents=True, exist_ok=True)
    tree_path = root / "raw" / "upstream_tree.json"
    if not tree_path.exists():
        tree_path.write_bytes(get_bytes(
            f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"))
    tree = json.loads(tree_path.read_text())
    if tree.get("truncated"):
        raise RuntimeError("Repository tree is truncated; refuse incomplete acquisition")
    entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    paths = list(METADATA)
    if not metadata_only:
        paths.append("data/clean_runs.json")
    if raw:
        paths.extend(p for p in entries if p.startswith(
            "data/models/gpt-4o-mini/objective_energy/") and p.endswith(".json"))
        paths.extend(p for p in entries if p.startswith(
            "data/lowrank/gpt-4o-mini/ood_lowrank/objective_rank_frustration/")
                     and p.endswith(".json"))
    paths = sorted(set(paths))
    missing = [p for p in paths if p not in entries]
    if missing:
        raise RuntimeError(f"Required paths absent from pinned tree: {missing}")
    manifest_path = root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "repository": REPOSITORY, "revision": REVISION,
        "canonical_source": "repository data/clean_runs.json", "license": "Apache-2.0",
        "dataset_revision": None, "dataset_revision_note": "Repository copy selected; no Hugging Face copy downloaded.",
        "files": {}, "commands": [],
    }
    if manifest["revision"] != REVISION:
        raise RuntimeError("Data root belongs to another source revision")

    def one(path):
        destination = source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{quote(path)}"
        data = destination.read_bytes() if destination.exists() else get_bytes(url)
        if blob_sha(data) != entries[path]["sha"]:
            raise RuntimeError(f"Git blob checksum mismatch: {path}")
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(destination)
        return path, {"url": url, "bytes": len(data), "git_blob_sha": entries[path]["sha"],
                      "sha256": hashlib.sha256(data).hexdigest()}

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, p): p for p in paths}
        for count, future in enumerate(as_completed(pending), 1):
            try:
                path, record = future.result()
                manifest["files"][path] = record
            except Exception as exc:
                errors.append({"path": pending[future], "error": str(exc)})
            if count % 50 == 0 or count == len(paths):
                manifest["acquisition_errors"] = errors
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                print(f"Acquired/verified {count}/{len(paths)} files; errors={len(errors)}", flush=True)
    manifest["commands"].append({"metadata_only": metadata_only, "raw": raw, "workers": workers})
    manifest["acquisition_errors"] = errors
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if errors:
        raise RuntimeError(f"{len(errors)} source files failed; see {manifest_path}; rerun to resume")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    acquire(args.root, metadata_only=args.metadata_only, raw=args.raw, workers=args.workers)
