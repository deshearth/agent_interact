"""Add a stopped-Mac snapshot to a benchmark workspace without changing its inputs.

Existing benchmark files must be byte-identical. Additional immutable source
records are copied first; the handoff manifest is committed atomically last.
This permits an already queued/running benchmark to finish in the same runtime.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import read, write, sha, relative_path


def promote(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Use a separate handoff staging directory")
    with (target / "hpc3/.promotion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before_sha = sha(target / "hpc3/manifest.json")
        old, new = read(target / "hpc3/manifest.json"), read(source / "hpc3/manifest.json")
        if new.get("mode") != "handoff" or not new.get("production_export"):
            raise ValueError("The source must be a stopped-Mac handoff")
        if old["source_protocol_sha256"] != new["source_protocol_sha256"]:
            raise ValueError("Scientific source protocol changed")
        for name, entry in old["files"].items():
            if new["files"].get(name) != entry:
                raise ValueError(f"Benchmark input would change: {name}")
        planned = []
        # Complete validation happens before any new data or manifest is written.
        for name, entry in new["files"].items():
            rel = relative_path(name, new["original_repository"])
            src, dest = source / rel, target / rel
            if not src.resolve().is_relative_to(source) or not dest.resolve().is_relative_to(target):
                raise ValueError("A package path escapes its workspace")
            if sha(src) != entry["sha256"]:
                raise ValueError(f"Damaged handoff file: {name}")
            if dest.exists():
                if sha(dest) != entry["sha256"]:
                    raise ValueError(f"Existing record conflicts: {name}")
            else:
                planned.append((src, dest, entry["sha256"]))
        for src, dest, digest in planned:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(dest.suffix + ".handoff.tmp")
            shutil.copyfile(src, temporary)
            if sha(temporary) != digest:
                raise ValueError(f"Copy verification failed: {dest}")
            temporary.replace(dest)
        if sha(target / "hpc3/manifest.json") != before_sha:
            raise RuntimeError("Another operation changed the target manifest")
        write(target / "hpc3/manifest.json", new)
        result = {"completed_utc": datetime.now(timezone.utc).isoformat(),
                  "previous_manifest_sha256": before_sha,
                  "handoff_manifest_sha256": sha(target / "hpc3/manifest.json"),
                  "new_files": len(planned), "existing_benchmark_inputs_unchanged": True}
        write(target / "hpc3/handoff_import.json", result)
        return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(promote(args.source, args.target))
