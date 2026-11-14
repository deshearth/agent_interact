"""Download a pinned public MLX conversion into this project's cache."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path("runs/local_llama_intervention_v1")
REPO = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "model_provenance.json"
    if target.exists():
        saved = json.loads(target.read_text())
        path = Path(saved["snapshot_path"])
        for file, expected in saved["files_sha256"].items():
            assert digest(path / file) == expected
        print(json.dumps({"available": True, "reused": True, "snapshot_path": str(path)})); return
    info = HfApi().model_info(REPO, files_metadata=True)
    revision = info.sha
    print(json.dumps({"repository": REPO, "revision": revision, "gated": info.gated,
                      "download_bytes": sum(s.size or 0 for s in info.siblings)}), flush=True)
    path = Path(snapshot_download(REPO, revision=revision,
                cache_dir=str(Path(".cache/huggingface_source").resolve()), max_workers=2,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer.model", "LICENSE*", "README.md"]))
    files = [p for p in path.rglob("*") if p.is_file()]
    record = {"repository": REPO, "revision": revision, "snapshot_path": str(path.resolve()),
              "downloaded_utc": datetime.now(timezone.utc).isoformat(),
              "base_model": "meta-llama/Llama-3.1-8B-Instruct", "conversion": "MLX 4-bit, group size in model config",
              "local_source_is_exact_historical_backend": False, "paid_api_calls": 0,
              "preparation_code_sha256": digest(__file__),
              "files_sha256": {str(p.relative_to(path)): digest(p) for p in files},
              "downloaded_bytes": sum(p.stat().st_size for p in files),
              "packages": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers", "huggingface-hub")}}
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"available": True, "snapshot_path": str(path), "bytes": record["downloaded_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
