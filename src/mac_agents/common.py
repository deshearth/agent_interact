from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import uuid

import numpy as np
import torch
import yaml


def config(path="configs/paper_binary.yaml"):
    return yaml.safe_load(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def code_digest():
    h = hashlib.sha256()
    for p in sorted(Path("src/mac_agents").rglob("*.py")):
        h.update(str(p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_for(request):
    if request == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable in this execution environment")
    return torch.device(request)


def environment():
    hardware = subprocess.run(["sysctl", "hw.memsize", "machdep.cpu.brand_string"],
                              capture_output=True, text=True) if platform.system() == "Darwin" else None
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "execution_location": "local Mac" if platform.system() == "Darwin" else "non-Mac host",
        "system": platform.system(), "platform": platform.platform(),
        "architecture": platform.machine(), "python": sys.version,
        "executable": sys.executable, "disk_free_bytes": shutil.disk_usage(".").free,
        "hardware": hardware.stdout.strip() if hardware and hardware.returncode == 0 else None,
        "hardware_query_error": hardware.stderr.strip() if hardware and hardware.returncode else None,
        "torch": torch.__version__, "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "torch", "scipy", "scikit-learn", "pyyaml", "pandas", "pyarrow"]},
    }


def new_run(label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path("runs") / f"{stamp}-{label}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def inspect_environment():
    result = environment()
    json_write("reports/environment.json", result)
    Path("reports/environment.md").write_text(
        "# Execution environment\n\n" + "\n".join(f"- **{k}:** `{v}`" for k, v in result.items()
                                                 if k != "packages") + "\n")
    print(json.dumps(result, indent=2))
    return result
