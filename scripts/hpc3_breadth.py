"""CUDA replay and resumable HPC3 collection using the frozen MLX generator."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hpc3_prepare import STUDY, read, relative_path, sha, write

MODEL = Path("models/source_llama")
PROVENANCE = Path("runs/local_llama_intervention_v1/model_provenance.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_package(root=Path(".")):
    root = Path(root)
    manifest = read(root / "hpc3/manifest.json")
    for path, expected in manifest["files"].items():
        rel = relative_path(path, manifest["original_repository"])
        if sha(root / rel) != expected["sha256"]:
            raise ValueError(f"Package file changed: {rel}")
    protocol = read(root / STUDY / "source_protocol.json")
    if sha(root / STUDY / "source_protocol.json") != manifest["source_protocol_sha256"]:
        raise ValueError("Source protocol changed")
    for path, expected in protocol["source_files"].items():
        rel = relative_path(path, manifest["original_repository"])
        if sha(root / rel) != expected:
            raise ValueError(f"Original source changed: {rel}")
    return manifest


def verify_model():
    provenance = read(PROVENANCE)
    for name, expected in provenance["files_sha256"].items():
        if sha(MODEL / name) != expected:
            raise ValueError(f"Model checksum mismatch: {name}")
    return provenance


def download_model():
    verify_package()
    from huggingface_hub import snapshot_download
    provenance = read(PROVENANCE)
    snapshot_download(provenance["repository"], revision=provenance["revision"],
                      local_dir=str(MODEL), allow_patterns=list(provenance["files_sha256"]))
    verify_model()
    print(json.dumps({"model_verified": True, "revision": provenance["revision"]}), flush=True)


def runtime_identity():
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the CUDA workload inside an allocated HPC3 Slurm job")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise RuntimeError("MLX_ENABLE_TF32=0 is required for the float32 source comparison")
    import mlx.core as mx
    if not mx.cuda.is_available() or mx.device_count(mx.gpu) != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    mx.set_default_device(mx.gpu)
    packages = {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions()}
    for name, version in read(PROVENANCE)["packages"].items():
        if packages.get(name) != version:
            raise ValueError(f"Expected {name}=={version}; got {packages.get(name)}")
    if packages.get("mlx-cuda-12") != "0.32.4":
        raise ValueError("Expected mlx-cuda-12==0.32.4")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    # No GPU UUID, PID, hostname or Slurm ID in the reusable identity: these can
    # change on resume. Architecture, driver, code, packages and fixtures cannot.
    info = mx.device_info()
    device = {k: v for k, v in info.items() if k in {"name", "device_name", "architecture", "total_memory", "memory_size"}}
    identity = {"packages": packages, "python": platform.python_version(),
                "gpu_device": device, "nvidia_gpu_types_and_drivers": sorted(set(smi.splitlines())),
                "mlx_enable_tf32": "0", "source_protocol_sha256": sha(STUDY / "source_protocol.json"),
                "fixtures_sha256": sha("hpc3/fixtures.json"),
                "adapter_sha256": sha(__file__), "container_sha256": read(".hpc3/environment.json")["container_sha256"]}
    return identity


def load_source(batch_size, execution=None):
    """Reuse all prompt, sampling, cache, parsing and generation methods.

    Only initialization supplies a portable model path and explicit CUDA device;
    it does not rewrite any pinned Mac source file or model provenance.
    """
    from scripts.breadth_source import BreadthSource
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    class CudaSource(BreadthSource):
        def __init__(self):
            self.mx, self.step, self.make_cache = mx, generate_step, make_prompt_cache
            self.sampler = make_sampler(temp=.7, top_p=1., min_p=0., top_k=0)
            self.provenance = verify_model()
            self.precision = "float32"
            self.execution_batch_size = batch_size
            self.execution = execution
            mx.set_default_device(mx.gpu)
            mx.set_cache_limit(512 * 1024 ** 2)
            begin = time.monotonic()
            self.model, self.tokenizer = load(str(MODEL), tokenizer_config={"trust_remote_code": False})
            self.model.set_dtype(mx.float32)
            mx.eval(self.model.parameters())
            self.load_seconds = time.monotonic() - begin
            self.letter_ids = {}
            for letter in ("A", "B"):
                ids = self.tokenizer.encode(letter, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError("Unexpected A/B tokenizer")
                self.letter_ids[letter] = ids[0]
            self.eos = set(self.tokenizer.eos_token_ids)

        def generate_many(self, prompts, identities, *, batch_size=4, max_tokens=256):
            result = super().generate_many(prompts, identities,
                         batch_size=max(batch_size, self.execution_batch_size), max_tokens=max_tokens)
            if self.execution is not None:
                result["hpc3_execution"] = self.execution
            return result

    return CudaSource()


def requests_from_reference(reference):
    prompts, responses = reference["prompts"], reference["responses"]
    lookup = {p["prompt_sha256"]: p["prompt"] for p in prompts}
    values = []
    for i, response in enumerate(responses):
        if "prompt_sha256" in response:
            values.append(lookup[response["prompt_sha256"]])
        elif len(prompts) == 1:
            values.append(prompts[0]["prompt"])
        elif len(prompts) == len(responses):
            values.append(prompts[i]["prompt"])
        else:
            raise ValueError("Ambiguous reference prompt association")
    return values, [r["sample_id"] for r in responses]


def compare_replay(reference, result, repeat, tolerance):
    expected_ids = [r["sample_id"] for r in reference["responses"]]
    if expected_ids != [r["sample_id"] for r in result["responses"]] or expected_ids != [r["sample_id"] for r in repeat["responses"]]:
        raise ValueError("Replay request identity/order changed")
    tokenized_equal = [(p["prompt"], p["token_ids"]) for p in reference["prompts"]] == [
                       (p["prompt"], p["token_ids"]) for p in result["prompts"]]
    delta, exact, deterministic = 0., 0, 0
    for old, new, again in zip(reference["responses"], result["responses"], repeat["responses"]):
        for key in ["first_literal_A_probability", "first_literal_B_probability", "other_first_token_probability"]:
            values = [float(r[key]) for r in [old, new, again]]
            if not all(math.isfinite(v) and -1e-6 <= v <= 1.000001 for v in values):
                raise ValueError("Invalid replay probability")
            delta = max(delta, abs(values[0] - values[1]), abs(values[1] - values[2]))
        exact += (old["generated_token_ids"], old["finish_reason"]) == (new["generated_token_ids"], new["finish_reason"])
        deterministic += (new["generated_token_ids"], new["finish_reason"]) == (again["generated_token_ids"], again["finish_reason"])
    count = len(expected_ids)
    return {"responses": count, "tokenization_equal": tokenized_equal,
            "exact_reference_responses": exact, "repeatable_responses": deterministic,
            "max_first_token_probability_difference": delta,
            "passed": count > 0 and tokenized_equal and exact == deterministic == count and delta < tolerance}


def benchmark():
    manifest = verify_package()
    fixtures, identity = read("hpc3/fixtures.json"), runtime_identity()
    output = Path("hpc3/results") / ("benchmark_" + os.environ["SLURM_JOB_ID"])
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    output.mkdir(parents=True)
    source = load_source(4)
    first = fixtures["groups"][0]["reference"]
    prompts, ids = requests_from_reference(first)
    source.generate_many(prompts, ids, max_tokens=first["max_tokens"])
    candidates = []
    for batch_size in [4, 8, 16]:
        source.execution_batch_size = batch_size
        checks, total = [], 0.
        for index, group in enumerate(fixtures["groups"]):
            reference = group["reference"]
            prompts, ids = requests_from_reference(reference)
            result = source.generate_many(prompts, ids, max_tokens=reference["max_tokens"])
            repeat = source.generate_many(prompts, ids, max_tokens=reference["max_tokens"])
            check = compare_replay(reference, result, repeat, fixtures["gate"]["max_first_token_probability_difference"])
            # The second invocation is timed to exclude first-shape JIT warmup.
            elapsed = repeat["prefill_seconds"] + repeat["generation_seconds"]
            total += elapsed
            check.update(category=group["category"], warmed_seconds=elapsed,
                         mac_seconds=reference["prefill_seconds"] + reference["generation_seconds"],
                         prompt_token_lengths=[len(p["token_ids"]) for p in reference["prompts"]])
            checks.append(check)
            write(output / f"batch{batch_size}_group{index}.json", {"group_id": group["id"], "result": result, "repeat": repeat, "check": check})
            print(json.dumps({"batch_size": batch_size, **check}), flush=True)
        candidates.append({"batch_size": batch_size, "warmed_seconds": total,
                           "passed": all(c["passed"] for c in checks), "checks": checks})
    passed = [c for c in candidates if c["passed"]]
    choice = min(passed, key=lambda c: c["warmed_seconds"]) if passed else None
    can_collect = bool(choice and fixtures["full_coverage"])
    report = {"created_utc": now(), "runtime": identity, "package_mode": manifest["mode"],
              "full_reference_coverage": fixtures["full_coverage"], "candidates": candidates,
              "selected_batch_size": choice["batch_size"] if choice else None,
              "numerical_gate_passed": can_collect, "production_submitted": False,
              "raw_results": {str(p): sha(p) for p in output.glob("*.json")},
              "interpretation": fixtures["interpretation"]}
    write(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), "numerical_gate_passed": can_collect,
                      "selected_batch_size": report["selected_batch_size"]}), flush=True)
    if not passed:
        raise SystemExit(2)


def check_collection_gate(manifest, report, identity, fixtures):
    if not manifest.get("production_export") or manifest.get("mode") != "handoff":
        raise RuntimeError("Production requires a stopped-Mac handoff export, not a benchmark snapshot")
    if not fixtures["full_coverage"] or not report.get("numerical_gate_passed"):
        raise RuntimeError("Production is locked: the complete cross-device replay has not passed")
    if report["runtime"] != identity:
        raise RuntimeError("GPU/software/fixtures changed since the replay; benchmark this environment again")
    selected = report["selected_batch_size"]
    candidate = [c for c in report["candidates"] if c["batch_size"] == selected]
    if len(candidate) != 1 or not candidate[0]["passed"] or not all(c["passed"] for c in candidate[0]["checks"]):
        raise ValueError("Invalid selected batch-size replay")
    return selected


def authorize_split(split, manifest):
    if split != "assessment":
        return
    path = STUDY / "models_frozen.json"
    if not path.exists():
        raise RuntimeError("Assessment conversation generation is locked until models are frozen")
    for path, digest in read(path)["artifacts"].items():
        rel = relative_path(path, manifest["original_repository"])
        if sha(rel) != digest:
            raise ValueError(f"Frozen predictor changed: {rel}")


def collect(split, report_path, hours):
    if not 0 < hours <= 71:
        raise ValueError("Collection guard must be between zero and 71 hours")
    manifest = verify_package()
    fixtures, report, identity = read("hpc3/fixtures.json"), read(report_path), runtime_identity()
    batch_size = check_collection_gate(manifest, report, identity, fixtures)
    for path, digest in report["raw_results"].items():
        if sha(path) != digest:
            raise ValueError(f"Replay output changed: {path}")
    authorize_split(split, manifest)
    from scripts.breadth_collect import Recorder, TimeLimit, run_episode
    from scripts.local_conversation_pilot import formatter
    from scripts.question_breadth import config, UPSTREAM
    cfg = config()
    questions = [q for q in read(STUDY / "selection.json")["questions"] if q["split"] == split]
    if len(questions) != cfg["questions"]["splits"][split]:
        raise ValueError("Unexpected split size")
    personas = [p["persona"] for p in read(UPSTREAM / "obj/personas.json")]
    graphs = {g["id"]: g["J"] for g in read(UPSTREAM / "J_matrices.json")["graphs"]}
    execution = {"backend": "MLX CUDA 12", "benchmark_report_sha256": sha(report_path),
                 "runtime": identity, "batch_size": batch_size,
                 "scientific_source_protocol_sha256": manifest["source_protocol_sha256"]}
    from scripts.breadth_source import text_sha

    class CudaRecorder(Recorder):
        def request(self, group_id, requests, max_tokens=256):
            path = self.root / "chunks" / f"{text_sha(group_id)}.json"
            if path.exists():
                old = read(path)
                if "hpc3_execution" in old and old["hpc3_execution"] != execution:
                    raise ValueError("Refusing to mix different CUDA execution settings on resume")
            return super().request(group_id, requests, max_tokens)

    with (STUDY / ".source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = load_source(batch_size, execution)
        recorder = CudaRecorder(source, STUDY, time.time() + 3600 * hours)
        write(STUDY / "hpc3_execution.json", execution)
        fmt = formatter()
        try:
            for q in questions:
                run_episode(recorder, q, personas, graphs[q["graph_id"]], fmt, cfg["source"]["update_rounds"])
        except TimeLimit:
            write(STUDY / "hpc3_session.json", {"status": "time_limit", "split": split, "updated_utc": now()})
            raise SystemExit(75)
        write(STUDY / "hpc3_session.json", {"status": "complete", "split": split, "updated_utc": now(),
                                          "new_samples": recorder.new_samples})


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("download-model")
    sub.add_parser("benchmark")
    run = sub.add_parser("collect")
    run.add_argument("--split", choices=["train", "validation", "assessment"], required=True)
    run.add_argument("--benchmark-report", type=Path, required=True)
    run.add_argument("--hours", type=float, default=23.5)
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify_package()
        print(json.dumps({"verified": True, "mode": manifest["mode"], "files": len(manifest["files"])}))
    elif args.command == "download-model":
        download_model()
    elif args.command == "benchmark":
        benchmark()
    else:
        collect(args.split, args.benchmark_report, args.hours)


if __name__ == "__main__":
    main()
