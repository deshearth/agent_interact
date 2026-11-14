"""Read-only removal of individual-message variation on familiar questions.

Fit input centroids on training replicas only, normalize them to unit length,
then replace every real incoming vector at evaluation. This is a dependence
probe of saved predictors, not a retrained control or a source-LLM intervention.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, device_for, digest, json_write
from scripts import argument_response as original
from scripts.argument_features import load_cache
from scripts.argument_mechanisms import CONFIG, ROOT, SplitSnapshots, make_model, split_index

OUT = Path("runs/argument_centroid_probe_v1")
VARIANTS = ("question_mean", "question_stance_mean")


def fit_centroids(arrays, index, vectors):
    qids = sorted({r["qid"] for r in index if r["diagnostic_role"] == "train"})
    question, question_stance = [], []
    for qid in qids:
        es = [e for e, r in enumerate(index) if r["qid"] == qid and r["diagnostic_role"] == "train"]
        ids = arrays["text_ids"][es].ravel()
        stances = arrays["stance"][es].ravel()[ids != 0]
        x = np.array(vectors[ids[ids != 0]], dtype=np.float64)
        question.append(x.mean(0))
        question_stance.append([x[stances == s].mean(0) for s in (-1, 1)])
    q, qs = np.array(question), np.array(question_stance)
    return qids, (q/np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32), (
        qs/np.linalg.norm(qs, axis=-1, keepdims=True)).astype(np.float32)


class CentroidSnapshots(SplitSnapshots):
    def __init__(self, arrays, index, vectors, donors, qids, q, qs, variant):
        super().__init__(arrays, index, vectors, "familiar", "full_arguments", donors)
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.lookup, self.q, self.qs, self.variant = {q: i for i, q in enumerate(qids)}, q, qs, variant

    def __getitem__(self, item):
        b = super().__getitem__(item)
        e, _ = self.pairs[item]
        q = self.lookup[self.index[e]["qid"]]
        if self.variant == "question_mean":
            x = np.broadcast_to(self.q[q], b["text"].shape).copy()
        else:
            x = self.qs[q, (b["stance"].numpy() == 1).astype(np.int64)].copy()
        x[b["signs"].numpy() == 0] = 0
        b["text"] = torch.from_numpy(x)
        return b


def main():
    cfg = config(CONFIG)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    device = device_for("mps")
    plan = json.loads((ROOT/"plan.json").read_text())
    if not plan["complete"]:
        raise ValueError("Complete the frozen fitting cohort before this read-only probe")
    arrays, index, vectors, _ = load_cache(cfg, verify=True)
    index = split_index(index, cfg)
    qids, q, qs = fit_centroids(arrays, index, vectors)
    OUT.mkdir(parents=True, exist_ok=True)
    references = {}
    for kind in ("A2", "separate"):
        for seed in cfg["training"]["seeds"]:
            name = f"{kind}-full_arguments-s{seed}"
            run = ROOT/name
            meta = json.loads((run/"manifest.json").read_text())
            assert meta["status"] == "complete"
            for file, sha in meta["artifact_sha256"].items():
                assert digest(run/file) == sha
            references[name] = {"best_sha256": digest(run/"best.pt"), "manifest_sha256": digest(run/"manifest.json")}
    identity = {"code_sha256": digest(__file__), "source_protocol_sha256": digest(ROOT/"protocol.json"),
                "feature_metadata_sha256": digest(Path(cfg["data"]["cache"])/"metadata.json"),
                "references": references, "variants": list(VARIANTS)}
    path = OUT/"protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert all(old[k] == v for k, v in identity.items())
        assert digest(OUT/"centroids.npz") == old["centroids_sha256"]
    else:
        np.savez(OUT/"centroids.npz", qids=np.array(qids), question=q, question_stance=qs)
        json_write(path, {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "centroids_sha256": digest(OUT/"centroids.npz"),
            "fit_scope": "Only input messages in fitting replicas 0/1 of 16 questions; occurrence-weighted means, normalized to unit length to preserve original embedding norm.",
            "evaluation": "Familiar-question replica 3 only. All real message vectors replaced, padding stays zero; every other input and all weights stay unchanged. No unseen-question centroids are inferred from held-out data.",
            "purpose": "Does individual-message variation, beyond the average question or question/stance vector, materially affect the selected predictor?",
            "limitations": "Post-hoc input substitution, possibly outside training distribution; not a retrained baseline, not a source-LLM causal intervention. Centroids retain average argument content; removing individual variation does not remove all reasoning. Familiar questions only; no claim about unseen question performance."})
    if (OUT/"integrity.json").exists():
        completed = json.loads((OUT/"integrity.json").read_text())
        assert completed["protocol_sha256"] == digest(path)
        for key, files in completed["outputs"].items():
            name, variant = key.split("/")
            for suffix, sha in files.items():
                assert digest(OUT/name/f"{variant}_{suffix}") == sha
        print(json.dumps({"complete": True, "reused_verified_outputs": True}))
        return
    audits, outputs = [], {}
    for name in references:
        kind, _, seedstr = name.split("-")
        seed = int(seedstr[1:])
        donors = np.load(ROOT/f"donors_s{seed}.npy", allow_pickle=False)
        base = SplitSnapshots(arrays, index, vectors, "familiar", "full_arguments", donors)
        model = make_model(cfg, kind).to(device)
        model.load_state_dict(torch.load(ROOT/name/"best.pt", map_location=device, weights_only=False)["model"])
        out = OUT/name
        out.mkdir(exist_ok=True)
        for variant in VARIANTS:
            ds = CentroidSnapshots(arrays, index, vectors, donors, qids, q, qs, variant)
            assert ds.pairs == base.pairs
            for item in (0, len(ds)//2, len(ds)-1):
                first, second = base[item], ds[item]
                for key, value in first.items():
                    if key == "text":
                        continue
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, second[key], rtol=0, atol=0)
                    else:
                        assert value == second[key]
                real = second["signs"] != 0
                np.testing.assert_allclose(second["text"][real].norm(dim=-1), 1., rtol=0, atol=2e-7)
                assert not second["text"][~real].any()
            original.save_predictions(out, variant, model, ds, cfg, device)
            outputs[f"{name}/{variant}"] = {suffix: digest(out/f"{variant}_{suffix}")
                for suffix in ("predictions.parquet", "metrics.json", "by_question.csv")}
        audits.append(name)
    for name, ref in references.items():
        assert digest(ROOT/name/"best.pt") == ref["best_sha256"]
        assert digest(ROOT/name/"manifest.json") == ref["manifest_sha256"]
    json_write(OUT/"integrity.json", {"protocol_sha256": digest(path), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "fit_only_centroids": True, "all_other_input_fields_unchanged": True,
        "padding_zero_and_real_vectors_unit_norm": True, "source_checkpoints_unchanged": True,
        "model_runs": audits, "outputs": outputs})
    print(json.dumps({"complete": True, "models": len(audits), "interventions_per_model": len(VARIANTS)}))


if __name__ == "__main__":
    main()
