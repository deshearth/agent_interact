"""Read-only audits and scientific summaries for the bounded argument investigation."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from mac_agents.metrics import score, paired_bootstrap
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import joint_argument_context as joint
from scripts import argument_order_diagnostic as order
from scripts import argument_residual_diagnostic as residual
from scripts import argument_residual_original_context as residual_pca3
from scripts import argument_question_identity as identity_control
from scripts import joint_argument_encoding as session
from scripts import joint_argument_encoding_windowed as encoding
from scripts.argument_features import load_cache

PREFIX = Path("reports/argument_investigation")
KEYS = ["episode_id", "round", "agent"]
METRICS = ["nll", "accuracy", "brier", "fcba", "flip_accuracy", "stay_accuracy"]


def geometry():
    path = Path(f"{PREFIX}_geometry.json")
    identity = {"method_sha256": digest(previous.__file__),
        "encoding_complete_sha256": digest(encoding.ROOT/"complete.json"),
        "features_sha256": digest("data/processed/argument_response_v1/metadata.json")}
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == v for k, v in identity.items())
        return saved
    cfg = config(question.CONFIG)
    arrays, index, vectors, _ = load_cache(cfg, verify=True)
    index = previous.split_index(index, cfg)
    result = {"original": previous.geometry(arrays, index, vectors)}
    with np.load(session.ROOT/"ids.npz", allow_pickle=False) as mapping:
        adapted = {**arrays, "text_ids": mapping["text_ids"]}
    for mode in encoding.MODES:
        vec = np.load(encoding.ROOT/f"{mode}.npy", mmap_mode="r", allow_pickle=False)
        result[mode] = previous.geometry(adapted, index, vec)
    saved = {**identity, "variants": result}
    json_write(path, saved)
    return saved


def diagnostics():
    for folder, manifest_name in ((session.AUDIT, "integrity.json"), (encoding.ROOT, "complete.json"),
                                  (session.SESSION/"conditional_neighbors", "complete.json")):
        saved = json.loads((folder/manifest_name).read_text())
        for file, sha in saved["artifacts"].items():
            assert digest(folder/file) == sha
    capacity = session.SESSION/"pair_capacity"
    protocol = json.loads((capacity/"protocol.json").read_text())
    assert protocol["code_sha256"] == digest("scripts/argument_pair_capacity.py")
    rows = json.loads((capacity/"summary.json").read_text())
    pairs = np.load(capacity/"selected_pairs.npy", allow_pickle=False)
    targets = (pairs.reshape(-1, 4)[:, -1] == 1).astype(np.int64)
    assert len(rows) == 6
    for r in rows:
        assert r["protocol_sha256"] == digest(capacity/"protocol.json")
        out = capacity/f"scale{int(r['scale'])}-s{r['seed']}"
        for file, sha in r["artifacts"].items():
            assert digest(out/file) == sha
        logits = np.load(out/"logits.npy", allow_pickle=False)
        np.testing.assert_allclose((logits.argmax(-1) != targets).mean(), r["error"], atol=1e-12)
    pd.DataFrame([{k:v for k,v in r.items() if k != "artifacts"} for r in rows]).to_csv(f"{PREFIX}_capacity.csv", index=False)
    arrays, index, _, _ = load_cache(config(question.CONFIG), verify=True)
    index = previous.split_index(index, config(question.CONFIG))
    ep = {r["episode_id"]: e for e, r in enumerate(index)}
    nn = pd.read_parquet(session.SESSION/"conditional_neighbors/predictions.parquet")
    nn["degree"] = [int((arrays["signs"][ep[e], j] != 0).sum()) for e, j in zip(nn.episode_id, nn.agent)]
    counts = nn.groupby(["role", "conflicting", nn.degree.eq(0).rename("isolated")]).size().rename("examples").reset_index()
    counts.to_csv(f"{PREFIX}_conditional_support.csv", index=False)
    # Empirical collision floors decomposed by whether the receiver has messages.
    with np.load(session.AUDIT/"input_keys.npz", allow_pickle=False) as inputs:
        floors = []
        for role in ("train", "validation", "familiar", "unseen"):
            es = [e for e,r in enumerate(index) if r["diagnostic_role"] == role]
            degree = np.broadcast_to((arrays["signs"][es] != 0).sum(-1)[:, None], (len(es), 8, 32))
            y = (arrays["spins"][es, 1:] == 1)
            for subset, mask in (("all", np.ones_like(y, dtype=bool)), ("receives_messages", degree>0), ("isolated", degree == 0)):
                for kind in ("opinion", "history", "history_text", "prompt_message_bag", "released_prompt_order"):
                    keys = inputs[kind][es][mask]
                    _, inv, count = np.unique(keys, return_inverse=True, return_counts=True)
                    positive = np.bincount(inv, weights=y[mask].astype(float), minlength=len(count))
                    mistakes = int(np.minimum(positive, count-positive).sum())
                    floors.append({"role": role, "subset": subset, "input": kind, "targets": len(keys),
                        "minimum_mistakes": mistakes, "empirical_minimum_error": mistakes/len(keys)})
        pd.DataFrame(floors).to_csv(f"{PREFIX}_collision_breakdown.csv", index=False)
    return rows


def model_metrics(require_complete=False):
    cfg = config(question.CONFIG)
    arrays, index, _, _ = load_cache(cfg, verify=True)
    index = previous.split_index(index, cfg)
    ep = {r["episode_id"]: e for e,r in enumerate(index)}
    items = [("original", "full_embedding", c, s, question.ROOT, question, ("train", "validation", "familiar", "unseen"))
             for s in (42,43,44) for c in ("stance_only", "full_arguments", "shuffled_arguments")]
    items += [("original_pca3", "A2", c, s, previous.ROOT, previous, ("train", "validation", "familiar", "unseen"))
              for s in (42,43,44) for c in ("stance_only", "full_arguments", "shuffled_arguments")]
    completed = {}
    for module in (joint, order, residual, residual_pca3, identity_control):
        plan_path = module.ROOT/"plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        ready = plan.get("complete", False)
        completed[module.ROOT.name] = ready
        if require_complete:
            assert ready, f"Incomplete: {module.ROOT}"
        if not ready:
            continue
        assert plan["protocol_sha256"] == digest(module.ROOT/"protocol.json")
        roles = ("train", "validation", "familiar") if module in (residual, residual_pca3, identity_control) else ("train", "validation", "familiar", "unseen")
        for r in plan["runs"]:
            assert r["status"] == "complete"
            items.append((r["kind"], r["kind"], r["condition"], r["seed"], module.ROOT, module, roles))
    truth_references, rows, qrows, sensitivity, provenance = {}, [], [], [], {}
    for representation, kind, condition, seed, root, module, roles in items:
        name = f"{kind}-{condition}-s{seed}"
        run, out = root/name, root/"evaluation"/name
        manifest = json.loads((run/"manifest.json").read_text())
        assert manifest["status"] == "complete" and not manifest["smoke"]
        assert manifest["protocol_sha256"] == digest(root/"protocol.json")
        assert manifest["config_sha256"] == digest(module.CONFIG)
        impl = hashlib.sha256(json.dumps(module.implementation(), sort_keys=True).encode()).hexdigest()
        assert manifest["implementation_sha256"] == impl
        assert manifest["train_snapshots"] == 2048 and manifest["validation_snapshots"] == 1024
        for file, sha in manifest["artifact_sha256"].items():
            assert digest(run/file) == sha
        ev = json.loads((out/"identity.json").read_text())
        assert ev["run_manifest_sha256"] == digest(run/"manifest.json")
        assert ev["best_sha256"] == digest(run/"best.pt")
        assert ev["protocol_sha256"] == digest(root/"protocol.json")
        for file, sha in ev["artifacts"].items():
            assert digest(out/file) == sha
        provenance[name] = {"manifest_sha256": digest(run/"manifest.json"), "evaluation_sha256": digest(out/"identity.json")}
        label = {"representation": representation, "condition": condition, "seed": seed}
        for role in roles:
            where = run if role == "validation" else out
            frame = pd.read_parquet(where/f"{role}_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
            assert not frame.duplicated(KEYS).any()
            truth = frame[KEYS+["qid", "current_state", "target"]]
            if role in truth_references:
                pd.testing.assert_frame_equal(truth_references[role], truth)
            else:
                truth_references[role] = truth
                es = frame.episode_id.map(ep).to_numpy()
                ts, agents = frame["round"].to_numpy(), frame.agent.to_numpy()
                assert {index[e]["diagnostic_role"] for e in es} == {role}
                np.testing.assert_array_equal(frame.target, arrays["spins"][es, ts+1, agents])
                np.testing.assert_array_equal(frame.current_state, arrays["spins"][es, ts, agents])
            metrics, questions = score(frame)
            saved = json.loads((where/f"{role}_metrics.json").read_text())
            for metric in METRICS:
                np.testing.assert_allclose(metrics[metric], saved[metric], rtol=0, atol=1e-12)
            rows.append({**label, "role": role, **metrics, "error": 1-metrics["accuracy"],
                "parameters": manifest["parameters"], "steps": manifest["steps"], "best_step": manifest["best_step"]})
            qrows.extend(questions.assign(**label, role=role).to_dict("records"))
            if condition == "full_arguments" and role in ("familiar", "unseen") and (where/f"{role}_substitution_predictions.parquet").exists():
                changed = pd.read_parquet(where/f"{role}_substitution_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
                pd.testing.assert_frame_equal(truth, changed[KEYS+["qid", "current_state", "target"]])
                for qid, group in frame.groupby("qid", sort=True):
                    alt = changed.loc[group.index]
                    m,_ = score(group); n,_ = score(alt)
                    sensitivity.append({**label, "role": role, "qid": qid, "nll_change": n["nll"]-m["nll"],
                        "accuracy_change": n["accuracy"]-m["accuracy"], "changed_choices": float((group.prediction != alt.prediction).mean())})
    runs, questions = pd.DataFrame(rows), pd.DataFrame(qrows)
    runs.to_csv(f"{PREFIX}_runs.csv", index=False)
    questions.to_csv(f"{PREFIX}_by_question.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(f"{PREFIX}_sensitivity.csv", index=False)
    means = runs.groupby(["role", "representation", "condition"], sort=True)[METRICS+["error"]].mean().reset_index()
    means.to_csv(f"{PREFIX}_summary.csv", index=False)
    qm = questions.groupby(["role", "representation", "condition", "qid"], sort=True)[METRICS].mean()
    contrasts = {}
    def contrast(name, a, b):
        contrasts[name] = {m: paired_bootstrap(a[m], b[m], 1000, 1729) for m in ("nll", "accuracy")}
    for role in ("train", "validation", "familiar", "unseen"):
        available = set(means[means.role == role].representation)
        for rep in available:
            if rep in ("original", "original_pca3"):
                continue
            reference = "original_pca3" if rep.startswith("pca3_") or rep in identity_control.VARIANTS else "original"
            contrast(f"{role}/{rep}: full versus original full", qm.loc[(role, rep, "full_arguments")], qm.loc[(role, reference, "full_arguments")])
            contrast(f"{role}/{rep}: full versus original stance", qm.loc[(role, rep, "full_arguments")], qm.loc[(role, reference, "stance_only")])
        if "question_conditioned" in available:
            for condition in joint.CONDITIONS:
                contrast(f"{role}/{condition}: question_conditioned minus message_only", qm.loc[(role, "question_conditioned", condition)], qm.loc[(role, "message_only", condition)])
        if "released_order" in available:
            for condition in ("stance_only", "full_arguments", "shuffled_arguments"):
                contrast(f"{role}/{condition}: released_order minus random_order", qm.loc[(role, "released_order", condition)], qm.loc[(role, "random_order", condition)])
    json_write(f"{PREFIX}_paired.json", contrasts)
    json_write(f"{PREFIX}_integrity.json", {"analysis_code_sha256": digest(__file__), "completed_studies": completed,
        "models_audited": len(items), "new_models": len(items)-18, "frozen_references": 18,
        "identical_targets_and_cohorts": True, "scores_recomputed": True, "provenance": provenance})
    return runs, means, contrasts


def plot(runs):
    labels = {"original": "Original text", "message_only": "Matched message-only", "question_conditioned": "Question-conditioned",
              "released_order": "Reconstructed order", "random_order": "Random order", "question_mean": "Question mean only",
              "question_stance_mean": "Question/stance mean", "individual_residual": "Individual residual"}
    reps = [r for r in labels if r in set(runs.representation)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for ax, role in zip(axes, ("train", "familiar", "unseen")):
        base = runs[(runs.role == role)&(runs.representation == "original")&(runs.condition == "stance_only")].error.mean()*100
        ax.axvline(base, color="#696969", linestyle="--", linewidth=1, label="Stance/history reference")
        for y, rep in enumerate(reps):
            values = runs[(runs.role == role)&(runs.representation == rep)&(runs.condition == "full_arguments")].error.to_numpy()*100
            if len(values):
                ax.scatter(values, np.repeat(y, len(values)), alpha=.3, s=25, color="#176e61")
                ax.scatter(values.mean(), y, s=65, marker="D", color="#176e61", zorder=3)
        ax.set_yticks(range(len(reps)), [labels[r] for r in reps])
        ax.set_xlabel("Prediction error (%) — lower is better")
        ax.set_title({"train":"Fitting conversations", "familiar":"16 familiar questions", "unseen":"4 unseen development questions"}[role])
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Why have individual arguments added little predictive value?", fontsize=15)
    fig.text(.5, .025, "Diamonds: three-seed means. Faint points: seeds. Dashed: shared no-text reference. Order is reconstructed, not logged.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.94])
    fig.savefig(f"{PREFIX}.png", dpi=180); fig.savefig(f"{PREFIX}.pdf"); plt.close(fig)
    pca_reps = ["original_pca3", *residual_pca3.VARIANTS]
    if all(r in set(runs.representation) for r in pca_reps):
        labels = ["Full individual arguments", "Question mean only", "Question/stance mean", "Individual residual"]
        if all(r in set(runs.representation) for r in identity_control.VARIANTS):
            pca_reps += list(identity_control.VARIANTS)
            labels += ["Question ID only", "Question/stance ID only"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.1), sharey=True)
        for ax, role in zip(axes, ("train", "familiar")):
            base = runs[(runs.role == role)&(runs.representation == "original_pca3")&(runs.condition == "stance_only")].error.mean()*100
            ax.axvline(base, color="#696969", linestyle="--", linewidth=1)
            for y, rep in enumerate(pca_reps):
                values = runs[(runs.role == role)&(runs.representation == rep)&(runs.condition == "full_arguments")].error.to_numpy()*100
                ax.scatter(values, np.repeat(y, len(values)), alpha=.3, s=25, color="#4d689c")
                ax.scatter(values.mean(), y, s=65, marker="D", color="#4d689c")
            ax.set_yticks(range(len(pca_reps)), labels)
            ax.set_xlabel("Prediction error (%) — lower is better")
            ax.set_title("Fitting conversations" if role == "train" else "Held-out conversations, 16 familiar questions")
            ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.2)
        axes[0].invert_yaxis()
        fig.suptitle("Does the earlier text gain require individual argument variation?")
        fig.text(.5, .035, "Original PCA3 architecture, 185,890 parameters. Dashed: no-text baseline. Centroids fit on training messages only.", ha="center", fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.93])
        fig.savefig(f"{PREFIX}_residual.png", dpi=180); fig.savefig(f"{PREFIX}_residual.pdf"); plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("geometry", "metrics", "all"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.stage in ("geometry", "all"):
        geometry()
    if args.stage in ("metrics", "all"):
        diagnostics()
        runs, means, paired = model_metrics(args.require_complete)
        plot(runs)
        print(means[(means.role.isin(["train", "familiar", "unseen"])) & (means.condition == "full_arguments")][["role", "representation", "error", "nll"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
