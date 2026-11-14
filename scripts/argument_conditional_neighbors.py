"""Within-question text similarity after exact matching on nontext model inputs.

This is a narrow diagnostic on familiar questions with repeated observable states,
not a competitor on the complete prediction task or a test of causal influence.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config, digest, json_write
from mac_agents.metrics import paired_bootstrap
from scripts import argument_mechanisms as previous
from scripts import joint_argument_encoding as session
from scripts import joint_argument_encoding_windowed as encoding
from scripts.argument_features import load_cache

ROOT = session.SESSION/"conditional_neighbors"
ALPHAS = (0., .5, 2., 8.)
BANDWIDTHS = (.01, .03, .1, .3, 1., 3., 10., float("inf"))


def metrics(frame, p):
    truth = (frame.target.to_numpy() == 1).astype(float)
    p = np.clip(p, 1e-6, 1-1e-6)
    temp = pd.DataFrame({"qid": frame.qid.to_numpy(), "nll": -(truth*np.log(p)+(1-truth)*np.log1p(-p)),
                         "error": ((p >= .5) != truth).astype(float)})
    return temp.groupby("qid", sort=True)[["nll", "error"]].mean()


def main():
    cfg = config("configs/argument_mechanisms.yaml")
    arrays, index, original, metadata = load_cache(cfg, verify=True)
    index = previous.split_index(index, cfg)
    keys = np.load(session.AUDIT/"input_keys.npz", allow_pickle=False)["history"].reshape(-1)
    ids = arrays["text_ids"].reshape(-1, arrays["text_ids"].shape[-1])
    targets = arrays["spins"][:, 1:].reshape(-1)
    fitting = np.array([r["diagnostic_role"] == "train" for r in index]).repeat(8*32)
    groups = defaultdict(list)
    for row in np.flatnonzero(fitting):
        groups[keys[row]].append(int(row))
    groups = {key: np.array(value) for key, value in groups.items()}
    priors = {}
    for qid in sorted({r["qid"] for r in index if r["diagnostic_role"] == "train"}):
        es = [e for e, r in enumerate(index) if r["qid"] == qid and r["diagnostic_role"] == "train"]
        y = arrays["spins"][es, 1:] == 1
        priors[qid] = (y.sum((0, 1))+1)/(y.shape[0]*y.shape[1]+2)
    records = []
    for e, r in enumerate(index):
        if r["diagnostic_role"] not in ("validation", "familiar"):
            continue
        for t in range(8):
            for receiver in range(32):
                row = (e*8+t)*32+receiver
                train = groups.get(keys[row], np.array([], dtype=int))
                positives = int((targets[train] == 1).sum())
                records.append({"row": row, "role": r["diagnostic_role"], "qid": r["qid"],
                    "episode_id": r["episode_id"], "round": t, "agent": receiver,
                    "target": int(targets[row]), "n_fit": len(train), "n_fit_positive": positives,
                    "conflicting": 0 < positives < len(train), "prior": float(priors[r["qid"]][receiver])})
    frame = pd.DataFrame(records)
    complete = json.loads((encoding.ROOT/"complete.json").read_text())
    for file, sha in complete["artifacts"].items():
        assert digest(encoding.ROOT/file) == sha
    with np.load(session.ROOT/"ids.npz", allow_pickle=False) as mapping:
        new_ids = mapping["text_ids"].reshape(ids.shape)
        np.testing.assert_array_equal(mapping["source_text_ids"][new_ids], ids)
    variants = {"original": (ids, original), **{m: (new_ids, np.load(encoding.ROOT/f"{m}.npy", mmap_mode="r", allow_pickle=False)) for m in encoding.MODES}}
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"code_sha256": digest(__file__), "input_keys_sha256": digest(session.AUDIT/"input_keys.npz"),
        "encoding_complete_sha256": digest(encoding.ROOT/"complete.json"), "alphas": list(ALPHAS),
        "bandwidth_multipliers": [str(x) for x in BANDWIDTHS], "roles": ["validation", "familiar"]}
    if (ROOT/"protocol.json").exists():
        saved = json.loads((ROOT/"protocol.json").read_text())
        assert all(saved[k] == v for k, v in identity.items())
    else:
        json_write(ROOT/"protocol.json", {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Is message similarity predictive within exactly repeated nontext input states, without optimizing a neural text branch?",
            "matching": "Same question/persona, signed neighbors and their degrees, current and previous receiver/neighbor opinions, history availability. Match keys come from the frozen input audit. Fit replicas 0/1 only; selection replica 2; evaluate replica 3.",
            "distance": "Mean squared vector distance over corresponding incoming edge slots. Training-only positive pair-distance median sets scale. RBF weights normalized to the fitting group count; alpha shrinks toward a fitting question/receiver prior.",
            "selection": "Choose bandwidth and alpha by question-mean NLL on selection examples whose fitting group contains both targets. The uniform-weight baseline tunes alpha on the same examples. Evaluation targets are unused until final reporting. All chosen settings and candidate results are retained.",
            "limitations": "A small, selected support subset of familiar questions; no unseen-question transfer assessment. Repeated states and targets are correlated within conversations. Similarity need not correspond to valid reasoning. Results are exploratory, not causal or a fresh confirmatory test."})
    fit_positive_pairs = []
    rng = np.random.default_rng(735)
    for rows in groups.values():
        if len(rows)>1:
            pair = rng.choice(rows, 2, replace=False)
            fit_positive_pairs.append(pair)
    fit_positive_pairs = np.array(fit_positive_pairs)
    if len(fit_positive_pairs)>5000:
        fit_positive_pairs = fit_positive_pairs[rng.choice(len(fit_positive_pairs), 5000, replace=False)]
    query_groups = defaultdict(list)
    for j, row in enumerate(frame.row):
        if keys[row] in groups:
            query_groups[keys[row]].append(j)
    prior = frame.prior.to_numpy()
    count = frame.n_fit.to_numpy()
    pos = frame.n_fit_positive.to_numpy()
    val = (frame.role == "validation").to_numpy() & frame.conflicting.to_numpy()
    candidates, selected, qrows = [], [], []
    prediction_output = frame.copy()
    for variant, (feature_ids, vectors) in variants.items():
        distances = []
        for p, q in fit_positive_pairs:
            x = vectors[feature_ids[p]]-vectors[feature_ids[q]]
            active = max(1, int((feature_ids[p] != 0).sum()))
            d = float(np.square(x).sum()/active)
            if d>1e-12:
                distances.append(d)
        scale = float(np.median(distances))
        assert scale>0
        fractions = np.broadcast_to(prior[:, None], (len(frame), len(BANDWIDTHS))).copy()
        for key, js in query_groups.items():
            train = groups[key]
            js = np.array(js)
            rows = frame.row.to_numpy()[js]
            tx = np.asarray(vectors[feature_ids[train]], dtype=np.float32).reshape(len(train), -1)
            qx = np.asarray(vectors[feature_ids[rows]], dtype=np.float32).reshape(len(rows), -1)
            active = max(1, int((feature_ids[train[0]] != 0).sum()))
            d = np.maximum(0., (np.square(qx).sum(-1)[:, None]+np.square(tx).sum(-1)[None]-2*qx@tx.T)/active)
            y = (targets[train] == 1).astype(float)
            for k, multiplier in enumerate(BANDWIDTHS):
                if np.isinf(multiplier):
                    fractions[js, k] = y.mean()
                else:
                    w = np.exp(-(d-d.min(-1, keepdims=True))/(scale*multiplier))
                    fractions[js, k] = (w@y)/w.sum(-1)
        options = []
        for k, multiplier in enumerate(BANDWIDTHS):
            for alpha in ALPHAS:
                denominator = count+alpha
                prediction = np.where(count>0, (count*fractions[:, k]+alpha*prior)/np.maximum(denominator, 1e-12), prior)
                nll = float(metrics(frame[val], prediction[val]).nll.mean())
                row = {"variant": variant, "multiplier": str(multiplier), "alpha": alpha, "scale": scale,
                       "selection_conflicting_nll": nll, "column": k}
                candidates.append(row)
                options.append((nll, k, alpha, prediction, row))
        choices = [(variant, min(options, key=lambda x: x[0]))]
        if variant == "original":
            choices.insert(0, ("uniform", min([o for o in options if np.isinf(BANDWIDTHS[o[1]])], key=lambda x: x[0])))
        for label, best in choices:
            _, k, alpha, prediction, row = best
            selected.append({**row, "variant": label})
            prediction_output[f"p_{label}"] = prediction
            for role in ("validation", "familiar"):
                for support, mask in (("matched", count>0), ("conflicting", frame.conflicting.to_numpy())):
                    mask = mask & (frame.role == role).to_numpy()
                    q = metrics(frame[mask], prediction[mask])
                    qrows.extend(q.reset_index().assign(variant=label, role=role, support=support, examples=int(mask.sum())).to_dict("records"))
        print(json.dumps({"variant": variant, "selected": selected[-1], "matched_familiar": int(((frame.role == "familiar") & (frame.n_fit>0)).sum())}), flush=True)
    prediction_output.to_parquet(ROOT/"predictions.parquet", index=False)
    pd.DataFrame(candidates).to_csv(ROOT/"selection_candidates.csv", index=False)
    json_write(ROOT/"selected.json", selected)
    questions = pd.DataFrame(qrows)
    questions.to_csv(ROOT/"by_question.csv", index=False)
    summary = questions.groupby(["variant", "role", "support"], sort=True)[["nll", "error", "examples"]].mean().reset_index()
    summary.to_csv(ROOT/"summary.csv", index=False)
    paired = {}
    q = questions.set_index(["variant", "role", "support", "qid"])
    for variant in variants:
        for support in ("matched", "conflicting"):
            paired[f"{variant}/{support}"] = {m: paired_bootstrap(q.loc[(variant, "familiar", support)][m], q.loc[("uniform", "familiar", support)][m], 1000, 1729) for m in ("nll", "error")}
    json_write(ROOT/"paired.json", paired)
    files = ("predictions.parquet", "selection_candidates.csv", "selected.json", "by_question.csv", "summary.csv", "paired.json")
    json_write(ROOT/"complete.json", {"protocol_sha256": digest(ROOT/"protocol.json"), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {file: digest(ROOT/file) for file in files}, "support_counts": frame.groupby("role").agg(total=("row", "size"), matched=("n_fit", lambda x: int((x>0).sum())), conflicting=("conflicting", "sum")).to_dict("index")})


if __name__ == "__main__":
    main()
