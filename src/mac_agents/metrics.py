from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_frame(logp, episode, rounds, arrays, index, kind):
    """One row per agent transition; stable keys survive batching and sorting."""
    rows = []
    probabilities = np.exp(logp)
    for b, (e, t) in enumerate(zip(episode, rounds)):
        meta = index[int(e)]
        for agent in range(arrays["spins"].shape[-1]):
            rows.append({"episode_id": meta["episode_id"], "qid": meta["qid"],
                         "graph_hash": meta["graph_hash"], "graph_family": meta["graph_family"],
                         "graph_role": meta["graph_role"], "question_split": meta["question_split"],
                         "agent": agent, "round": int(t), "current_state": int(arrays["spins"][e, t, agent]),
                         "target": int(arrays["spins"][e, t+1, agent]),
                         "p_positive": float(probabilities[b, agent, 1]),
                         "logp_negative": float(logp[b, agent, 0]) if kind != "P0" else np.nan,
                         "logp_positive": float(logp[b, agent, 1]) if kind != "P0" else np.nan,
                         "prediction": int(1 if probabilities[b, agent, 1] > probabilities[b, agent, 0] else -1)})
    return pd.DataFrame(rows)


def score(frame, kind="A2"):
    by_question = []
    for qid, group in frame.groupby("qid", sort=True):
        y = group.target.to_numpy()
        correct = group.prediction.to_numpy() == y
        previous = group.current_state.to_numpy()
        p = group.p_positive.to_numpy()
        fcba = [correct[(previous == a) & (y == b)].mean()
                for a in (-1, 1) for b in (-1, 1) if ((previous == a) & (y == b)).any()]
        flip = previous != y
        selected_logp = np.where(y == 1, group.logp_positive.to_numpy(), group.logp_negative.to_numpy())
        by_question.append({"qid": qid, "count": len(group),
                            "nll": float(-selected_logp.mean()) if kind != "P0" else None,
                            "brier": float((2*(p-(y == 1))**2).mean()) if kind != "P0" else None,
                            "accuracy": float(correct.mean()), "fcba": float(np.mean(fcba)),
                            "flip_accuracy": float(correct[flip].mean()) if flip.any() else None,
                            "stay_accuracy": float(correct[~flip].mean()) if (~flip).any() else None})
    qframe = pd.DataFrame(by_question)
    result = {"questions": len(qframe), "agent_transitions": len(frame),
              "aggregation": "equal weight per question; multiclass Brier is sum over two classes",
              "likelihood_clipping": "none; neural log_softmax and baseline log-sigmoid used"}
    for key in ("nll", "brier", "accuracy", "fcba", "flip_accuracy", "stay_accuracy"):
        values = pd.to_numeric(qframe[key], errors="coerce").dropna()
        result[key] = float(values.mean()) if len(values) else None
    return result, qframe


def paired_bootstrap(first, second, replicates=1000, seed=1729):
    """Input question-level paired scores, never agent-level bootstrap samples."""
    a, b = first.align(second, join="inner")
    differences = (a-b).dropna().to_numpy()
    if not len(differences):
        return None
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (replicates, len(differences)))].mean(1)
    return {"paired_questions": len(differences), "difference": float(differences.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(), "replicates": replicates}
