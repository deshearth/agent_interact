"""Keep question pairing and model-seed aggregation intact in graph evaluation."""
import numpy as np
import pytest
from scripts import evaluate_expanded_unseen_graph as study


def synthetic_records():
    records = []
    for count in (80, 320):
        for regime, graph_effect in (("seen", 0.), ("unseen", -.04)):
            for model, seeds in (("I0", [-1]), ("nano", [142, 143, 144])):
                for i in range(5):
                    for seed in seeds:
                        effect = 0 if model == "I0" else .01 + (seed - 143) * .005
                        value = .70 + i * .02 + graph_effect + effect
                        records.append({"training_questions": count, "regime": regime, "model": model,
                            "seed": seed, "qid": f"q{i}", "graph_id": "g0",
                            **{key: value for key in study.METRICS}})
    return records


def test_seed_average_and_paired_difference_are_question_level():
    qids = [f"q{i}" for i in range(5)]
    records = synthetic_records()
    rows = study.aggregate_records(records, qids)
    contrast = [(1, 320, "unseen", "nano"), (-1, 320, "unseen", "I0")]
    result = study.paired_contrast(rows, contrast, qids, 17, 1000)
    shuffled = study.aggregate_records(list(reversed(records)), qids)
    result2 = study.paired_contrast(shuffled, contrast, list(reversed(qids)), 17, 1000)
    for key in study.METRICS:
        np.testing.assert_allclose(list(result[key].values()), .01, atol=1e-14)
        np.testing.assert_allclose(list(result[key].values()), list(result2[key].values()), atol=1e-14)


def test_paired_graph_interaction_removes_shared_difficulty_shift():
    qids = [f"q{i}" for i in range(5)]
    rows = study.aggregate_records(synthetic_records(), qids)
    terms = [(1, 320, "unseen", "nano"), (-1, 320, "seen", "nano"),
             (-1, 320, "unseen", "I0"), (1, 320, "seen", "I0")]
    result = study.paired_contrast(rows, terms, qids, 17, 1000)
    for value in result.values():
        np.testing.assert_allclose(list(value.values()), 0., atol=1e-14)


def test_missing_or_duplicate_seed_cannot_change_question_weights():
    records = synthetic_records()
    qids = [f"q{i}" for i in range(5)]
    with pytest.raises(AssertionError):
        study.aggregate_records(records[:-1], qids)
    with pytest.raises(AssertionError):
        study.aggregate_records(records + [records[-1]], qids)
