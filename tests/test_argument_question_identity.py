import numpy as np

from mac_agents.common import config
from scripts import argument_question_identity as study
from scripts.argument_features import load_cache


def test_only_question_and_observed_stance_determine_the_extra_input():
    cfg = config(study.CONFIG)
    caches = study.prepare(cfg)
    raw,_,_,_ = load_cache(cfg,verify=True)
    for kind,(a,index,vectors,meta) in caches.items():
        size = 16 if kind == "question_id" else 32
        assert vectors.shape == (size+1,384)
        np.testing.assert_array_equal(vectors[1:,:size],np.eye(size,dtype=np.float32))
        assert not vectors[0].any() and not vectors[:,size:].any()
        np.testing.assert_array_equal(a["context"],raw["context"])
        values = {}
        for e,row in enumerate(index):
            ids = a["text_ids"][e]
            if row["diagnostic_role"] == "unseen":
                assert not ids.any()
                continue
            for stance in (-1,1):
                selected = ids[raw["stance"][e] == stance]
                if len(selected):
                    assert len(np.unique(selected)) == 1
                    key = (row["qid"],stance if kind == "question_stance_id" else 0)
                    if key in values:assert values[key] == int(selected[0])
                    values[key] = int(selected[0])
        assert len(set(values.values())) == size


def test_both_identity_conditions_use_the_original_model():
    cfg = config(study.CONFIG)
    assert all(sum(p.numel() for p in study.make_model(cfg,k).parameters()) == 185890 for k in study.VARIANTS)
