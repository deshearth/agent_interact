import copy
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path

import pytest

spec=spec_from_file_location("existing_icl",Path("experiments/icl_existing_graph_v1/pilot.py"))
pilot=module_from_spec(spec);spec.loader.exec_module(pilot)


def test_readable_test_table_does_not_export_training_outcomes_or_extras():
    packet={"panel_id":"test_00","question":"synthetic","choices":{"A":"x","B":"y"},"agents":[{"agent_id":0,"current_opinion":"A"},{"agent_id":1,"current_opinion":"B"}],"signed_edges":[{"receiver":0,"sender":1,"relationship":"unfriendly"}],"training_outcomes":["B","A"],"future":"PRIVATE"}
    first=pilot.readable(packet)
    assert first["rows"][0]=={"agent":0,"current":"A","fA":0,"fB":0,"uA":0,"uB":1}
    changed=copy.deepcopy(packet);changed["training_outcomes"]=["A","B"]
    assert pilot.readable(changed)==first
    assert "next" not in first["rows"][0] and "PRIVATE" not in str(first)


def test_scoring_cannot_access_truth_or_predictors_without_commit(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,"ROOT",tmp_path);monkeypatch.setattr(pilot,"verify",lambda:None)
    with pytest.raises(RuntimeError,match="Commit all128"):
        pilot.score()


def test_probability_metrics_use_next_majority_not_correct_option():
    m=pilot.metrics([.8,.3],[1,0])
    assert m["accuracy"]==1 and m["error"]==0
    assert m["brier"]==pytest.approx(.065)
    assert m["nll"]==pytest.approx(.2899092476)
