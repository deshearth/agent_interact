import copy
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path

import pytest

spec=spec_from_file_location("blind_pilot",Path("experiments/icl_forecast_v1/blind_pilot.py"))
pilot=module_from_spec(spec);spec.loader.exec_module(pilot)


def test_panel_excludes_old_validation_and_all_assessment():
    settings=pilot.read(pilot.SETTINGS)
    questions=[{"qid":f"q{i:03d}","graph_id":f"J{i%8}","split":"train" if i<80 else "validation" if i<160 else "assessment","origin":"retained" if i<100 or i>=160 else "new"} for i in range(200)]
    test=pilot.make_panel(questions,settings);train=pilot.make_panel(questions,settings,True)
    assert test==pilot.make_panel(list(reversed(questions)),settings)
    assert len(test)==20 and sum(len(q["agents"]) for q in test)==160
    assert all(100<=int(q["qid"][1:])<160 for q in test)
    assert all(int(q["qid"][1:])<80 for q in train)
    assert len({q["qid"] for q in test+train})==28


def test_export_ignores_all_future_outcomes_messages_and_correct_answers():
    episode={"statement":"Synthetic question","choices":{"A":"x","B":"y"},"personas":["synthetic"]*32,
             "J":[[0]*32 for _ in range(32)],"spins_history":[[1]*32 for _ in range(9)],
             "spins_raw_history":"PRIVATE","messages_history":"PRIVATE","answer":"A","solution":"PRIVATE"}
    item={"t":3,"split":"validation","agents":[0,4],"panel_id":"test_00"}
    first=pilot.visible_packet(episode,item)
    changed=copy.deepcopy(episode)
    for t in [0,1,2,4,5,6,7,8]:changed["spins_history"][t]=[-1]*32
    changed.update(messages_history="CHANGED",spins_raw_history="CHANGED",answer="B",solution="CHANGED",first_literal_A_probability=.999)
    assert pilot.visible_packet(changed,item)==first
    assert not any(x in str(first) for x in ["PRIVATE","solution","spins_raw_history","first_literal"])
    changed["spins_history"][3][0]=-1
    assert pilot.visible_packet(changed,item)!=first


def test_score_refuses_to_open_truth_before_predictions_commit(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,"ROOT",tmp_path)
    monkeypatch.setattr(pilot,"verify_ready",lambda:None)
    with pytest.raises(RuntimeError,match="Seal all predictions"):
        pilot.score()


def test_commit_requires_complete_probabilities_and_refuses_revisions(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,"ROOT",tmp_path)
    monkeypatch.setattr(pilot,"verify_ready",lambda:{"forecasts":2})
    pilot.write(tmp_path/"ready.json",{"synthetic":True})
    pilot.write(tmp_path/"inputs/test_inputs.json",[{"forecast_requests":[{"case_id":"a"},{"case_id":"b"}]}])
    proposal=tmp_path/"draft.json"
    pilot.write(proposal,{"a":.7})
    with pytest.raises(AssertionError):pilot.commit(proposal)
    pilot.write(proposal,{"a":.7,"b":True})
    with pytest.raises(AssertionError):pilot.commit(proposal)
    pilot.write(proposal,{"a":.7,"b":.4});pilot.commit(proposal)
    assert pilot.read(tmp_path/"predictions.json")=={"a":.7,"b":.4}
    with pytest.raises(RuntimeError,match="already sealed"):pilot.commit(proposal)
    pilot.write(tmp_path/"predictions.json",{"a":.9,"b":.6})
    with pytest.raises(AssertionError):pilot.score()
