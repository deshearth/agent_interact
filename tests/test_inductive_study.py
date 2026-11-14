"""Checks for the study adapter and its immutable evaluation namespace."""
from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from scripts import inductive_study as study
from mac_agents import evaluate as evaluation
from mac_agents import train as training
from mac_agents.common import json_write
from mac_agents.data.features import history_features


def test_availability_control_drops_past_values_in_observed_and_generated_features():
    spins=np.array([[-1,1],[1,-1],[-1,-1]],dtype=np.float32)
    changed=spins.copy()
    changed[0]*=-1
    first=study.availability_history(spins,1,2)
    second=study.availability_history(changed,1,2)
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(first[:,2:],history_features(spins,1,2)[:,2:])
    np.testing.assert_array_equal(study.availability_history(spins,0,2),history_features(spins,0,2))
    tensor=study.mask_previous_values(torch.from_numpy(history_features(spins,1,2)))
    np.testing.assert_array_equal(tensor.numpy(),first)
    arrays={"spins":spins[None],"valid":np.ones((1,2),dtype=bool),
            "context":np.zeros((1,2,16),dtype=np.float32),"J":np.zeros((1,2,2),dtype=np.float32)}
    dataset=study.AvailabilitySnapshots(arrays,[{"question_split":"train","graph_role":"fit_graph"}],
                                       {"state":{"history_window":2}},"train","history")
    np.testing.assert_array_equal(dataset[1]["extra"].numpy(),first)


def test_control_rollout_uses_generated_current_state_and_correct_availability():
    seen=[]
    class Flip(torch.nn.Module):
        def forward(self,current,context,graph,extra):
            seen.append((current.cpu().numpy().copy(),extra.cpu().numpy().copy()))
            return torch.stack([10*current,-10*current],-1)
    s0=np.array([[-1,1]],dtype=np.float32)
    cfg={"state":{"condition":"history","history_window":2}}
    with study.runtime("availability_only"):
        result=evaluation.rollout(Flip(),s0,np.zeros((1,2,16),dtype=np.float32),
                                  np.zeros((1,2,2),dtype=np.float32),cfg,3,torch.device("cpu"))
    np.testing.assert_array_equal(result,np.stack([-s0,s0,-s0],axis=1))
    for t,(current,extra) in enumerate(seen):
        assert not extra[...,:2].any()
        assert np.all(extra[...,2]==int(t>0))
        np.testing.assert_array_equal(extra[...,4],current==1)


def test_ordinary_cli_cannot_evaluate_study_without_adapter(tmp_path):
    assert study.study_code_digest()!=study.base_code_digest()
    json_write(tmp_path/"manifest.json",{"config":{},"code_sha256":study.study_code_digest()})
    with pytest.raises(ValueError,match="code differs"):
        evaluation.evaluate(Namespace(run=tmp_path,partition="validation",mode="one_step"))
    before=(training.Snapshots,evaluation.Snapshots,evaluation.history_features,training.code_digest)
    with pytest.raises(RuntimeError):
        with study.runtime("availability_only"):
            assert evaluation.code_digest()==study.study_code_digest()
            assert evaluation.Snapshots is study.AvailabilitySnapshots
            raise RuntimeError("fixture interruption")
    assert before==(training.Snapshots,evaluation.Snapshots,evaluation.history_features,training.code_digest)


def test_study_freeze_preserves_checkpoint_and_manifest_guards(tmp_path):
    run=tmp_path/"run"
    json_write(run/"manifest.json",{"smoke":False,"stopping_reason":"baseline_fitted",
               "code_sha256":study.study_code_digest(),"kind":"I0"})
    json_write(run/"baseline.json",{"kind":"I0"})
    json_write(run/"validation_metrics.json",{"nll":0.5})
    selection=tmp_path/"selection.json"
    with study.runtime("opinion"):
        evaluation.freeze([run],selection,"Synthetic guard test")
    evaluation.check_selection(run,selection)
    json_write(run/"baseline.json",{"kind":"P0"})
    with pytest.raises(ValueError,match="checkpoint changed"):
        evaluation.check_selection(run,selection)
    json_write(run/"baseline.json",{"kind":"I0"})
    meta=json.loads((run/"manifest.json").read_text())
    meta["altered"]=True
    json_write(run/"manifest.json",meta)
    with pytest.raises(ValueError,match="changed since freezing"):
        evaluation.check_selection(run,selection)
