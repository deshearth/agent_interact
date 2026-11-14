"""The scale diagnostic must alter only text magnitude and isolate its adapter."""
import pytest
import torch

from scripts import argument_response as study
from scripts.argument_features import ResponseSnapshots
from scripts.argument_scale_diagnostic import FACTOR, ScaledSnapshots, implementation, runtime
from test_argument_response import fixture_cache


def test_scale_control_keeps_stance_history_graph_and_targets_identical():
    arrays,index,vectors,donors=fixture_cache()
    for condition in ("stance_only","full_arguments","shuffled_arguments"):
        first=ResponseSnapshots(arrays,index,vectors,"train",condition,donors)
        second=ScaledSnapshots(arrays,index,vectors,"train",condition,donors)
        for i in range(len(first)):
            a,b=first[i],second[i]
            for key,value in a.items():
                if isinstance(value,torch.Tensor):
                    torch.testing.assert_close(b[key],value*FACTOR if key=="text" else value,rtol=0,atol=0)
                else: assert b[key]==value
            if condition=="stance_only":assert not b["text"].any()


def test_scale_adapter_restores_original_runner_and_separates_hashes():
    original={name:getattr(study,name) for name in ("ROOT","CONFIG","implementation","ResponseSnapshots")}
    original_digest=study.implementation_digest()
    with pytest.raises(RuntimeError):
        with runtime():
            assert study.implementation_digest()!=original_digest
            assert study.implementation==implementation
            assert study.ResponseSnapshots==ScaledSnapshots
            raise RuntimeError("Check cleanup after failure")
    for name,value in original.items():assert getattr(study,name)==value
