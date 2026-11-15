"""Check the new augmentation interface against the previously audited models."""
import gc
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts import modal_scaling_study as study
from scripts import modal_scaling_data as inputs


@pytest.mark.parametrize('name',list(study.MODEL_SOURCES))
def test_original_inputs_and_predictions_preserved(name):
    if not (inputs.ROOT/'features_complete.json').exists(): pytest.skip('Requires audited local export and frozen encodings')
    data=study.Data(name); module,key=study.MODEL_SOURCES[name]
    old=module.Data('full')
    old.condition='mode'; ids=np.arange(3200); batch=old.batch(ids)
    for field,value in data.values.items():
        assert field in batch
        np.testing.assert_array_equal(value[:3200].numpy(),batch[field].numpy(),err_msg=field)
    checkpoint=Path(study.specifications()[name]['refits']['142'])/'best.pt'
    model=study.make_model(name,142); model.load_state_dict(torch.load(checkpoint,weights_only=False)['model']); model.eval()
    selected=np.array([0,7,8,39,600,1599,3199,3200,4735])
    with torch.inference_mode():
        actual=study.core.score_model(model,data.batch(selected))
        altered=data.batch(selected); altered['target']=1-altered['target']
        torch.testing.assert_close(actual,study.core.score_model(model,altered),rtol=0,atol=0)
        original=study.core.score_model(model,old.batch(selected[selected<3200]))
        torch.testing.assert_close(actual[:len(original)],original,rtol=0,atol=0)
    del data,old,batch; gc.collect()


def test_time_alignment_and_split_exclusion():
    if not (inputs.ROOT/'features_complete.json').exists(): pytest.skip('Requires completed features')
    p=study.protocol(); data=study.Data('history_arguments2306'); e,t=data.common['pairs'].T
    single=np.load(inputs.DATA/'single.npy')
    np.testing.assert_array_equal(data.current,single[e,t]); np.testing.assert_array_equal(data.source,single[e,t+1])
    np.testing.assert_array_equal(data.y,single[e,t+1]>.5)
    h=data.values['observed_history'].numpy()
    for b in (0,1,7,3200,3205,4735):
        assert not h[b,:,t[b]+1:].any()
        np.testing.assert_allclose(h[b,:,t[b],1],2*single[e[b],t[b]]-1,atol=1e-7)
    assert set(data.common['episode_qids'])==set(p['train512'])|set(p['validation'])
    assert not set(data.common['episode_qids'])&set(p['assessment'])
    assert len(set(data.qids[data.ids(p['train512'])]))==512


def test_checkpoint_budgets_and_initialization():
    p=study.protocol()
    for name,spec in p['models'].items():
        old=study.read(Path(spec['refits']['142'])/'manifest.json')
        assert spec['steps']['same_updates']==old['selected_step']
        assert 0 <= spec['steps']['same_passes']-spec['steps']['same_updates']*1.6 < 1
        model=study.make_model(name,142)
        assert study.core.dist.base.model_weights_sha(model)==old['initial_weights_sha256']
