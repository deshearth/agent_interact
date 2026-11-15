import numpy as np
import torch
from scripts import modal_numeric_scaling as study


def test_old_features_and_initializations_are_preserved():
    old=study.prior.Data('full');new=study.Data('numeric')
    for key in ['s','J','context','current_probability','neighbors','signs','positions','observed_history','current_round','text_numeric16','text_blank16']:
        torch.testing.assert_close(old.values[key],new.values[key][:3200],rtol=0,atol=0)
    np.testing.assert_array_equal(old.y,new.y[:3200]);np.testing.assert_array_equal(old.source,new.source[:3200])
    for name in ['numeric','blank']:
        a=study.make_model(name,142);b=study.prior.make_model(study.prior.config(),name,142)
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)


def test_training_sets_do_not_include_new_validation():
    p=study.protocol();assert len(p['train512'])==512 and len(set(p['validation']))==80
    assert set(p['train320'])<set(p['train512']) and not set(p['train512'])&set(p['validation'])
    assert not set(p['train512'])&set(study.prior.read(study.prior.ROOT/'protocol.json')['fold']['assessment_questions'])
    for m in p['models'].values():assert m['steps']['same_passes']==int(np.ceil(m['steps']['same_updates']*512/320))
