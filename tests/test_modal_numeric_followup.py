import numpy as np
import torch
from scripts import modal_numeric_followup as study


def test_matched_initialization_and_semantic_scale():
    cfg=study.config();a=study.make_model(cfg,'numeric',42);b=study.make_model(cfg,'blank',42)
    assert sum(v.numel() for v in a.parameters())==2370
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
    data=study.Data('inner');x=data.values
    torch.testing.assert_close(x['text_numeric16'][...,:8]*np.sqrt(2),x['text_arguments8'],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(x['text_numeric16'][...,:8],x['text_blank16'][...,:8],rtol=0,atol=0)
    assert torch.count_nonzero(x['text_blank16'][...,8:])==0
    assert torch.count_nonzero(x['text_numeric16'][...,8:])>0


def test_numeric_branch_has_gradient_and_no_target_input():
    torch.set_num_threads(2);data=study.Data('inner');model=study.make_model(study.config(),'numeric',42).eval()
    with torch.no_grad():model.classifier[-1].weight.fill_(.1);model.classifier[-1].weight[1].mul_(-1)
    batch=data.batch(np.arange(2));batch['text_numeric16']=batch['text_numeric16'].clone().requires_grad_()
    z=study.core.score_model(model,batch);z.sum().backward()
    assert float(batch['text_numeric16'].grad[...,8:].abs().sum())>0
    changed=dict(batch)
    for key in ['target','target_mode','target_soft']:changed[key]=1-batch[key]
    torch.testing.assert_close(z,study.core.score_model(model,changed),rtol=0,atol=0)
