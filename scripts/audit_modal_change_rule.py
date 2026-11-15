"""Independent replay and internal-only report for potential-difference rules."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_change_rule as study


def predict(model,data,ids):
    model.eval(); values=[]
    with torch.inference_mode():
        for begin in range(0,len(ids),128):
            score=model.score(data.batch(ids[begin:begin+128])); values.append(torch.stack([-F.softplus(score),-F.softplus(-score)],-1).numpy())
    return np.concatenate(values)


def main():
    p=study.validate(); root=study.ROOT; result=study.read(root/'results.json')
    for path,sha in result['artifacts'].items(): assert digest(path)==sha,path
    tests=ET.parse('reports/modal_change_rule_checks.xml').getroot(); suites=list(tests) if tests.tag=='testsuites' else [tests]
    assert sum(int(s.get('tests',0)) for s in suites)==3
    assert sum(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))==0
    torch.set_num_threads(2); data=study.Data('inner'); records=[]; maximum=0.; metric_max=0.; class_differences=0
    for manifest in sorted((root/'tune').glob('*/manifest.json')):
        folder=manifest.parent; item=study.read(manifest)
        assert item['status']=='complete' and item['protocol_sha256']==digest(root/'protocol.json')
        for name,sha in item['artifacts'].items(): assert digest(folder/name)==sha,name
        model=study.make_model(study.config(),item['width'],42); model.load_state_dict(torch.load(folder/'best.pt',weights_only=False)['model'])
        for split,key,file in [('train','fit','train_logp.npy'),('internal','validation','validation_logp.npy')]:
            ids=data.ids(p['inner'][key]); saved=np.load(folder/file); actual=predict(model,data,ids)
            maximum=max(maximum,float(np.max(np.abs(actual-saved)))); class_differences+=int((actual.argmax(-1)!=saved.argmax(-1)).sum())
            np.testing.assert_allclose(actual,saved,rtol=0,atol=1e-12)
            yy=data.y[ids].astype(float); lp=actual.astype(float); pp=np.exp(lp[...,1])
            metrics=dict(error=float((lp.argmax(-1)!=yy).mean()),nll=float((-(1-yy)*lp[...,0]-yy*lp[...,1]).mean()),brier_two_class=float((2*(pp-yy)**2).mean()))
            expected=item['train' if split=='train' else 'validation']
            for k,v in metrics.items(): metric_max=max(metric_max,abs(v-expected[k]))
        rows=data.ids(p['inner']['validation'])[:2]; batch=data.batch(rows)
        with torch.inference_mode():
            before=model.score(batch); batch['target']=1-batch['target']; after=model.score(batch)
            np.testing.assert_array_equal(before.numpy(),after.numpy())
        records.append(dict(model=item['width'],candidate=item['candidate'],parameters=item['parameters'],selected_step=item['selected_step'],
            training_accuracy=1-item['train']['error'],internal_accuracy=1-item['validation']['error'],gain=float(model.gain.detach()),
            checkpoint_sha256=digest(folder/'best.pt')))
    assert len(records)==8 and metric_max<1e-7 and class_differences==0
    accuracy=max(r['internal_accuracy'] for r in records)
    assert accuracy==result['best_internal_accuracy'] and result['promotion_gate_passed']==(accuracy>=p['promotion_threshold'])
    audit=dict(passed=True,completed_utc=study.now(),results_sha256=digest(root/'results.json'),model_replays=8,prediction_arrays=16,
        class_disagreements=class_differences,max_logp_difference=maximum,max_metric_difference=metric_max,focused_tests=3,
        records=records,source_calls=0,outer_validation_scored=False,assessment_loaded=False,audit_code_sha256=digest(__file__))
    json_write(root/'completion_audit.json',audit)
    lines=['# Predicting changes in belief: internal test','',
        ('**Promotion gate passed; further all-case outer evaluation is required.**' if result['promotion_gate_passed'] else '**No candidate passed the promotion gate.**'),'',
        r'Prediction rule: $\widehat{\ell}_{t+1}=\alpha\ell_t+f(X_t)-f(X_{t-1})$, with $\ell_t=\log[p_t/(1-p_t)]$.','',
        'The shared compact attention function sees current and previous inbox information: sender states, choice or argument tokens, friendship signs, exact positions and question/persona context. '
        'The previous inbox is empty before discussion. Current probability is an observed input; next probability supplies only the modal training label. '
        'Alpha is either fixed at1 or learned. The fixed version preserves current probability when both model inbox inputs are identical.','',
        '| Input / gain | Parameters | Training accuracy | Internal accuracy | Learned gain |','| --- | ---: | ---: | ---: | ---: |']
    for name,item in result['selected'].items():
        row=next(r for r in records if r['model']==name and r['candidate']==item['candidate'])
        lines.append(f"| {name} | {row['parameters']:,} | {100*row['training_accuracy']:.3f}% | {100*row['internal_accuracy']:.3f}% | {row['gain']:.4f} |")
    lines+=['',f"The existing history-choice reference reaches {100*p['promotion_reference_accuracy']:.3f}% on the same internal64 questions; "
        f"promotion required {100*p['promotion_threshold']:.3f}%. All8 rounds and32 agents are included. "
        'Two learning rates per condition and accuracy/NLL checkpoint selection were fixed in advance. These are internal tuning results, not outer-validation estimates.','',
        'This constrains the update through one shared potential difference. A poor result rejects this tested formulation as an improvement; '
        'it does not rule out every model of belief changes or prove an accuracy ceiling. Earlier ad hoc residual corrections also did not improve training accuracy.','',
        f"All8 selected checkpoints and16 prediction arrays replay with {class_differences} class disagreements. "
        f"Maximum log-probability difference is {maximum:.2g}; independent metric discrepancy is {metric_max:.2g}. Three focused checks pass: "
        'causal previous-inbox alignment, fixed-gain identity/target independence, and nonzero gradients through both text branches.','',
        'An initial arithmetic identity test caught cancellation from evaluating `(current+new)-old`; the difference is now grouped as `current+(new-old)`. '
        'The interrupted initial fits are archived separately and excluded. No outer validation, new source calls or assessment scoring was performed.','']
    Path('reports/modal_change_rule_320.md').write_text('\n'.join(lines)); print(audit,flush=True)


if __name__=='__main__': main()
