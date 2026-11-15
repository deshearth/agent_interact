"""Verify and report the training-only complete-prompt feasibility screen."""
from pathlib import Path
import os
import pickle
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_inbox_screen as study


def main():
    p=study.validate(); root=study.ROOT; result=study.read(root/'results.json'); enc=study.read(root/'encoding.json')
    assert result['protocol_sha256']==digest(root/'protocol.json')
    for path,sha in result['artifacts'].items(): assert digest(path)==sha,path
    labels=np.load(root/'data/labels.npz'); fit=np.isin(labels['qids'],p['inner']['fit']); hold=np.isin(labels['qids'],p['inner']['validation'])
    assert (fit^hold).all() and len(set(labels['qids']))==320
    metadata={q['qid']:q for q in study.read('runs/question_breadth_320_v1/selection.json')['questions']}
    np.testing.assert_array_equal(labels['truth'],[1 if metadata[q]['answer']=='A' else -1 for q in labels['qids']])
    assert np.unique(labels['truth'],return_counts=True)[1].tolist()==[10240,10240]
    common=np.load(study.core.state.config()['common']); single=np.load(Path(study.core.dist.config()['probability_reference'])/'probabilities.npz')['single']
    e,t=labels['pairs'].T; np.testing.assert_array_equal(labels['target'],(single[e,t+1]>.5).ravel())
    assert set(t)=={0,4}
    vectors=np.load(root/'data/vectors.npy'); f=np.load(root/'data/pca32.npz')
    np.testing.assert_array_equal(f['training_indices'],np.flatnonzero(fit)); np.testing.assert_allclose(f['mean'],vectors[fit].astype(float).mean(0),rtol=0,atol=1e-14)
    np.testing.assert_allclose(f['axes']@f['axes'].T,np.eye(32),rtol=0,atol=1e-12)
    np.testing.assert_allclose((vectors-f['mean'])@f['axes'].T,np.load(root/'data/vectors32.npy'),rtol=0,atol=6e-8)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(2); snapshot=study.encoding.paired.config()['encoder_snapshot']
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True); encoder=AutoModel.from_pretrained(snapshot,local_files_only=True).eval()
    lengths=np.load(root/'data/lengths.npy'); ids=np.unique(np.concatenate([np.arange(3),np.argsort(lengths)[-3:]]))
    texts=study.read(root/'data/texts.json'); replay,count=study.encoding.encoding.encode_block([texts[i] for i in ids],tokenizer,encoder,'cpu',study.encoding.paired.config())
    vector_error=float(np.abs(replay-vectors[ids]).max()); assert vector_error<2e-5; np.testing.assert_array_equal(count,lengths[ids])
    rows=(labels['pair_indices'][:,None]*32+np.arange(32)).ravel(); baseline=np.load(study.core.screen.ROOT/'features/inner/semantic8.npy',mmap_mode='r')[rows]
    per={}; checks=[]
    with threadpool_limits(limits=2):
        for record in result['results']:
            name=record['variant']; folder=root/'fits'/name
            if name=='state': x=baseline
            elif name=='state_truth_oracle': x=np.concatenate([baseline,labels['truth'][:,None]],-1).astype(np.float32)
            else: x=np.concatenate([baseline,np.load(root/'data'/('vectors32.npy' if name.endswith('32') else 'vectors.npy'))],-1).astype(np.float32)
            with (folder/'terminal_model.pkl').open('rb') as stream: model=pickle.load(stream)
            chosen=None; training=None
            for step,prob in enumerate(model.staged_predict_proba(x[hold]),1):
                if step==record['selected_step']: chosen=np.clip(prob[:,1],1e-10,1-1e-10); break
            for step,prob in enumerate(model.staged_predict_proba(x[fit]),1):
                if step==record['selected_step']: training=prob[:,1]; break
            np.testing.assert_array_equal(chosen,np.load(folder/'selected_probability.npy'))
            accuracy=float(((chosen>.5)==labels['target'][hold]).mean()); assert accuracy==record['inner_accuracy']
            pp=chosen; yy=labels['target'][hold]; nll=float((-yy*np.log(pp)-(1-yy)*np.log1p(-pp)).mean()); assert abs(nll-record['inner_nll'])<1e-12
            per[name]=np.array([((chosen>.5)==labels['target'][hold])[labels['qids'][hold]==q].mean() for q in sorted(p['inner']['validation'])])
            checks.append(dict(variant=name,training_accuracy_at_selected_step=float(((training>.5)==labels['target'][fit]).mean()),internal_accuracy=accuracy,selected_step=record['selected_step']))
    draws=np.random.default_rng(92913).integers(64,size=(5000,64)); comparisons=[]
    for name in study.VARIANTS[1:]:
        d=per[name]-per['state']; comparisons.append(dict(variant=name,difference=float(d.mean()),ci95=np.quantile(d[draws].mean(1),[.025,.975]).tolist()))
    gate=max(r['inner_accuracy'] for r in result['results'] if r['eligible_for_promotion'])>=next(r for r in result['results'] if r['variant']=='state')['inner_accuracy']+p['promotion_gain']
    assert gate==result['promotion_gate_passed']
    audit=dict(passed=True,completed_utc=study.now(),results_sha256=digest(root/'results.json'),model_replays=len(checks),checks=checks,comparisons=comparisons,
        cpu_encoder_replay_texts=len(ids),cpu_max_vector_difference=vector_error,source_calls=0,outer_validation_scored=False,new_validation_loaded=False,assessment_loaded=False,
        implementation_correction=p.get('implementation_correction'),audit_code_sha256=digest(__file__))
    json_write(root/'completion_audit.json',audit)
    lines=['# Complete receiver-prompt representation: internal pilot','',
        '**No candidate passed the promotion gate.** On two fixed transition rounds within the old training questions, full-prompt encodings do not improve the matched state model. '
        'These are internal 256/64 results on two rounds, not scores on all 80 validation questions and not a claim about an attainable accuracy ceiling.','',
        '| Input | Training at selected step | Internal accuracy |','| --- | ---: | ---: |']
    for row in checks: lines.append(f"| {row['variant']} | {100*row['training_accuracy_at_selected_step']:.3f}% | {100*row['internal_accuracy']:.3f}% |")
    lines+=['','The baseline already includes observed probabilities, history, question/persona context, ordered states and argument PCA8. '
        'Prompt variants add the exact receiver-facing persona/question/options/friendly/unfriendly message prompt, encoded with frozen MiniLM. '
        'The `state_truth_oracle` column supplies the correct A/B answer only as a privileged diagnostic; it is never eligible to satisfy the prediction goal.','',
        f"All {enc['texts']:,} prompts are source-hash verified. All {enc['tokens']:,} tokens were retained. "
        f"However, {enc['multiwindow_prompts']:,} prompts exceed510 tokens (maximum {enc['max_tokens']:,}); "
        'MiniLM attends only within independent windows. This does not test a model with attention across the complete long prompt.','',
        'PCA32 fits only the256 fitting questions. Every candidate uses the same booster recipe and accuracy/NLL checkpoint selection. '
        'Promotion required at least+0.5percentage points over the matched state-only pilot. No outer validation or new source calls followed this failed gate.','',
        f"Independent verification replays all4 models and6 full prompt encodings on CPU; maximum vector discrepancy is {vector_error:.2g}. "
        'The initial oracle implementation mistakenly read the numeric correct-answer value instead of the A/B answer label, producing a constant diagnostic column. '
        'That invalid oracle fit is archived and excluded; the corrected A/B oracle was refitted. All legitimate features, targets, encodings and three non-oracle fits are unchanged.','',
        'The earlier best all-round validation results remain88.51% for a neural model and88.68% for the tree on512 training questions. '
        'The next separate hypothesis tests a shared function of new and previous inboxes to predict changes in log odds.','']
    Path('reports/modal_inbox_screen_320.md').write_text('\n'.join(lines))
    print({k:v for k,v in audit.items() if k not in ('implementation_correction',)},flush=True)


if __name__=='__main__': main()
