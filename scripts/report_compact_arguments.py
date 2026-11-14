"""Score frozen compact argument fits and their matched controls."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from mac_agents.models.baselines import baseline_logp
from scripts import compact_argument_study as study
from scripts import breadth_argument_study as shared
from scripts import breadth_argument_features as shared_features
from scripts import transformer_generalization as base

NAMES = ['pico_pca8', 'pico_full', 'nano_pca8', 'nano_full']

def paired_contrasts(rows, cfg):
    questions = sorted({r['qid'] for r in rows if r['split'] == 'validation'})
    grouped = {}
    for r in rows:
        if r['split'] != 'validation': continue
        for metric in ('nll', 'error', 'brier_two_class'):
            grouped.setdefault((r['model'], metric, r['qid']), []).append(r[metric])
    rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    draws = rng.integers(0,len(questions),(cfg['evaluation']['bootstrap_replicates'],len(questions)))
    pairs = []
    for name in NAMES:
        pairs.extend([(name+'_arguments', name+'_conclusion', 'arguments_vs_conclusion'),
                      (name+'_arguments', 'opinion_'+name.split('_')[0], 'arguments_vs_original'),
                      (name+'_arguments', 'I0', 'arguments_vs_Ising'),
                      (name+'_arguments', name+'_arguments_matched', 'postfit_matched_text_diagnostic')])
    pairs += [('pico_full_arguments','pico_pca8_arguments','compression'),
              ('nano_full_arguments','nano_pca8_arguments','compression'),
              ('nano_full_arguments','pico_full_arguments','backbone_width')]
    result = []
    for left,right,category in pairs:
        for metric in ('nll','error','brier_two_class'):
            delta = np.array([np.mean(grouped[left,metric,q])-np.mean(grouped[right,metric,q]) for q in questions])
            result.append({'left':left,'right':right,'category':category,'metric':metric,
                'mean_difference':float(delta.mean()),'ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                'questions':len(questions)})
    return result

def evaluate(cfg, device):
    frozen = study.validate_frozen(cfg); root = Path(cfg['output']); out = root/'results'
    if (out/'summary.json').exists(): return
    companion_cfg = shared_features.config(); companion = shared.validate_frozen(companion_cfg)
    for key in cfg['training']:
        if key != 'aggregate_minutes': assert cfg['training'][key] == companion_cfg['training'][key]
    protocol = study.read(root/'protocol.json'); out.mkdir(exist_ok=True)
    data_cfg = dict(cfg); data_cfg['models'] = {**cfg['models'], 'nano_full':{**companion_cfg['models']['nano'],'text_dim':384}}
    data = study.Data(data_cfg,device)
    seed_rows, question_rows, index, replay, sensitivity = [], [], [], [], []
    parameters = {'I0':19, 'opinion_pico':314,'opinion_nano':818}
    references = []

    def record(name,seed,split,lp,ids,origin):
        metric, rows = base.metrics(lp,data.y[ids],data.qids[ids]);metric.update(shared.additional_metrics(lp,data,ids))
        seed_rows.append({'model':name,'seed':seed,'split':split,**metric})
        question_rows.extend({'model':name,'seed':seed,'split':split,**r} for r in rows)
        file = out/f'{name}-s{seed}-{split}.npz'
        np.savez_compressed(file,logp=lp,target=data.y[ids],current=data.s[ids],qids=data.qids[ids],pairs=data.common['pairs'][ids])
        index.append({'path':str(file),'sha256':digest(file),'model':name,'seed':seed,'split':split,'origin':origin})

    planned = [(Path(p),None) for p in frozen['refits']]
    planned += [(Path(p),'nano_full') for p in companion['refits'] if study.read(Path(p)/'manifest.json')['width']=='nano']
    assert len(planned)==24
    for path,alias in planned:
        manifest = study.read(path/'manifest.json')
        width,condition,seed = alias or manifest['width'],manifest['condition'],manifest['seed']
        data.set_width(width);data.condition=condition
        model=study.make_model(data_cfg,width).to(device)
        ck=torch.load(path/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
        assert base.model_weights_sha(model)==manifest['selected_weights_sha256']
        name=width+'_'+condition;parameters[name]=manifest['parameters']
        references.append({'path':str(path),'manifest_sha256':digest(path/'manifest.json'),'width':width,'condition':condition,'seed':seed})
        for split in ('train','validation'):
            ids=data.ids(protocol['fold'][split+'_questions'])
            lp=shared.predictions(model,data,ids,cfg['training']['evaluation_batch_size'])
            record(name,seed,split,lp,ids,str(path))
            if split=='train':
                expected=np.load(path/'train_logp.npy',allow_pickle=False)
                delta=float(np.abs(np.exp(lp)-np.exp(expected)).max())
                assert delta<5e-5
                replay.append({'model':name,'seed':seed,'maximum_probability_difference':delta,
                               'class_disagreements':int((lp.argmax(-1)!=expected.argmax(-1)).sum())})
            if split=='validation' and condition=='arguments':
                swapped=shared.predictions(model,data,ids,cfg['training']['evaluation_batch_size'],'matched')
                record(name+'_matched',seed,split,swapped,ids,str(path))
                sensitivity.append({'model':name,'seed':seed,
                    'probability_mae':float(np.abs(np.exp(lp[...,1])-np.exp(swapped[...,1])).mean()),
                    'class_disagreement':float((lp.argmax(-1)!=swapped.argmax(-1)).mean())})
        print(json.dumps({'scored':name,'seed':seed}),flush=True)
    size_cfg=base.config(Path('configs/breadth_320_size_sweep.yaml'))
    for width,reference in [('pico',Path(cfg['size_reference'])),('nano',Path(cfg['reference']))]:
        paths=sorted((reference/'predictors/refit/main').glob(width+'-*/manifest.json'));assert len(paths)==3
        for path in paths:
            m=study.read(path);model=base.make_model(size_cfg,width).to(device)
            ck=torch.load(path.parent/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
            for split in ('train','validation'):
                ids=data.ids(protocol['fold'][split+'_questions'])
                record('opinion_'+width,m['seed'],split,shared.predictions(model,data,ids,cfg['training']['evaluation_batch_size'],'zero',True),ids,str(path.parent))
    ising=study.read(root/'data/ising.json')
    for split in ('train','validation'):
        ids=data.ids(protocol['fold'][split+'_questions'])
        record('I0',-1,split,baseline_logp(ising,data.s[ids],data.context[ids],data.graph[ids]),ids,str(root/'data/ising.json'))
    summary={}
    for name in sorted({r['model'] for r in seed_rows}):
        summary[name]={}
        for split in ('train','validation'):
            rows=[r for r in seed_rows if r['model']==name and r['split']==split]
            if not rows:continue
            summary[name][split]={k:float(np.mean([r[k] for r in rows])) for k in ('nll','error','brier_two_class','switch_error','stay_error')}
    old=study.read(Path(cfg['size_reference'])/'results/summary.json')
    for name,width in [('opinion_pico','pico'),('opinion_nano','nano')]:
        for split in ('train','validation'):
            for metric in ('nll','error','brier_two_class'):
                np.testing.assert_allclose(summary[name][split][metric],old['summary'][width][split][metric],atol=3e-6)
    json_write(out/'by_question.json',question_rows);json_write(out/'predictions.json',index)
    json_write(out/'summary.json',{'completed_utc':shared.now(),'summary':summary,'by_seed':seed_rows,
        'paired_differences':paired_contrasts(question_rows,cfg),'text_sensitivity':sensitivity,
        'parameters':parameters,'refit_sources':references,'cpu_reproduction':replay,
        'models_frozen_sha256':digest(root/'models_frozen.json'),
        'companion_models_frozen_sha256':digest(Path(cfg['message_reference'])/'models_frozen.json'),
        'new_fit_seconds':frozen['fit_seconds'],'new_tuning_fits':frozen['tuning_fits'],'new_refits':frozen['refits_completed'],
        'reused_companion_refits':6,'reused_opinion_refits':6,'reused_Ising_fits':1,
        'pca_explained_variance_fraction':protocol['pca_explained_variance_fraction'],
        'shuffle_audit':protocol['shuffle_audit'],'encoder_parameters':22713216,
        'assessment_loaded':False,'source_llm_calls':0,'evaluation_device':str(device),
        'scope':'Exploratory reused 320/80 development split; delivered-message response prediction. Frozen encoder; no autonomous rollout.'})
    print(json.dumps({'evaluation':'complete','summary':summary}),flush=True)

if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    cfg=study.config();torch.set_num_threads(cfg['training']['torch_threads']);evaluate(cfg,args.device)
