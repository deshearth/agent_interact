"""Report the completed, independently audited 320-question argument comparison."""
from pathlib import Path
import json
import os
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import breadth_argument_features as features

LABELS={'I0':'Ising reference','opinion_nano':'Opinion/context graph transformer',
        'nano_conclusion':'Compact + conclusion-only text','nano_arguments':'Compact + delivered arguments',
        'opinion_full':'Larger opinion/context transformer','full_conclusion':'Larger + conclusion-only text',
        'full_arguments':'Larger + delivered arguments'}
ORDER=list(LABELS)

def main():
    cfg=features.config();root=Path(cfg['output']);r=json.loads((root/'results/summary.json').read_text())
    audit=json.loads((root/'completion_audit.json').read_text());assert audit['status']=='passed'
    assert audit['results_artifacts'][str(root/'results/summary.json')]==digest(root/'results/summary.json')
    protocol=json.loads((root/'protocol.json').read_text());selection=json.loads((root/'selection.json').read_text())
    geometry=json.loads((root/'representation_audit.json').read_text())
    def contrast(width,metric):return next(x for x in r['paired_differences'] if x['left']==width+'_arguments' and x['right']==width+'_conclusion' and x['metric']==metric)
    findings=[]
    for width in cfg['models']:
        d=contrast(width,'error');n=contrast(width,'nll');lo,hi=d['ci95']
        findings.append(f"For **{r['parameters'][width+'_arguments']:,} parameters**, delivered arguments change validation accuracy by **{-100*d['mean_difference']:+.2f} percentage points** (descriptive paired 95% interval **{-100*hi:+.2f} to {-100*lo:+.2f}**) versus the conclusion-only control. NLL changes by **{n['mean_difference']:+.5f}** (interval {n['ci95'][0]:+.5f} to {n['ci95'][1]:+.5f}; negative is better).")
    lines=['# Delivered arguments on 320 Llama training questions','',*sum(([p,''] for p in findings),[]),
        'This is the first argument-aware comparison on the expanded 320/80 corpus. It is separate from the earlier 818-parameter opinion-only size and question-count sweeps. Scores predict the next sampled majority-of-five opinion, not mathematical correctness. The same 80 validation questions select settings, so these are exploratory development results. The 40 reserved assessment conversations were not generated, loaded, encoded or scored.','',
        '## Matched data and inputs','',
        'All models use 320 training questions (81,920 agent transitions) and the same 80 validation questions (20,480 transitions). Entire questions stay within their existing partitions. The exact previous training-only question PCA, known-persona PCA, and 16-component context are reused. Current opinions, prescribed signed graph and fixed persona population are shared. Each receiver consumes its neighbors’ current broadcasts through the prescribed graph. Arguments, when present, are observed after delivery for the target update and before target votes. The original opinion-only models and Ising therefore forecast from an earlier information boundary. No autonomous rollout is attempted.','',
        'The compact argument model extends the 818-weight network with a learned bias-free 384-to-8 projection (3,890 parameters total). The larger model extends the 135,458-weight network with a 384-to-128 projection (184,610 total). Projected text is added to normalized sender state for attention keys and values; receiver queries and local state are unchanged. Unit-length input text vectors are scaled by sqrt(384), equally in both conditions, to avoid an unnecessarily small initial text activation. The pretrained 22,713,216-weight MiniLM encoder remains frozen and is excluded from these trainable counts. There is no PCA on arguments and no encoder fine-tuning.','',
        'The matched conclusion-only control receives the fixed string “I favor option A.” or “I favor option B.” determined solely by the sender’s current opinion. Both conditions predict the same original outcomes generated with full messages; the source conversations are never altered. The control shares all parameter shapes, initialization, learning-rate search, optimizer and stopping policy with its argument counterpart. Its text projection is active, although the two-value input has much lower rank; equal nominal parameter count is not equal effective statistical capacity. The original 818 and 135,458 opinion-only checkpoints and Ising coefficients are reproduced without refitting.','',
        '## Accuracy, probability loss and fitting','',
        '| Predictor | Trainable parameters | Training accuracy | Validation accuracy | Validation NLL | Validation two-class Brier |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in ORDER:
        train,val=r['summary'][key]['train'],r['summary'][key]['validation']
        lines.append(f"| {LABELS[key]} | {r['parameters'][key]:,} | {100*(1-train['error']):.2f}% | {100*(1-val['error']):.2f}% | {val['nll']:.5f} | {val['brier_two_class']:.5f} |")
    gains=[100*(r['summary'][w+'_conclusion']['train']['error']-r['summary'][w+'_arguments']['train']['error']) for w in cfg['models']]
    lines+=['','Neural values average the three individual model scores, not ensemble predictions. NLL and Brier are lower when better. Brier here sums over both binary classes and is twice the single-probability binary Brier.','',
            f"Arguments improve training accuracy by {gains[0]:.2f} percentage points in the compact model and {gains[1]:.2f} points in the larger model, while both validation accuracy and probability loss worsen on average. This pattern is consistent with overfitting under the tested representation and training procedure. It does not establish that argument information is inherently unhelpful or that another representation or more training questions cannot help.",'',
            '![Training and validation accuracy and validation NLL](breadth_arguments_320.png)','',
            '## Selected settings and variation','',
            '| Width / condition | Candidate | Fixed refit updates | Validation seed accuracies |',
            '| --- | --- | ---: | --- |']
    for choice in selection['choices']:
        key=choice['width']+'_'+choice['condition']
        values=sorted((x for x in r['by_seed'] if x['model']==key and x['split']=='validation'),key=lambda x:x['seed'])
        lines.append(f"| {choice['width']} / {choice['condition']} | {choice['candidate']} | {choice['refit_steps']:,} | "+', '.join(f"{x['seed']}: {100*(1-x['error']):.2f}%" for x in values)+' |')
    cap_selected=[x['width']+'/'+x['condition'] for x in selection['choices'] if x['refit_steps']==cfg['training']['max_steps']]
    lines+=['','Each width/condition receives nine choices from learning rates 0.0001/0.0003/0.001 and output-logit penalties 0/0.001/0.01, at tuning seed42. AdamW uses weight decay 0.001, dropout 0.1, batch32 snapshots, gradient clipping1, a 32,768-update cap, validation every512 updates, and ten-check patience. Validation NLL fixes settings and duration before fresh seeds142/143/144 train. Checkpoints preserve optimizer and RNG state for exact same-backend recovery.',
            'Selected checkpoints at the update cap: '+(', '.join(cap_selected) if cap_selected else 'none')+'. This is a finite search and does not establish an architecture or optimization optimum.','',
            'Paired intervals resample 80 whole questions with 5,000 bootstrap draws after seed averaging. They are pointwise, conditional on the fitted models, and do not include training-data resampling or adjustment for repeated validation/model selection. A small interval containing zero does not establish equivalence.','',
            '## Opinion switches and predictor sensitivity','',
            '| Predictor | Error on opinion stays | Error on opinion switches |','| --- | ---: | ---: |']
    for key in ORDER:
        v=r['summary'][key]['validation'];lines.append(f"| {LABELS[key]} | {100*v['stay_error']:.2f}% | {100*v['switch_error']:.2f}% |")
    lines+=['','These subgroups use observed future changes for retrospective evaluation, never as predictor inputs.','',
            '| Argument model | Mean probability change when replacing arguments by conclusions | Predicted-class changes | Accuracy with replaced text |',
            '| --- | ---: | ---: | ---: |']
    for width in cfg['models']:
        rows=[x for x in r['message_replacement_diagnostic'] if x['model']==width+'_arguments']
        lines.append(f"| {width} | {100*np.mean([x['original_vs_conclusion_probability_mae'] for x in rows]):.2f} pp | {100*np.mean([x['class_disagreement'] for x in rows]):.2f}% | {100*(1-np.mean([x['replacement_validation']['error'] for x in rows])):.2f}% |")
    lines+=['','Replacement is an input sensitivity check on a fixed fitted predictor, scored against the original observed outcomes. It is not a new source-model intervention, nor an estimate of what agents would do under the replaced text.','',
            '## Encoding, source audit and limitations','',
            f"The 100,000 sender messages correspond to 358,400 delivered edges. Broadcasting, edge identities and alignment to the current opinion were verified. The source-message audit finds 96,983 standard explicit stance claims matching the prescribed opinion, 3,015 without that standard phrase, and two conflicting claims. Benefits from text may include stance expression, wording, topic and earlier conversational information; they do not isolate valid reasoning. No messages were excluded on the basis of prediction outcomes.", '',
            f"The encoding cache has {r['encoding']['unique_texts']:,} unique texts including two controls, {r['encoding']['tokens']:,} retained tokens, maximum length {r['encoding']['maximum_tokens']} tokens, and zero dropped tokens. Frozen encoding took {r['encoding']['seconds']:.1f} seconds; new predictor fitting took {r['fit_seconds']/60:.1f} minutes. These are measured stage costs, not an end-to-end source-model speedup. No source LLM calls were made.",'',
            f"There are {geometry['unique_train_argument_texts']:,} unique training argument strings and {geometry['unique_validation_argument_texts']:,} unique validation strings, with {geometry['identical_texts_across_question_splits']} exact string overlaps. On training broadcasts, question centroids explain {100*geometry['question_centroid_variance_fraction']:.2f}% of embedding variance; question/current-stance centroids explain {100*geometry['question_and_current_stance_centroid_variance_fraction']:.2f}%. This diagnostic weights each connected sender broadcast once. These variance fractions are not measures of mathematical information or proof that residual features are useless.",'',
            'The question/persona representations still use three PCA components each. A lack of argument gain could reflect question context, the frozen embedding, fusion/optimization, limited generalization, stochastic targets or the task itself; this comparison does not distinguish all of them. A gain would establish useful extra information for this conditional response task, not a universal update law, causal credibility weights or logical-quality recognition. The same 32 personas, eight familiar graphs, entropy-selected objective MATH questions and one conversation per question limit generalization. No new graph-family, persona or untouched-assessment evaluation was added.','',
            '## Verification and artifacts','',
            f"Eight focused checks cover temporal isolation, source alignment, parameter counts, exact recovery of the old graph model with zero text, one-hop masking, isolated nodes, permutation equivariance, text gradients, synthetic tuning/refitting, checkpoint recovery, paired statistics and CPU/GPU agreement. The independent completion audit recomputed {audit['prediction_arrays_recomputed']} saved prediction arrays, {audit['replacement_diagnostics_recomputed']} replacement diagnostics and {audit['paired_contrasts_recomputed']} paired contrasts, verified {audit['source_episodes_verified']} source episodes, {audit['prior_artifacts_verified']} previous artifacts and {audit['frozen_artifacts_verified']} new frozen artifacts. All twelve fresh refits have nonzero text-projection gradients and changed text weights. CPU replay and backend discrepancies are recorded explicitly in the audit.",'',
            'These are computational implementation checks, not independent external replication. Existing scientific code, earlier model/results artifacts, assessment protocols and the separate HPC3 collection remain unchanged.','',
            f'- Protocol, preprocessing, encoding, checkpoints, predictions and by-question scores: `{root}/`',
            f'- Full numerical results and paired intervals: `{root}/results/summary.json`',
            f'- Completion audit: `{root}/completion_audit.json`',
            '- Design and commands: `experiments/breadth_arguments_320_v1/README.md`',
            '- Recompute the audit: `.venv/bin/python scripts/audit_breadth_arguments.py`','']
    Path('reports/breadth_arguments_320.md').write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.9))
    short=['Ising 19','Opinions 818','Conclusions 3,890','Arguments 3,890','Opinions 135,458','Conclusions 184,610','Arguments 184,610']
    y=np.arange(len(ORDER));colors=['#999999','#5b78a0','#97b5c7','#24748e','#9a80af','#c4b0d4','#81579a']
    for i,key in enumerate(ORDER):
        v=r['summary'][key]
        axes[0].plot(100*(1-v['train']['error']),i,'o',mfc='white',mec=colors[i],ms=6)
        axes[0].plot(100*(1-v['validation']['error']),i,'o',color=colors[i],ms=6)
        axes[1].plot(v['validation']['nll'],i,'o',color=colors[i],ms=6)
    for ax in axes:
        ax.set_yticks(y,short);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlabel('Accuracy (%)');axes[1].set_xlabel('Validation NLL (lower is better)')
    axes[0].set_title('Open: training · Filled: validation');axes[1].set_title('Neural models: mean of three refits')
    fig.suptitle('Delivered arguments on 320 training / 80 validation questions',fontsize=13)
    fig.tight_layout();fig.savefig('reports/breadth_arguments_320.png',dpi=180);plt.close(fig)
    json_write('reports/breadth_arguments_320_provenance.json',{'summary_sha256':digest(root/'results/summary.json'),
        'audit_sha256':digest(root/'completion_audit.json'),'report_sha256':digest('reports/breadth_arguments_320.md'),
        'figure_sha256':digest('reports/breadth_arguments_320.png'),'report_code_sha256':digest(__file__)})
    print('\n\n'.join(findings))

if __name__=='__main__':main()
