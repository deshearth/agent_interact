"""Report the audited data-augmentation contrast without selecting a new winner."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_scaling_study as study

LABELS={'choice978':'Ordered choice · 978','history_choice2306':'Choice + history · 2,306',
    'history_arguments2306':'Arguments + history · 2,306','joint_arguments26210':'Joint question/arguments · 26,210',
    'ising22':'Ising · 22','tree_semantic8':'Argument-feature tree'}


def main():
    root=study.ROOT; out=root/'evaluation'; p=study.protocol(); result=study.read(out/'results.json'); audit=study.read(out/'completion_audit.json')
    assert audit['passed'] and audit['results_sha256']==digest(out/'results.json')
    source=study.read(root/'source_audit.json'); encoding=study.read(root/'encoding.json'); composition=study.read(out/'composition.json')
    summary=result['summary']; names=['ising22','choice978','history_choice2306','history_arguments2306','joint_arguments26210','tree_semantic8']
    argument_comparison=study.read(out/'argument_comparison_descriptive.json')
    def metric(name,cohort,budget,split='validation'): return 100*summary[f'{name}_{cohort}_{budget}'][split]['accuracy']
    def change(name,budget): return next(c for c in result['contrasts'] if (c['model'],c['budget'],c['metric'])==(name,budget,'error'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13.6,7.5),gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.25,right=.98,bottom=.20,top=.82,wspace=.25)
    for y,name in enumerate(names[::-1]):
        baseline=metric(name,320,'original'); axes[0].scatter(baseline,y,color='#7a8793',marker='o',label='320 original' if y==0 else None)
        budgets=['same_updates','same_passes'] if name in study.MODEL_SOURCES else ['fixed']
        for budget in budgets:
            offset,color,marker=((-0.12,'#267aa5','s') if budget=='same_updates' else (.12,'#c37b35','^') if budget=='same_passes' else (0,'#35694d','D'))
            value=metric(name,512,budget); axes[0].scatter(value,y+offset,color=color,marker=marker,
                label={'same_updates':'512 · same updates','same_passes':'512 · same data passes','fixed':'512 · fixed baseline recipe'}[budget] if (y==0 or (name=='joint_arguments26210')) else None)
            axes[0].plot([baseline,value],[y,y+offset],color=color,linewidth=.8,alpha=.5)
            c=change(name,budget); center=100*c['reduction']; lo,hi=100*np.asarray(c['ci95'])
            axes[1].errorbar(center,y+offset,xerr=[[center-lo],[hi-center]],fmt=marker,color=color,capsize=3,markersize=5)
    axes[0].set(yticks=range(len(names)),yticklabels=[LABELS[n] for n in names[::-1]],xlabel='All-case mode accuracy (%)',xlim=(84.5,95.6),ylim=(-.6,len(names)-.4))
    axes[0].axvline(95,color='#ae4940',linestyle='--',linewidth=1); axes[0].text(95.08,len(names)-.5,'95% target',rotation=90,va='top',color='#ae4940',fontsize=9)
    axes[1].axvline(0,color='#88939a',linewidth=1); axes[1].set(yticks=[],xlabel='Change from 320 questions (percentage points)',ylim=axes[0].get_ylim(),title='Paired 95% question intervals')
    for ax in axes: ax.grid(axis='x',alpha=.15)
    fig.suptitle('Does adding training questions improve prediction?',x=.60,y=.95,fontsize=16)
    fig.text(.60,.90,'320 → 512 training questions · unchanged 80-question validation anchor · all updates retained',ha='center',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels(); unique=dict(zip(labels,handles))
    fig.legend(unique.values(),unique.keys(),loc='center left',bbox_to_anchor=(.25,.10),ncol=2,frameon=False,fontsize=9)
    fig.text(.25,.025,'Neural values average three individual seeds. Frozen encoder/PCA unchanged; added questions follow revised topic sampling.\nIntervals are descriptive on reused validation. Newly collected validation and reserved assessment remain unused.',fontsize=9,color='#4b5964')
    figure=Path('reports/modal_scaling_512.png'); fig.savefig(figure,dpi=170); plt.close(fig)
    best=max((k for k in summary if '_512_' in k),key=lambda k:summary[k]['validation']['accuracy'])
    best_accuracy=100*summary[best]['validation']['accuracy']
    lines=['# Training-question augmentation: 320 versus 512','',f"Completed {result['completed_utc']}.",'',
        f"Best observed 512-question validation accuracy: **{best_accuracy:.3f}%** ({best}). "
        +('The 95% target remains unmet.' if best_accuracy<94.5 else 'This is an exploratory validation result, not an untouched assessment claim.'),'',
        'The new cohort adds the first 192 training units from the larger collection\'s frozen ordering to the old 320. '
        'Every selected unit was complete before export. The old 80 validation questions are the common anchor; new validation conversations and all 40 reserved assessment conversations remain unused. '
        'The outcome is the most probable next source A/B response, using all 256 agent updates per question on existing stochastic histories.','',
        '## Fixed comparison','',
        '| Model | 320 validation | 512, same updates | 512, same data passes |','| --- | ---: | ---: | ---: |']
    for name in names:
        old=metric(name,320,'original')
        if name in study.MODEL_SOURCES: a=f"{metric(name,512,'same_updates'):.3f}%"; b=f"{metric(name,512,'same_passes'):.3f}%"
        else: a=f"{metric(name,512,'fixed'):.3f}% (fixed recipe)"; b='—'
        lines.append(f'| {LABELS[name]} | {old:.3f}% | {a} | {b} |')
    lines += ['', 'Ising and tree are refitted once with their fixed original recipes. Neural results average the same three initialization seeds; no ensembling or best-seed selection. '
        'Same updates holds optimizer steps fixed. Same data passes increases steps by approximately 512/320; both checkpoints were declared before fitting. '
        'All 320 endpoints replay unchanged saved models. Settings were originally selected inside the old training set. There is no new validation-based retuning.','',
        'The argument/history model gains **0.143 pp [0.013, 0.277]** from more questions at equal data passes. '
        'Other accuracy-change intervals include zero. This is a small descriptive improvement, with no adjustment for the many comparisons or earlier validation reuse. '
        'The wider argument model gains 0.239 pp at equal passes, but its interval includes zero and it still trails the compact models.','',
        '| Model / 512 budget | Training accuracy | Validation change | Paired 95% interval |','| --- | ---: | ---: | ---: |']
    for name in names:
        for budget in (['same_updates','same_passes'] if name in study.MODEL_SOURCES else ['fixed']):
            c=change(name,budget)
            lines.append(f"| {LABELS[name]} / {budget.replace('_',' ')} | {metric(name,512,budget,'train'):.3f}% | {100*c['reduction']:+.3f} pp | [{100*c['ci95'][0]:+.3f}, {100*c['ci95'][1]:+.3f}] |")
    lines += ['', 'At 512 questions, the history model still has **no clear argument advantage** over its choice counterpart: '
        'the post-fit difference is +0.011 pp [−0.138, +0.160] at equal updates and +0.093 pp [−0.067, +0.254] at equal data passes. '
        'Their original internally selected training durations differ; these are comparisons of the fitted pipelines.','',
        '## Representation and data controls','',
        'The old question/persona PCA and message PCA remain frozen at their 320-question fits. Old inputs are byte-identical. Added questions and messages use the same frozen MiniLM encoder '
        'and complete token pooling. The full-vector argument model jointly encodes the question, options and message. '
        'All models use current probabilities; none receives next probabilities, answer keys, worked solutions or screening scores as features. '
        'The compact history pair uses the same numerical history and differs in choice-only versus actual argument tokens. Frozen MiniLM adds 22,713,216 pretrained weights to the update-rule counts.','',
        f"Encoding added {encoding['texts']:,} distinct texts and retained all {encoding['tokens']:,} tokens in {encoding['seconds']:.1f} seconds. "
        f"A six-text CPU replay differs by at most {encoding['cpu_max_vector_difference']:.2g} per vector entry. No source-model calls were made for this study.",'',
        '## The added questions have a different topic mix','',
        '| Topic | Original 320 | Added 192 |','| --- | ---: | ---: |']
    for topic in sorted(set(composition['original320']['type'])|set(composition['added192']['type'])):
        lines.append(f"| {topic} | {composition['original320']['type'].get(topic,0)} | {composition['added192']['type'].get(topic,0)} |")
    lines += ['', 'The larger MATH bank follows the user-approved availability-proportional topic quotas and revised selection fractions. '
        'Consequently this experiment measures augmentation from that selected pool, rather than a pure sample-size change under an identical question distribution. '
        'Graph, label and difficulty counts are retained in the numerical composition file.','',
        '## Verification','',
        f"All {source['prompt_reconstructions']:,} new full prompt hashes and {source['raw_votes']:,} votes reproduce. "
        f"The 192 added conversations comprise {source['execution_types'].get('NVIDIA A30',0)} A30 and {source['execution_types'].get('NVIDIA L40S',0)} L40S runs. "
        f"Maximum repeated-prompt normalized probability spread is {source['max_within_prompt_normalized_probability_spread']:.3g}; none changes the preferred answer. "
        'Targets consistently use the first saved response\'s conditional A/B distribution. All cases are retained.','',
        f"Six focused checks pass. The independent audit verifies all {audit['prediction_files']} prediction arrays and {audit['contrasts']} paired contrasts, "
        f"replays all {audit['neural_validation_replays']} neural and {audit['baseline_validation_replays']} baseline validation outputs, "
        f"and confirms {audit['old_prediction_arrays_unchanged']} old prediction arrays are unchanged. "
        f"Maximum independent metric difference is {audit['independent_metric_max_difference']:.2g}. "
        'All final checkpoints froze before validation scoring. A tiny lexical-scaling precision mismatch was fixed during the pre-fit input-equivalence check; the unscored protocol revision is archived.','',
        'The 80 validation questions have been reused extensively. Bootstrap intervals resample whole questions and condition on the fitted models; '
        'they do not account for the earlier search over architectures, repeated validation, or new training-cohort sampling. '
        'This remains familiar-graph, familiar-persona, one-step prediction after message delivery. It does not establish a universal accuracy ceiling or autonomous rollout performance.','',
        'Artifacts: [results](../runs/modal_scaling_512_v1/evaluation/results.json), [audit](../runs/modal_scaling_512_v1/evaluation/completion_audit.json), '
        '[protocol](../runs/modal_scaling_512_v1/protocol.json), [figure](modal_scaling_512.png).','']
    report=Path('reports/modal_scaling_512.md'); report.write_text('\n'.join(lines))
    json_write(out/'report_summary.json',dict(best_model=best,best_accuracy=best_accuracy/100,target_accuracy=.95,
        report_sha256=digest(report),figure_sha256=digest(figure),report_code_sha256=digest(__file__)))
    print(study.read(out/'report_summary.json'),flush=True)


if __name__=='__main__': main()
