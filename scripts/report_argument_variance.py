"""Render the audited accuracy diagnosis in the repository's report format."""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import argument_variance_diagnosis as study

def main():
    cfg=study.config();root=Path(cfg['output']);r=study.read(root/'results/summary.json');audit=study.read(root/'completion_audit.json')
    assert audit['status']=='passed'
    for p,sha in audit['results_sha256'].items():assert digest(p)==sha
    def score(name,split='validation'):
        return next(row for row in r['models'] if row['model']==name and row['split']==split)
    def comparison(a,b):
        return next(row for row in r['comparisons'] if row['left']==a and row['right']==b and row['split']=='validation')
    def improvement(record,metric='error'):
        c=record['paired'][metric];factor=100 if metric=='error' else 1
        return -factor*c['mean_difference'],(-factor*np.asarray(c['ci95'])[::-1]).tolist()
    model_pairs=[('opinion_nano','I0'),('pico_pca8_arguments','pico_pca8_conclusion'),
        ('nano_pca8_arguments','nano_pca8_conclusion'),('nano_full_arguments','nano_full_conclusion')]
    labels=['818 vs Ising','346 arguments vs choice','882 arguments vs choice','3,890 arguments vs choice']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    blue='#246f9e';red='#bd574c';gray='#c3cbd1';green='#31866f'
    y=np.arange(4);a=axes[0,0]
    corrections=[100*comparison(*p)['corrected'] for p in model_pairs]
    introduced=[100*comparison(*p)['introduced'] for p in model_pairs]
    a.barh(y-.16,corrections,height=.29,color=blue,label='Errors corrected')
    a.barh(y+.16,introduced,height=.29,color=red,label='Errors introduced')
    a.set_yticks(y,labels);a.invert_yaxis();a.set_xlabel('Percentage of all validation updates')
    a.set_title('Similar accuracy can hide offsetting changes',loc='left',fontweight='bold',fontsize=11)
    a.legend(frameon=False,fontsize=9)
    a=axes[0,1];names=['full384','ordinary_pca8','within_pca8'];labs=['Full 384','Ordinary PCA8','Within-question PCA8']
    between=np.array([r['representation_geometry']['train'][n]['between_question_fraction'] for n in names])*100
    a.barh(np.arange(3),between,color=gray,label='Between questions')
    a.barh(np.arange(3),100-between,left=between,color=green,label='Within a question')
    a.set_yticks(np.arange(3),labs);a.invert_yaxis();a.set_xlim(0,100);a.set_xlabel('Fraction of training message variation (%)')
    a.set_title('PCA emphasizes differences between questions',loc='left',fontweight='bold',fontsize=11)
    a.legend(frameon=False,fontsize=9,loc='lower center',bbox_to_anchor=(.5,-.3),ncol=2)
    for i,b in enumerate(between):a.text(b/2,i,f'{b:.1f}%',ha='center',va='center',fontsize=10)
    a=axes[1,0];selected=['nano_pca8_conclusion','nano_pca8_arguments','within_pca8_conclusion','within_pca8_arguments','nano_full_arguments']
    labs=['Ordinary PCA8\nchoice only','Ordinary PCA8\narguments','Within PCA8\nchoice only','Within PCA8\narguments','Full 384\narguments']
    x=np.arange(len(selected))
    a.plot(x,[100*(1-score(n,'train')['error']) for n in selected],'o-',color=blue,label='Training')
    a.plot(x,[100*(1-score(n)['error']) for n in selected],'o-',color=red,label='Validation')
    a.set_xticks(x,labs,fontsize=8);a.set_ylabel('Accuracy (%)');a.set_title('Representation change at fixed training settings',loc='left',fontweight='bold',fontsize=11)
    a.legend(frameon=False,fontsize=9)
    a=axes[1,1];anchors=r['anchor']['validation'];ref=100*anchors['sampling_reference_error']
    model_names=['I0','opinion_nano','nano_pca8_arguments','nano_full_arguments']
    expected=np.array([100*anchors['models'][n]['expected_error'] for n in model_names])
    a.barh(np.arange(4),np.repeat(ref,4),color=gray,label='Full-prompt sampling reference')
    a.barh(np.arange(4),expected-ref,left=ref,color=blue,label='Additional expected error')
    a.set_yticks(np.arange(4),['Ising','818 opinions','882 arguments','3,890 arguments']);a.invert_yaxis()
    a.set_xlabel('Expected error on retained 20-question anchor (%)')
    a.set_title('Sampling variation leaves a substantial model gap',loc='left',fontweight='bold',fontsize=11)
    a.legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.32))
    for i,v in enumerate(expected):a.text(v+.15,i,f'{v:.2f}%',va='center',fontsize=9)
    a.set_xlim(0,expected.max()+2)
    for a in axes.flat:
        a.spines[['top','right']].set_visible(False);a.grid(axis='x',color='#eeeeee',zorder=0);a.set_axisbelow(True)
    fig.suptitle('Why are the prediction accuracies close?\n320 training and 80 validation questions',fontsize=16,fontweight='bold')
    figure=Path('reports/argument_variance_320.png');fig.savefig(figure,dpi=180);plt.close(fig)
    lines=['# Why the opinion prediction accuracies remain close','',
        'The models share many errors, and most new argument-driven decisions correct one error while introducing another. Source sampling contributes substantial uncertainty, but a retained probability diagnostic leaves a sizeable gap beyond it. Ordinary PCA emphasizes differences between questions; the controlled follow-up below tests whether that compression choice explains the weak argument benefit.',
        '',
        'The dataset contains 320 training and 80 validation MATH-based Llama-3.1-8B conversations on eight familiar signed graphs. Accuracy predicts each agent’s next majority-of-five A/B opinion, not mathematical correctness. Neural scores average three fitted seeds, not an ensemble. All analyses use the existing development corpus; the 40 assessment questions remain untouched.',
        '', '## Corrections and new errors cancel','',
        '| Comparison | Decisions changed | Errors corrected | Errors introduced | Net accuracy gain |',
        '| --- | ---: | ---: | ---: | ---: |']
    for label,pair in zip(labels,model_pairs):
        c=comparison(*pair)
        lines.append(f'| {label} | {100*c["disagreement"]:.3f}% | {100*c["corrected"]:.3f}% | {100*c["introduced"]:.3f}% | {100*(c["corrected"]-c["introduced"]):+.3f} pp |')
    c=comparison('nano_pca8_arguments','nano_pca8_conclusion')
    lines+=['',f'The 882-parameter argument model agrees with its matched choice-only control on {100*(1-c["disagreement"]):.2f}% of decisions. Of its errors, {100*c["shared_fraction_of_left_errors"]:.2f}% are shared with that control. On decisions where they disagree, the argument model is correct only {100*c["left_accuracy_when_disagree"]:.2f}% of the time. Equal aggregate accuracy therefore reflects both largely shared predictions and cancellation among the changed ones.',
        '', '## Split source votes account for most errors','',
        '| Next-state vote agreement | Share of validation updates | Ising error | 818 opinion error | 882 argument error | 3890 argument error |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for k in ('3','4','5'):
        lines.append(f'| {k}/5 | {100*score("opinion_nano")["vote_agreement"][k]["fraction"]:.2f}% | '+
            ' | '.join(f'{100*score(n)["vote_agreement"][k]["error"]:.2f}%' for n in ['I0','opinion_nano','nano_pca8_arguments','nano_full_arguments'])+' |')
    lines+=['',f'Split votes account for {100*score("opinion_nano")["error_fraction_on_split_votes"]:.2f}% of the 818 model’s errors. Vote agreement uses the realized target votes, so this is a retrospective difficulty breakdown, not a usable predictor feature or an estimated noise ceiling.',
        '', '## The source probabilities leave room for improvement','',
        'Saved source probabilities cover the retained 20 validation questions, rather than all 80. Their complete source prompts and votes match the present data. Normalize the first A/B-token probabilities to p, then compute majority-A probability q = p³(10 − 15p + 6p²). Averaging errors under q reduces dependence on the single observed five-vote label.',
        '', '| Model or reference | Expected error on retained validation questions |',
        '| --- | ---: |',f'| Full-prompt sampling reference | {ref:.3f}% |']
    for n,label in [('I0','Ising'),('opinion_nano','818 opinion model'),('nano_pca8_arguments','882 argument model'),('nano_full_arguments','3890 argument model')]:
        lines.append(f'| {label} | {100*anchors["models"][n]["expected_error"]:.3f}% |')
    lines+=['',f'The 818 model has {100*anchors["models"]["opinion_nano"]["excess_expected_error"]:.2f} percentage points of expected error above the full-prompt sampling reference. Sampling variation explains part of the error but does not support a claim that current accuracy is optimal. This first-token approximation conditions on the full realized inbox and question; it is not a universal ceiling for the compressed predictor inputs. The anchor differs in composition from the full 80-question validation set.',
        '', '## What the eight text features preserve','',
        '| Representation | Between-question variation | Within-question variation |',
        '| --- | ---: | ---: |']
    for n,label in zip(names,['Full 384-dimensional embeddings','Ordinary PCA8','Within-question PCA8']):
        g=r['representation_geometry']['train'][n]
        lines.append(f'| {label} | {100*g["between_question_fraction"]:.2f}% | {100*g["within_question_fraction"]:.2f}% |')
    orig,new=(r['training_unique_text_subspaces'][n] for n in ['ordinary_pca8','within_pca8'])
    lines+=['',
        'These fractions describe numerical embedding variation, not percentages of information about reasoning correctness. The between-question component can include question content and question-associated message patterns. The residual also includes wording, stance, persona and round effects, not just argument quality.',
        '',
        f'The follow-up replaces ordinary PCA with the top eight directions of the within-question covariance of training messages. Under the same unique-text weighting, it retains {100*new["within_variance_retained"]:.2f}% of within-question variation versus {100*orig["within_variance_retained"]:.2f}% previously. Its retained total variance decreases from {100*orig["total_variance_retained"]:.2f}% to {100*new["total_variance_retained"]:.2f}%. Thus the intended redistribution of representation capacity occurred.',
        '', '## Controlled compression experiment','',
        'Both representations use the same 882-parameter transformer, frozen MiniLM, signed graphs, question/persona context, training examples and three initialization seeds. Each condition reuses its original learning rate, penalty and training duration. Six new models were fitted, with no new hyperparameter search. All three seeds began from exactly the corresponding original weights.',
        '',
        'Training-only question means construct the covariance. Every message at prediction time uses a single fixed global training mean and fixed projection; held-out question means, future messages and future labels are not needed. The 882 count excludes the frozen encoder and transforms.',
        '', '| Compression and input | Training accuracy | Validation accuracy | Validation NLL |',
        '| --- | ---: | ---: | ---: |']
    for n,label in [('nano_pca8_conclusion','Ordinary PCA8, choice only'),('nano_pca8_arguments','Ordinary PCA8, arguments'),
        ('within_pca8_conclusion','Within-question PCA8, choice only'),('within_pca8_arguments','Within-question PCA8, arguments')]:
        train,val=score(n,'train'),score(n)
        lines.append(f'| {label} | {100*(1-train["error"]):.3f}% | {100*(1-val["error"]):.3f}% | {val["nll"]:.5f} |')
    lines+=['', '| Contrast | Accuracy gain in percentage points | 95% interval | NLL reduction | 95% interval |',
        '| --- | ---: | --- | ---: | --- |']
    for a,b,label in [('within_pca8_arguments','nano_pca8_arguments','New versus ordinary compression with arguments'),
        ('within_pca8_conclusion','nano_pca8_conclusion','New versus ordinary compression with choice only'),
        ('within_pca8_arguments','within_pca8_conclusion','Arguments versus choice under new compression'),
        ('within_pca8_arguments','opinion_nano','New argument model versus original 818')]:
        c=comparison(a,b);gain,ci=improvement(c);nll,nci=improvement(c,'nll')
        lines.append(f'| {label} | {gain:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {nll:+.5f} | [{nci[0]:+.5f}, {nci[1]:+.5f}] |')
    did=r['diagnostics']['difference_in_differences'];ci=-100*np.asarray(did['error']['ci95'])[::-1]
    lines+=['',f'After subtracting the change in the choice-only control, the change in argument accuracy benefit is {-100*did["error"]["mean_difference"]:+.3f} points, with interval [{ci[0]:+.3f}, {ci[1]:+.3f}]. These are pointwise paired-question bootstrap intervals, conditional on the fitted models and reused validation set.',
        '',
        'This fixed-recipe experiment tests whether one specific unsupervised compression change improves transfer. It does not establish the best achievable performance after retuning or show that the retained directions encode argument validity.',
        '', '## Interpretation','',
        'The shared-error and correction accounting explains mechanically why the reported accuracies remain close. Stronger fitting by the full-vector models, alongside worse validation performance, supports a generalization problem in the tested pipeline. The source-probability diagnostic rules against attributing the entire remaining error to vote sampling.',
        '',
        'MiniLM geometry and the graph predictor may still fail to represent how a particular argument bears on a particular question and receiver. That is a hypothesis, not a demonstrated root cause. More question diversity and representations trained to predict the source response to controlled argument changes remain discriminating directions; another parameter sweep alone would not distinguish them.',
        '', '## Verification','',
        f'All four focused tests passed. Six new refits took {r["fit_seconds"]/60:.2f} fit minutes. The independent audit recomputed {audit["prediction_arrays_recomputed"]} prediction arrays and {audit["paired_contrasts_recomputed"]} paired contrasts, checked the covariance against explicit weighted residuals, verified the global transform and original initializations, and matched all 100 source-anchor episodes. Maximum aggregate metric discrepancy was {audit["maximum_metric_difference"]:.3g}. Frozen earlier artifacts and background source collection remain unchanged.',
        '',
        'Artifacts: `runs/argument_variance_320_v1/protocol.json`, `models_frozen.json`, `results/summary.json` and `completion_audit.json`.',
        '', '![Accuracy diagnosis and representation experiment](argument_variance_320.png)','']
    report=Path('reports/argument_variance_320.md');report.write_text('\n'.join(lines))
    json_write(root/'report_artifacts.json',{'created_utc':study.shared.now(),'report_script_sha256':digest(__file__),
        'artifacts':{str(p):digest(p) for p in [report,figure]}})
    print(str(report));print(str(figure))

if __name__=='__main__':main()
