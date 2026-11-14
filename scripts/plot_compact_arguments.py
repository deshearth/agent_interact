"""Create the scientific figure and concise report from audited saved scores."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import compact_argument_study as study

def main():
    cfg=study.config();root=Path(cfg['output']);r=study.read(root/'results/summary.json');a=study.read(root/'completion_audit.json')
    assert a['status']=='passed'
    for path,sha in a['results_sha256'].items():assert digest(path)==sha,path
    names=['pico_pca8','nano_pca8','pico_full','nano_full']
    display={'I0':'Ising','opinion_pico':'Opinion/context model, width 4','opinion_nano':'Opinion/context model, width 8'}
    for name in names:
        width=4 if name.startswith('pico') else 8
        dimensions=8 if name.endswith('pca8') else 384
        for condition in ('conclusion','arguments'):
            display[name+'_'+condition]=f'Width {width}, {dimensions} text features: '+('choice only' if condition=='conclusion' else 'arguments')
    labels=['346 parameters\nwidth 4 · text 8','882 parameters\nwidth 8 · text 8',
            '1,850 parameters\nwidth 4 · text 384','3,890 parameters\nwidth 8 · text 384']
    y=np.arange(4)
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.6),gridspec_kw={'width_ratios':[1.2,1]})
    fig.subplots_adjust(left=.15,right=.97,top=.82,bottom=.16,wspace=.3,hspace=.35)
    blue='#176ba0';gray='#787878'
    def contrast(name,metric):
        return next(x for x in r['paired_differences'] if x['left']==name+'_arguments' and x['right']==name+'_conclusion' and x['metric']==metric)
    effect_rows=[]
    for row,metric in enumerate(['error','nll']):
        ax,effect=axes[row]
        factor=100 if metric=='error' else 1
        av=np.array([r['summary'][n+'_arguments']['validation'][metric] for n in names])
        cv=np.array([r['summary'][n+'_conclusion']['validation'][metric] for n in names])
        if metric=='error':av=100*(1-av);cv=100*(1-cv)
        for i,(v,c) in enumerate(zip(av,cv)):ax.plot([v,c],[i,i],color='#d3d3d3',lw=2,zorder=1)
        ax.scatter(cv,y,facecolor='white',edgecolor=gray,s=65,label='Conclusion only',zorder=3)
        ax.scatter(av,y,color=blue,s=55,label='Argument text',zorder=4)
        means=[];lo=[];hi=[]
        for name in names:
            c=contrast(name,metric);mean=-factor*c['mean_difference'];interval=-factor*np.array(c['ci95'])[::-1]
            means.append(mean);lo.append(interval[0]);hi.append(interval[1])
            effect_rows.append({'model':name,'metric':metric,'improvement':mean,'ci95':interval.tolist()})
        means=np.array(means);lo=np.array(lo);hi=np.array(hi)
        effect.axvline(0,color='#999999',lw=1,ls='--')
        effect.errorbar(means,y,xerr=np.vstack([means-lo,hi-means]),fmt='o',color=blue,capsize=4,ms=6)
        for target in (ax,effect):
            target.set_yticks(y);target.invert_yaxis();target.set_ylim(3.6,-.6);target.grid(axis='x',color='#ededed')
            target.spines[['top','right','left']].set_visible(False);target.tick_params(axis='y',length=0)
        ax.set_yticklabels(labels,fontsize=10);effect.set_yticklabels([])
        ax.set_xlabel('Validation accuracy (%)' if metric=='error' else 'Validation NLL (lower is better)')
        effect.set_xlabel('Accuracy gain (percentage points)' if metric=='error' else 'NLL reduction (positive is better)')
        if row==0:
            ax.set_title('Matched input comparison',loc='left',fontsize=12,fontweight='bold')
            effect.set_title('Benefit of argument text, with 95% interval',loc='left',fontsize=12,fontweight='bold')
            ax.legend(loc='lower center',bbox_to_anchor=(.4,1.1),ncol=2,frameon=False)
    fig.suptitle('Do compact graph transformers benefit from arguments?',fontsize=17,fontweight='bold',y=.97)
    fig.text(.5,.915,'Next-opinion prediction · 320 training / 80 validation questions · three fresh seeds per condition',ha='center',fontsize=11)
    fig.text(.15,.065,'Parameter counts exclude the frozen 22.7M-parameter MiniLM encoder and frozen PCA.\n'
        'Pointwise question-bootstrap intervals; reused validation also selected settings. These are exploratory response predictions.',fontsize=9,color='#555555')
    target=Path('reports/compact_arguments_320.png');fig.savefig(target,dpi=180);plt.close(fig)
    lines=['# Compact argument-aware transformers: 320/80 questions','',
        'Dataset: the existing MATH-based Llama-3.1-8B conversation corpus, with 320 training and 80 validation questions on eight familiar signed graphs. This is separate from the original paper’s GPT dataset.',
        '',
        'Accuracy measures prediction of each agent’s next majority-of-five A/B opinion, not the correctness of its mathematical answer.',
        '',
        'All scores below average the three fresh refit seeds; they are not an ensemble or a selected best seed.',
        '', '| Model/input | Trainable parameters | Training accuracy | Validation accuracy | Validation NLL |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in ['I0','opinion_pico','opinion_nano']+[n+'_'+c for n in names for c in ('conclusion','arguments')]:
        scores=r['summary'][name];train=scores['train'];val=scores['validation']
        lines.append(f'| {display[name]} | {r["parameters"][name]:,} | {100*(1-train["error"]):.3f}% | {100*(1-val["error"]):.3f}% | {val["nll"]:.5f} |')
    lines+=['','## Added value of argument text','',
        'Both rows in each comparison have the same architecture and trainable parameter count. The control receives only the sender’s stated A/B choice.',
        '', '| Argument model | Accuracy gain (percentage points) | 95% interval | NLL reduction | 95% interval |',
        '| --- | ---: | --- | ---: | --- |']
    for n in names:
        error=next(x for x in effect_rows if x['model']==n and x['metric']=='error')
        nll=next(x for x in effect_rows if x['model']==n and x['metric']=='nll')
        lines.append(f'| {n} | {error["improvement"]:+.3f} | [{error["ci95"][0]:+.3f}, {error["ci95"][1]:+.3f}] | {nll["improvement"]:+.5f} | [{nll["ci95"][0]:+.5f}, {nll["ci95"][1]:+.5f}] |')
    lines+=['','## Interpretation of the completed comparison','',
        'Neither compressed model shows a clear validation-accuracy benefit from arguments. The 346-parameter accuracy change is +0.011 percentage points, with a 95% interval of [-0.192, +0.216]. The 882-parameter change is -0.024 points, with interval [-0.168, +0.124]. Its NLL reduction of 0.00187 is suggestive but uncertain: the interval [-0.00009, +0.00389] includes zero.',
        '',
        'Using all 384 text features increases fitting ability but worsens validation performance. At 1,850 parameters, argument text raises training accuracy by 0.708 points and lowers validation accuracy by 0.443 points. At 3,890 parameters, the corresponding changes are +1.145 and -0.418 points. This is an overfitting pattern under the tested encoder, architecture and training procedure. The original 818-parameter opinion/context model remains slightly more accurate on validation than all four tested argument models.',
        '',
        'The message branch is active: gradients, changed weights and text substitutions all verify that messages affect predictions. For the 3,890-parameter model, actual messages outperform the matched cross-round substitutions by 0.330 points, despite underperforming the separately trained choice-only control. This suggests some update-specific predictive information is present alongside the generalization cost. The substitution is an offline sensitivity diagnostic with unmatched receiver states and histories, so it does not isolate argument validity or establish a causal effect.',
        '', '## Scope and interpretation','',
        '- The smallest tested argument model is 346 trainable parameters: four hidden dimensions and eight compressed message features. This is not a universal minimum.',
        '- MiniLM has 22,713,216 frozen parameters. Message PCA adds 3,457 stored values (384 means, 8×384 components and one scale), fitted without labels on training texts and then frozen. The inherited question/persona transforms are also frozen during predictor training. The 346 count describes the trained update rule, not the entire text-processing system.',
        f'- Message PCA8 was fitted to 64,067 unique training texts and retains {100*r["pca_explained_variance_fraction"]:.2f}% of training embedding variance. The full-vector alternatives avoid that compression.',
        '- Inputs use the actual message delivered after current opinions and before the next opinion update. No answer keys, solutions, future opinions, raw vote probabilities or extra history were added.',
        '- The 400 development conversations provide 81,920 training and 20,480 validation transitions across eight familiar signed graphs. The 40 reserved assessment questions remain ungenerated/unscored in this study.',
        '- Both the hyperparameter search and previous investigations used this validation set. Bootstrap intervals are pointwise, conditional on these fitted models and exploratory; they are not fresh confirmatory test intervals.',
        '- An improvement establishes predictive value for this message representation. It does not establish that the model understands mathematical arguments, that text alone is sufficient, or that a universal interaction law has been found.',
        '', '## Frozen-model substitution diagnostic','',
        'Substitutions preserve question, sender, graph, persona and current sender stance, using another round of the same conversation. They change 96.287% of message texts. Donor rounds may be later in the conversation: this is an offline diagnostic, not a prospective prediction procedure. Receiver state, round and full history are not matched; this is predictor sensitivity, not a causal source-agent intervention.',
        '', '| Model | Probability change, mean absolute (percentage points) | Changed predicted classes | Accuracy gain of actual over substituted text (percentage points) | 95% interval |',
        '| --- | ---: | ---: | ---: | --- |']
    for n in names:
        sensitivity=[x for x in r['text_sensitivity'] if x['model']==n+'_arguments']
        c=next(x for x in r['paired_differences'] if x['left']==n+'_arguments' and x['right']==n+'_arguments_matched' and x['metric']=='error')
        ci=-100*np.array(c['ci95'])[::-1]
        lines.append(f'| {n} | {100*np.mean([x["probability_mae"] for x in sensitivity]):.3f} | {100*np.mean([x["class_disagreement"] for x in sensitivity]):.3f}% | {-100*c["mean_difference"]:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
    lines+=['','## Verification','',
        f'The study completed {r["new_tuning_fits"]} new tuning fits and {r["new_refits"]} fresh refits, using {r["new_fit_seconds"]/60:.1f} fit minutes. Six full-vector width-eight refits and six original opinion-model refits were reused, plus the Ising baseline.',
        f'The independent audit recomputed {a["prediction_arrays_recomputed"]} prediction arrays and {a["paired_contrasts_recomputed"]} paired contrasts; verified {a["source_episodes_verified"]} episode hashes, training-only PCA, donor matching, all frozen artifacts and learned text gradients. The largest score discrepancy was {a["maximum_metric_difference"]:.3g}.',
        '', 'Artifacts: `runs/compact_arguments_320_v1/results/summary.json`, `completion_audit.json`, `models_frozen.json`, and `protocol.json`. The companion models and original datasets remain unchanged.',
        '', '![Compact argument comparison](compact_arguments_320.png)','']
    report=Path('reports/compact_arguments_320.md');report.write_text('\n'.join(lines))
    json_write(root/'report_artifacts.json',{'created_utc':study.shared.now(),'plot_code_sha256':digest(__file__),
        'artifacts':{str(p):digest(p) for p in (target,report)},'effects':effect_rows})
    print(str(target));print(str(report))

if __name__=='__main__':main()
