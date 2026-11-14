"""Render the completed exploratory rule-distillation and pooling diagnostics."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config,digest,json_write
from scripts.distill_transformer_rule import CONFIG,REPORT,LABELS,validate


def main():
    cfg=config(CONFIG); root=Path(cfg['output']); protocol=validate(cfg)
    completed=json.loads((root/'evaluation_complete.json').read_text())
    floors_complete=json.loads((root/'pooled_state_limits/complete.json').read_text())
    for p,sha in {**completed['artifacts'],**floors_complete['artifacts']}.items(): assert digest(p)==sha,p
    assert completed['fits_complete_sha256']==digest(root/'fits_complete.json')
    tests=ET.parse('reports/transformer_rule_distillation_checks.xml').getroot()
    suites=[tests] if tests.tag=='testsuite' else list(tests.iter('testsuite'))
    test_count=sum(int(s.attrib.get('tests',0)) for s in suites)
    assert test_count==10 and all(int(s.attrib.get('errors',0))==int(s.attrib.get('failures',0))==0 for s in suites)
    performance=pd.read_csv(root/'evaluation/performance_summary.csv').set_index(['split','model'])
    fidelity=pd.read_csv(root/'evaluation/fidelity_summary.csv').set_index(['split','model'])
    contrasts=pd.read_csv(root/'evaluation/paired_differences.csv')
    floors=pd.read_csv(root/'pooled_state_limits/summary.csv').set_index('split')
    fits=[json.loads(p.read_text()) for p in (root/'fits').glob('*/*.json')]
    assert len(fits)==48 and all(r['optimizer_success'] for r in fits)
    labels=['Ising + own\n3 + 19 retained','Pooled\n21','Contextual\n39','Quadratic\n42']
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),layout='constrained')
    x=np.arange(4); width=.35
    train=[100*fidelity.loc[('train',f),'probability_mae'] for f in cfg['families']]
    outer=[100*fidelity.loc[('outer',f),'probability_mae'] for f in cfg['families']]
    axes[0].bar(x-width/2,train,width,label='Fitting questions',color='#a6bed2')
    axes[0].bar(x+width/2,outer,width,label='Old held-out questions',color='#245f89')
    axes[0].axhline(100*floors.loc['outer','minimum_probability_mae'],color='#b45a30',linestyle=':',label='Empirical pooling floor')
    axes[0].set_xticks(x,labels,fontsize=9)
    axes[0].set_ylabel('Mean probability difference from transformer (pp)')
    axes[0].set_title('Simple formulas leave a fitting gap')
    axes[0].legend(fontsize=8,loc='lower left')
    rows=contrasts[(contrasts.reference=='teacher')&(contrasts.metric=='error')].set_index('model').loc[cfg['families']]
    point=100*rows.difference.to_numpy(); lo=100*rows.lower.to_numpy(); hi=100*rows.upper.to_numpy()
    axes[1].errorbar(point,x,xerr=np.vstack([point-lo,hi-point]),fmt='o',capsize=4,color='#245f89')
    axes[1].axvline(0,color='#666666',linewidth=1)
    axes[1].set_yticks(x,['Ising + own','Pooled','Contextual','Quadratic'])
    axes[1].set_ylim(3.65,-.35); axes[1].set_xlabel('Formula error − transformer error (pp)')
    axes[1].set_title('Older held-out questions: accuracy gap')
    axes[1].text(.02,.02,'95% descriptive paired-question intervals\nPreviously studied data; not fresh confirmation',transform=axes[1].transAxes,fontsize=8,va='bottom')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Can a small explicit rule reproduce the 818-parameter transformer?',fontsize=13)
    image=Path('reports/transformer_rule_distillation.png'); fig.savefig(image,dpi=180)
    fig.savefig(image.with_suffix('.pdf')); plt.close(fig)
    example_rows=json.loads((root/'pooled_state_limits/natural_state_examples.json').read_text())
    example=max(example_rows,key=lambda r:r['cases'][1]['teacher_probability_A']-r['cases'][0]['teacher_probability_A'])
    a,b=example['cases']; f=a['common_pooled_features']
    lines=['# Can explicit formulas reproduce the 818-parameter transformer?','',
        'The four tested formulas do not reproduce the frozen transformer closely. They miss its probabilities by about five percentage points on both fitting and older held-out questions, and their held-out errors remain near the strengthened Ising baseline. More context-dependent coefficients and quadratic terms give only a small fidelity improvement. This is a completed CPU-only diagnostic on the old twenty-question GPT development pool; it does not use the new Llama cohort.','',
        '![Formula fidelity and paired error gaps](transformer_rule_distillation.png)','',
        '## What was fitted','',
        'Each of twelve frozen transformers (four outer folds, three seeds) supplies soft probability targets on its fifteen fitting questions. Each surrogate minimizes equal-question binary cross-entropy to those probabilities, with a fixed 0.0001 ridge penalty on standardized non-intercept coefficients. Standardization uses fitting inputs only. Observed opinion labels never enter the surrogate optimizer. There is no formula or penalty selection from held-out results. All 48 fits were frozen before evaluating their corresponding five older held-out questions.','',
        'Let `s` be the receiver\'s current ±1 opinion, `F` the sum of current opinions over friendly neighbors, `U` the raw opinion sum over unfriendly neighbors, and `d+`, `d−` their counts. The coefficient of `U` carries its sign. Let `φ` be the shared 16-dimensional context: a constant, three persona components, three question components and nine products.','',
        '| Formula | Fitted coefficients | Expression |',
        '| --- | ---: | --- |',
        '| Ising + own | 3, retaining 19 Ising coefficients | `b + a·Ising_score + c·s` |',
        '| Pooled linear | 21 | `w·φ + ρs + αF + βU + η+d+ + η−d−` |',
        '| Context-dependent | 39 | The pooled rule, with `ρ`, `α`, `β` each affine in the six persona/question PCA components |',
        '| Context-dependent + quadratic | 42 | The context-dependent rule plus `γ1 F² + γ2 FU + γ3 U²` |','',
        'Each expression is a logit; applying the logistic function yields P(next majority = A). The 21/39/42-coefficient rules are standalone predictors over shared context. The first rule also requires its retained Ising coefficients. Pretrained encoder and PCA quantities are outside all these counts. PCA coordinates remain opaque text features, so even a successful surrogate would not yet provide a fully semantic explanation.','',
        '## Agreement with the transformer','',
        'These numbers measure imitation of the predictor, independently of whether its next-opinion prediction is correct. Means average seed scores within question and then weight questions equally.','',
        '| Formula | Fitting probability MAE | Old held-out probability MAE | Old held-out label agreement | Old held-out KL |',
        '| --- | ---: | ---: | ---: | ---: |']
    for family in cfg['families']:
        tr=fidelity.loc[('train',family)]; out=fidelity.loc[('outer',family)]
        lines.append(f'| {LABELS[family]} | {100*tr.probability_mae:.2f} pp | {100*out.probability_mae:.2f} pp | {100*(1-out.teacher_label_disagreement):.2f}% | {out.teacher_kl:.5f} |')
    lines+=['','The similar fitting and held-out probability gaps show that these particular formulas already fail to reproduce the teacher on the states used to fit them. This is not simply an additional generalization gap of the surrogates. The fixed formula families, regularization and supplied aggregate features all constrain these fits; this experiment does not establish a lower bound for every interpretable expression.','',
        '## Prediction of actual next opinions','',
        '| Model | Fitting error | Old held-out error | Old held-out NLL |',
        '| --- | ---: | ---: | ---: |']
    for family in ['I0','I0_current','teacher']+cfg['families']:
        out=performance.loc[('outer',family)]
        tr=f'{100*performance.loc[("train",family),"error"]:.2f}%' if ('train',family) in performance.index else 'Previously reported'
        lines.append(f'| {LABELS[family]} | {tr} | {100*out.error:.2f}% | {out.nll:.5f} |')
    lines+=['','The observed-label Ising correction is the previously fitted baseline. The new three-coefficient Ising surrogate learns from transformer probabilities instead, so their objectives differ. No new Ising or transformer weights were trained.','',
        '| Formula − transformer | Error difference (pp) | Descriptive 95% interval | NLL difference | Descriptive 95% interval |',
        '| --- | ---: | --- | ---: | --- |']
    for family in cfg['families']:
        er=contrasts[(contrasts.model==family)&(contrasts.reference=='teacher')&(contrasts.metric=='error')].iloc[0]
        nl=contrasts[(contrasts.model==family)&(contrasts.reference=='teacher')&(contrasts.metric=='nll')].iloc[0]
        lines.append(f'| {LABELS[family]} | {100*er.difference:+.2f} | [{100*er.lower:+.2f}, {100*er.upper:+.2f}] | {nl.difference:+.5f} | [{nl.lower:+.5f}, {nl.upper:+.5f}] |')
    lines+=['','Intervals use 3,000 paired question resamples, with neural seed scores averaged within question. These are pointwise exploratory comparisons on a repeatedly studied twenty-question pool with overlapping training folds, not independent confirmation or equivalence tests.','',
        '## Does pooling discard essential information?','',
        'After seeing the fitting gap, a separate recorded follow-up grouped states having exactly the same 21 pooled features. The key excludes arbitrary question/agent/graph IDs and all outcomes. The receiver context is included, so different teacher outputs within a group cannot be explained by differing receiver context, own opinion, friendly/unfriendly sums or counts. They can reflect differences in individual neighbor context/opinions/degrees or other graph information accessible to the transformer.','',
        '| Finite-sample constraint | Fitting states | Older held-out states |',
        '| --- | ---: | ---: |',
        f'| Minimum probability MAE from identical pooled inputs | {100*floors.loc["train","minimum_probability_mae"]:.3f} pp | {100*floors.loc["outer","minimum_probability_mae"]:.3f} pp |',
        f'| Minimum label disagreement from identical pooled inputs | {100*floors.loc["train","minimum_teacher_label_disagreement"]:.3f}% | {100*floors.loc["outer","minimum_teacher_label_disagreement"]:.3f}% |',
        f'| States in groups spanning more than 1 pp in teacher probability | {100*floors.loc["train","state_fraction_in_groups_with_range_over_1pp"]:.2f}% | {100*floors.loc["outer","state_fraction_in_groups_with_range_over_1pp"]:.2f}% |','',
        'These are empirical oracle floors obtained using teacher predictions within each group: the group median minimizes absolute probability error, the group mean minimizes squared error and teacher KL, and the group majority minimizes label disagreement. They need not share one optimizer. They are not implementable unseen-question predictors, estimates of source-LLM sampling noise, or upper bounds on the cost of aggregation. Small observed floors do not prove that pooling is sufficient for transfer.','',
        f'The observed held-out MAE floor is {100*floors.loc["outer","minimum_probability_mae"]:.3f} pp, far below the tested quadratic rule\'s {100*fidelity.loc[("outer","quadratic"),"probability_mae"]:.2f} pp. The exact collisions therefore demonstrate some lost information but do not establish pooling as the main explanation for the remaining gap. Nonlinear functional restrictions, regularization, omitted neighborhood detail and transfer across question contexts remain distinguishable possibilities.','',
        'One illustrative extreme pair from the older held-out states is independently reproduced on CPU:','',
        f'- Question `{a["qid"]}`, receiver {a["receiver"]}; own opinion {f["own_opinion"]:+.0f}, friendly sum {f["friendly_sum"]:+.0f}, unfriendly sum {f["unfriendly_sum"]:+.0f}, friendly/unfriendly counts {f["friendly_degree"]:.0f}/{f["unfriendly_degree"]:.0f}. All 21 pooled features match exactly.',
        f'- Transformer P(A): **{100*a["teacher_probability_A"]:.2f}% versus {100*b["teacher_probability_A"]:.2f}%**, on graphs {a["graph_id"]} and {b["graph_id"]}. The full recorded neighborhoods differ.',
        '- This pair was selected to illustrate the largest difference among the saved fold/seed examples; it is not a typical effect or evidence of a causal LLM response.','',
        '## Implication and verification','',
        'The 818-parameter predictor has learned behavior beyond the tested linear/context-affine/quadratic aggregate rules. Its accuracy gain over Ising is modest, yet its function is not closely reproduced by those rules. This does not establish that 818 weights are necessary or that no simpler equation exists. A useful next diagnostic would hold the aggregate input fixed while testing a richer nonlinear predictor, separating limitations of the chosen formulas from limitations of the aggregate state; another would examine individual-neighbor contributions on observed states. Neither should change the ongoing frozen 80/20/40 Llama comparison.','',
        f'All 48 convex fits converged; total optimizer time {sum(r["seconds"] for r in fits):.1f} seconds, maximum final gradient {max(r["gradient_maximum"] for r in fits):.2e}. Ten focused tests passed. All twelve teacher checkpoints reproduce on CPU on both training and older held-out inputs, with maximum probability difference {completed["maximum_probability_reproduction_difference"]:.2e}. Twelve illustrative pooled-input pairs also reproduce on CPU. All {completed["prior_artifacts_unchanged"]:,} tracked earlier artifacts and {completed["active_study_frozen_files_unchanged"]} frozen files of the running Llama study remain unchanged. Execution used one CPU thread, with no GPU fitting, new source calls, new assessment data or paid API use.','',
        'The pooling follow-up was specified after seeing the four surrogate results; its separate protocol records that sequence. This is an exploratory investigation, not a preregistered confirmatory test.','',
        'Artifacts: `runs/transformer_rule_distillation_v1/`, including frozen protocols, 48 coefficient files, raw-unit expressions, saved predictions, per-question fidelity, subgroups and paired differences. Raw PCA coefficients should be interpreted within their fold because PCA axes differ between folds.','',
        'Commands:','',
        '```sh',
        '.venv/bin/python scripts/distill_transformer_rule.py prepare',
        '.venv/bin/python scripts/distill_transformer_rule.py fit',
        '.venv/bin/python scripts/distill_transformer_rule.py evaluate',
        '.venv/bin/python scripts/analyze_pooled_rule_limits.py',
        '.venv/bin/python scripts/report_transformer_rule_distillation.py',
        '```']
    REPORT.write_text('\n'.join(lines)+'\n')
    prior=json.loads((root/'prior_artifacts.json').read_text())
    for p,sha in prior.items(): assert digest(p)==sha,p
    paths=[REPORT,image,image.with_suffix('.pdf'),CONFIG,Path(__file__),Path('scripts/distill_transformer_rule.py'),
           Path('scripts/analyze_pooled_rule_limits.py'),Path('tests/test_transformer_rule_distillation.py'),
           Path('tests/test_pooled_rule_limits.py'),Path('reports/transformer_rule_distillation_checks.xml')]
    paths += [p for p in root.rglob('*') if p.is_file() and p.name!='completion_audit.json']
    json_write(root/'completion_audit.json',{'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'formula_fits':48,'teacher_checkpoints_reproduced':12,'focused_tests':10,
        'prior_artifacts_unchanged':len(prior),'active_study_frozen_files_unchanged':len(protocol['active_study_frozen_files']),
        'new_assessment_data_read':False,'source_calls':0,'artifacts':{str(p):digest(p) for p in paths}})
    print(json.dumps({'report':str(REPORT),'figure':str(image),'complete':True}))


if __name__=='__main__': main()
