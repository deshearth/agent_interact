"""Report audited distribution predictions without conflating them with accuracy."""
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import distribution_prediction_study as study

FAMILIES = [('ising', 'Ising', (19, 22)), ('graph', 'Graph transformer', (818, 826)),
            ('ordered_choice', 'Ordered choice control', (970, 978)),
            ('ordered_arguments', 'Ordered arguments', (970, 978))]


def contrast(results, left, right, metric):
    return next(c for c in results['paired_differences'] if (c['left'], c['right'], c['metric']) == (left, right, metric))


def reduction(c, scale=1, decimals=4):
    return f"{-scale*c['difference']:+.{decimals}f} [{-scale*c['ci95'][1]:+.{decimals}f}, {-scale*c['ci95'][0]:+.{decimals}f}]"


def main():
    cfg = study.config(); root = Path(cfg['output']); report_dir = root/'report_v1'; report_dir.mkdir(exist_ok=True)
    result = study.read(root/'results.json'); summary = result['summary']; protocol = study.validate(cfg)
    audit = study.read(root/'completion_audit.json')
    assert audit['passed'] and audit['results_sha256'] == digest(root/'results.json')
    assert audit['audit_code_sha256'] == digest('scripts/audit_distribution_prediction.py')
    reference = study.read(root/'reference_results.json')
    assert audit['reference_results_sha256'] == digest(root/'reference_results.json')
    seed_rows = study.read(root/'seed_by_question.json'); secondary = study.read(root/'majority_secondary.json')
    seed_ranges, secondary_summary = {}, {}
    for name in summary:
        seed_ranges[name] = {}
        seeds = sorted({r['seed'] for r in seed_rows if r['model'] == name})
        for metric in ('kl', 'mae', 'within_05', 'mode_error'):
            values = [float(np.mean([r[metric] for r in seed_rows if r['model'] == name and r['split'] == 'validation' and r['seed'] == seed])) for seed in seeds]
            seed_ranges[name][metric] = {'seeds': seeds, 'values': values, 'min': min(values), 'max': max(values)}
        secondary_summary[name] = {metric: float(np.mean([r[metric] for r in secondary if r['model'] == name and r['split'] == 'validation']))
                                   for metric in ('majority_probability_mae', 'sampled_majority_accuracy')}
    # Post-fit diagnostics only: fixed probability regimes, all validation cases retained.
    bins = [.5, .6, .8, .95, 1.0000001]; regime_rows = []
    for item in result['prediction_files']:
        if item['split'] != 'validation':
            continue
        d = dict(np.load(item['path'], allow_pickle=False)); truth = d['target_probability']; predicted = d['probability']
        confidence = np.maximum(truth, 1-truth)
        for lo, hi in zip(bins, bins[1:]):
            mask = (confidence >= lo)&(confidence < hi)
            regime_rows.append(dict(model=item['model'], seed=item['seed'], lower=lo, upper=min(hi, 1.),
                updates=int(mask.sum()), share=float(mask.mean()), mae=float(np.abs(truth[mask]-predicted[mask]).mean()),
                mode_agreement=float(((predicted[mask] > .5) == (truth[mask] > .5)).mean())))
    json_write(report_dir/'source_confidence_regimes.json', regime_rows)
    tunes = [study.read(p) for p in (root/'tune').glob('*/manifest.json')]
    refits = [study.read(p) for p in (root/'refit').glob('*/manifest.json')]
    start = min(datetime.fromisoformat(r['created_utc']) for r in tunes)
    finish = max(datetime.fromisoformat(r['completed_utc']) for r in refits)
    fit_minutes = (finish-start).total_seconds()/60
    worker_minutes = sum(r['seconds'] for r in tunes+refits)/60
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.subplots_adjust(left=.19, right=.985, bottom=.16, top=.79, wspace=.07)
    colors = ['#3375a8', '#c66a2d']; offsets = [.13, -.13]
    for ax, metric, scale, xlabel in zip(axes, ('kl', 'mae'), (1, 100), ('KL divergence (nats)', 'Mean absolute probability error (percentage points)')):
        maximum = max(summary[family+'_'+state]['validation'][metric]*scale for family, _, _ in FAMILIES for state in ('opinion', 'probability'))
        for y, (family, label, counts) in zip(range(3, -1, -1), FAMILIES):
            values = [summary[family+'_'+state]['validation'][metric]*scale for state in ('opinion', 'probability')]
            ax.plot(values, [y+offsets[0], y+offsets[1]], color='#bac3ca', linewidth=1.4)
            for value, offset, color, marker in zip(values, offsets, colors, ('o', 's')):
                ax.scatter(value, y+offset, color=color, marker=marker, s=50, zorder=3)
                ax.text(value+maximum*.018, y+offset, f'{value:.4f}' if metric == 'kl' else f'{value:.2f}', va='center', fontsize=9, color=color)
        labels = [f'{label}\n({counts[0]} / {counts[1]} learned)' for _, label, counts in FAMILIES]
        ax.set(yticks=range(3, -1, -1), yticklabels=labels if ax is axes[0] else [], ylim=(-.65, 3.65),
               xlim=(0, maximum*1.22), xlabel=xlabel, title='Lower is better' if metric == 'kl' else 'Lower is better')
        ax.grid(axis='x', alpha=.17)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle='None', label=l) for c, m, l in zip(colors, ('o', 's'), ('Current opinions', 'Opinions + current probabilities'))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.57, .91), ncols=2, frameon=False)
    fig.suptitle("Predicting each agent’s probability of choosing A", fontsize=15, y=.985)
    fig.supxlabel('320 training / 80 reused validation questions. Neural scores average three seeds. All models train on probability targets.', fontsize=9, y=.015)
    fig.savefig('reports/distribution_prediction_320.png', dpi=180); plt.close(fig)
    completed = datetime.fromisoformat(audit['completed_utc']).astimezone(ZoneInfo('America/Chicago')).strftime('%B %d, %Y, %I:%M %p Central')
    lines = ['# Predicting opinion distributions instead of sampled opinions', '', f'Completed and audited {completed}.', '',
        'This experiment implements both requested input states on the existing **320 training /80 validation MATH/Llama conversations**, generated with the pinned Llama-3.1-8B MLX 4-bit checkpoint. It compares the original Ising rule, a graph transformer, and matched ordinary-attention models with and without argument content. All eight models predict the source agent’s next single-response probability of choosing A. No new source calls were needed.', '',
        '**Current probabilities help every family; arguments have no clear additional KL benefit in this representation.** The graph transformer improves validation MAE from 13.989 to 13.303 percentage points when current probabilities are added, while the ordered choice-only model improves from 13.530 to 12.772. The corresponding enriched Ising baseline has 15.451 points of error. All four current-probability gains have positive paired KL and MAE reduction intervals. Ordered arguments reach 12.918 points, slightly worse than the matched choice-only model in MAE, with effectively identical KL.', '',
        '![Validation probability prediction comparison](distribution_prediction_320.png)', '',
        '## What is being predicted', '',
        r'\[p_i(t)=\frac{P(\text{first literal token A}\mid\text{actual prompt}_{i,t})}{P(A\mid\text{actual prompt}_{i,t})+P(B\mid\text{actual prompt}_{i,t})},\qquad \boldsymbol p_i(t)=(1-p_i(t),p_i(t)).\]', '',
        'The probabilities were saved at the actual source temperature 0.7. They describe how the source answers a repeated identical prompt, not a direct measure of subjective belief or correctness. They are probabilities of one response, not the empirical count from five responses. The primary target is this normalized first-token distribution; its interpretation as the complete parsed-opinion distribution is approximate. All cases remain included.', '',
        r'\[\hat p_i(t+1)=\sigma\!\left(f_\theta(s_t,J,\phi[,p_t][,\text{delivered arguments and order}])_i\right).\]', '',
        'Here s is the recorded current majority opinion, J is the signed communication graph, and φ is the original frozen question/persona context. In the enriched state, every agent additionally supplies its current probability through m=2p−1. Current probabilities come from round t; only p at t+1 is a training target. These numerical probabilities enrich the predictor’s inputs; the source agents were not newly prompted with probability vectors. This is one-step prediction with observed current state and already-delivered messages. It does not test a free-running rollout using the predictor’s own previous probability estimates.', '',
        'All predictors are newly trained on probability targets. The old majority-of-five distribution is q=M₅(p)=10p³(1−p)²+5p⁴(1−p)+p⁵; it is reported only as a secondary check. The earlier approximately 91.4% expected sampled-majority accuracy does **not** cap probability fidelity. Perfect distribution prediction would have zero KL and zero probability error.', '',
        '## Primary results', '',
        r'\[\mathrm{KL}(p\Vert\hat p)=p\log\frac{p}{\hat p}+(1-p)\log\frac{1-p}{1-\hat p},\qquad \mathrm{MAE}=\mathbb E[|p-\hat p|].\]', '',
        'KL (source || predictor), in nats, is the primary metric. MAE is shown in percentage points: a 12-point error means, for example, predicting 0.62 when the target is 0.50. “Within 5 points” counts updates with |p−p̂|≤0.05; it is not sampled-opinion accuracy. Scores average questions, and then the three individual neural-seed scores, without probability ensembling.', '',
        '| Input state | Model | Learned parameters | Train MAE, pp | Validation MAE, pp | Validation KL | Within 5 points |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for suffix, state_label in [('opinion', 'Opinions'), ('probability', 'Opinions + probabilities')]:
        for family, label, counts in FAMILIES:
            tr, va = summary[family+'_'+suffix]['train'], summary[family+'_'+suffix]['validation']
            lines.append(f"| {state_label} | {label} | {counts[suffix == 'probability']:,} | {100*tr['mae']:.3f} | {100*va['mae']:.3f} | {va['kl']:.5f} | {100*va['within_05']:.2f}% |")
    lines += ['', 'The original opinion graph model has 818 weights. Adding the current probability costs eight weights, giving 826. Ordered choice and argument models have 970/978 weights and share the same actual message order and architecture. Their difference isolates the tested argument representation. Frozen MiniLM has **22,713,216 pretrained parameters**, excluded from these learned-rule counts; PCA8 is fitted only on training messages. Thus the entire text-processing system is larger than the update rule.', '',
        'Two descriptive references were fixed before any outer scoring, separately from model selection. Persistence predicts p(t+1)=p(t); the constant predicts the training-only mean probability '+f"{reference['training_mean']:.5f}"+'. Their validation MAE /KL are **'+f"{100*reference['summary']['persistence']['validation']['mae']:.3f} pp /{reference['summary']['persistence']['validation']['kl']:.5f}"+'** for persistence and **'+f"{100*reference['summary']['training_mean']['validation']['mae']:.3f} pp /{reference['summary']['training_mean']['validation']['kl']:.5f}"+'** for the training mean. These distinguish learning the update from merely carrying current probabilities forward.', '',
        '## Paired comparisons', '',
        'Positive reductions favor the first model. Brackets are pointwise 95% intervals from 5,000 paired whole-question bootstrap samples. These are exploratory intervals conditional on fitted models, not fresh-assessment or multiple-comparison-adjusted claims.', '',
        '| Comparison | KL reduction, nats [95% interval] | MAE reduction, pp [95% interval] |',
        '| --- | ---: | ---: |']
    comparisons = [('graph_opinion', 'ising_opinion', '818 graph vs Ising19'),
                   ('graph_probability', 'ising_probability', '826 graph vs enriched Ising22')]
    comparisons += [(f+'_probability', f+'_opinion', 'Add current probabilities: '+l) for f, l, _ in FAMILIES]
    comparisons += [('ordered_arguments_'+s, 'ordered_choice_'+s, 'Arguments vs choice: '+l) for s, l in [('opinion', 'opinion state'), ('probability', 'probability state')]]
    comparisons += [('ordered_choice_'+s, 'graph_'+s, 'Ordered architecture vs graph: '+l) for s, l in [('opinion', 'opinion state'), ('probability', 'probability state')]]
    for left, right, label in comparisons:
        lines.append(f"| {label} | {reduction(contrast(result, left, right, 'kl'))} | {reduction(contrast(result, left, right, 'mae'), 100, 3)} |")
    lines += ['', 'The ordered-versus-graph comparison changes both architecture and available order; it does not isolate order alone. Within each ordered architecture, arguments and the choice-only control have identical parameter counts and actual positions.', '',
        '## Fitting, seed variation, and secondary accuracy', '',
        '| Model / state | Train KL | Validation KL | Validation MAE seed range, pp | Source modal-choice agreement | Sampled-majority accuracy (secondary) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for suffix, state_label in [('opinion', 'opinions'), ('probability', '+ probabilities')]:
        for family, label, _ in FAMILIES:
            name = family+'_'+suffix; tr, va = summary[name]['train'], summary[name]['validation']; span = seed_ranges[name]['mae']
            lines.append(f"| {label}, {state_label} | {tr['kl']:.5f} | {va['kl']:.5f} | {100*span['min']:.3f}–{100*span['max']:.3f} | {100*(1-va['mode_error']):.2f}% | {100*secondary_summary[name]['sampled_majority_accuracy']:.2f}% |")
    lines += ['', 'Modal-choice agreement compares the predictor’s more likely option with the source distribution’s more likely option. Sampled-majority accuracy compares it with a random recorded outcome. Neither is the same as matching the probability distribution. A successful change must be described using the metric it actually improves.', '',
        'The selected ordered choice model with probabilities has 12.492 pp training MAE and 12.772 pp validation MAE. Its remaining error is therefore already substantial on the training set, while its train–validation gap is small. This points toward investigating the fitted rule and available representation; it does not establish an information-theoretic limit or exclude effects of regularization and optimization.', '',
        '## Model and fitting definitions', '',
        r'\[\hat p_i=\sigma\left(\phi_i^\top w+\beta_+\sum_jJ^+_{ij}s_j+\beta_-\sum_jJ^-_{ij}s_j+\beta_0\sum_j|J_{ij}|s_j\right),\quad J^+=\max(J,0),\ J^-=\min(J,0).\]', '',
        'The original Ising baseline keeps 16 context coefficients plus three graph coefficients, and the author’s pinned two-stage optimizer. The 22-coefficient version adds the receiver’s mᵢ and signed friendly/unfriendly sums of neighboring mⱼ. It retains all original opinion fields. Both use fractional-target Bernoulli cross-entropy and the unchanged regularization; neither is tuned on the 80 validation questions. An independent training-gradient check confirms convergence below 1e−6 in standardized second-stage coordinates.', '',
        'Each neural condition tests the same nine learning-rate/output-penalty settings on the same internal **256 fit /64 tune** split within the training 320. Preprocessing is fitted on 256 during tuning and 320 during final fitting. Selection minimizes soft cross-entropy, equivalent to minimizing KL because the target entropy is fixed. Selected durations are refitted with seeds142/143/144. All eighteen neural refits and both Ising fits freeze before the outer80 are scored.', '',
        f'Execution completed **54 tuning fits, 18 neural refits, and 2 Ising fits**, using four local CPU workers. Neural fitting took {fit_minutes:.1f} elapsed minutes and {worker_minutes:.1f} summed worker-minutes. There was no new LLM generation or text-encoder training.', '',
        '## Data quality and limits', '',
        'There are 81,920 training and 20,480 validation agent updates, from 32 known personas and eight updates per conversation. All cases are retained. Validation target outside-A/B first-token mass averages '+f"{protocol['quality']['validation']['target']['other_mass_mean']:.3g}"+' and peaks at '+f"{protocol['quality']['validation']['target']['other_mass_max']:.6f}"+'. Training has '+str(protocol['quality']['train']['target']['other_mass_above_001'])+' target states with outside-A/B mass above 0.001, with a maximum of '+f"{protocol['quality']['train']['target']['other_mass_max']:.4f}"+'. These remain legitimate conditional A/B targets but are weaker approximations to the full parsed-response distribution.', '',
        'The 80 validation questions have been reused in earlier experiments and helped motivate this study. Results remain developmental. The 40 reserved assessment conversations stay ungenerated and unused. This study tests familiar signed graphs; it adds no unseen-graph distribution claim. Original source logs, prior experiments and the separate HPC3 collection are unchanged.', '',
        'Probability targets remove the random final choice from the label, but they do not guarantee zero error from a compressed state. Opinion-only predictors omit the realized arguments; argument predictors retain only compressed text and context. Under soft cross-entropy, the best predictor from available information X is the conditional mean E[p_source(t+1)|X]. Variation in the source probability that remains hidden from X can therefore still limit fidelity. The old sampled-label reference no longer applies, but this is not proof that the present inputs determine the full source distribution.', '',
        'The independent audit checks all input/model hashes and split boundaries; reconstructs current and next probabilities from source logs; verifies all 54 tuning fits and six selections; recomputes all **40 prediction files /8,000 seed-question rows /60 paired contrasts**, plus two fixed references and their six contrasts; replays all 18 neural checkpoints and both Ising fits; and verifies the intended input branches have trained gradients while target replacement leaves outputs unchanged. '+f"Maximum aggregate numerical difference is {audit['max_metric_difference']:.3g}."+' Six focused pre-training checks also passed.', '',
        'Artifacts: [experiment protocol](../experiments/distribution_prediction_320_v1/README.md), [frozen configuration](../configs/distribution_prediction_320.yaml), [numerical results](../runs/distribution_prediction_320_v1/results.json), [completion audit](../runs/distribution_prediction_320_v1/completion_audit.json), [training settings](../runs/distribution_prediction_320_v1/selection.json).', '']
    Path('reports/distribution_prediction_320.md').write_text('\n'.join(lines))
    json_write(report_dir/'summary.json', dict(completed_utc=study.now(), report_code_sha256=digest(__file__),
        metrics=summary, secondary_validation=secondary_summary, validation_seed_ranges=seed_ranges,
        elapsed_fit_minutes=fit_minutes, summed_worker_minutes=worker_minutes, assessment_loaded=False,
        provenance={str(root/name): digest(root/name) for name in ('protocol.json', 'prepared.json', 'models_frozen.json', 'results.json', 'completion_audit.json', 'reference_protocol.json', 'reference_results.json')},
        artifacts={name: digest(name) for name in ('reports/distribution_prediction_320.md', 'reports/distribution_prediction_320.png')}))
    print('Created audited distribution report and figure.', flush=True)


if __name__ == '__main__':
    main()
