"""Consolidate four frozen, audited cycles and the full-prompt noise reference."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write

ROOT = Path('runs/generalization_investigation_320_v1')
OUT = ROOT/'report_v1'
CYCLES = ['state_models_v1', 'distillation_v1', 'ordered_transformer_v1', 'lexical_v1']

def read(path): return json.loads(Path(path).read_text())
def pc(value): return f'{value*100:.3f}%'

def contrast(result, left, right, metric='error'):
    return next(c for c in result['paired_differences'] if c['left'] == left and c['right'] == right and c['metric'] == metric)

def gain(c):
    return f"{-100*c['difference']:+.3f} pp [{-100*c['ci95'][1]:+.3f}, {-100*c['ci95'][0]:+.3f}]"

def main():
    OUT.mkdir(parents=True, exist_ok=True); results, provenance = {}, {}
    for cycle in CYCLES:
        folder = ROOT/cycle; audit = read(folder/'completion_audit.json'); result = read(folder/'results.json')
        assert audit['passed'] and audit['results_sha256'] == digest(folder/'results.json')
        results[cycle] = result
        for name in ('protocol.json', 'prepared.json', 'models_frozen.json', 'results.json', 'completion_audit.json'):
            provenance[str(folder/name)] = digest(folder/name)
    noise = read(ROOT/'noise_audit_v1/audit.json'); provenance[str(ROOT/'noise_audit_v1/audit.json')] = digest(ROOT/'noise_audit_v1/audit.json')
    p = read(ROOT/'state_models_v1/protocol.json'); qs = sorted(p['fold']['validation_questions'])
    teacher_data = np.load(ROOT/'noise_audit_v1/probabilities.npz', allow_pickle=False)
    teacher = np.clip(teacher_data['majority'], 0, 1); pair_qids = teacher_data['qids']
    common = np.load('runs/question_breadth_320_train_v1/predictors/development.npz'); e, t = common['pairs'].T
    hard = (common['spins'][e, t+1] == 1).astype(int)
    manifest = [dict(key=r['model'], **r) for r in read('runs/compact_arguments_320_v1/results/predictions.json') if r['model'] in ('I0', 'opinion_nano')]
    for cycle, result in results.items(): manifest += [dict(key=cycle+'/'+r['model'], **r) for r in result['prediction_files']]
    cases, question_scores = defaultdict(list), defaultdict(list)
    for item in manifest:
        if item['split'] != 'validation': continue
        assert digest(item['path']) == item['sha256']; d = np.load(item['path'])
        take = np.isin(pair_qids, qs); q = teacher[take].ravel(); y = hard[take].ravel(); qids = np.repeat(pair_qids[take], 32)
        if 'logp' in d:
            probability = np.exp(d['logp'].astype(float)[..., 1]).ravel(); predicted = d['logp'].argmax(-1).ravel()
            actual_qids = np.repeat(d['qids'], 32)
        else: probability = d['probability']; predicted = (probability > .5).astype(int); actual_qids = d['qids']
        np.testing.assert_array_equal(actual_qids, qids); np.testing.assert_array_equal(d['target'].ravel(), y)
        expected_error = np.where(predicted == 1, 1-q, q); floor = np.minimum(q, 1-q)
        regret = np.abs(2*q-1)*(predicted != (q > .5))
        np.testing.assert_allclose(expected_error, floor+regret, atol=1e-15)
        prob = np.clip(probability, 1e-10, 1-1e-10)
        expected_nll = -q*np.log(prob)-(1-q)*np.log1p(-prob)
        values = {'observed_accuracy': (predicted == y).astype(float), 'expected_accuracy': 1-expected_error,
            'sampling_error': floor, 'excess_error': regret, 'expected_nll': expected_nll}
        for question in qs:
            keep = qids == question
            question_scores[item['key']].append(dict(qid=question, seed=item.get('seed', -1), **{k: float(v[keep].mean()) for k, v in values.items()}))
        confidence = np.maximum(q, 1-q)
        for low, high in zip([.5, .6, .7, .8, .9, .95, .99], [.6, .7, .8, .9, .95, .99, 1.0000001]):
            keep = (confidence >= low)&(confidence < high)
            cases[item['key']].append({'seed': item.get('seed', -1), 'confidence_lower': low, 'confidence_upper': min(high, 1),
                'updates': int(keep.sum()), 'share': float(keep.mean()),
                **{k: float(v[keep].mean()) for k, v in values.items()},
                'observed_error_contribution': float(((predicted != y)&keep).mean())})
    summaries = {model: {m: float(np.mean([r[m] for r in rows])) for m in ('observed_accuracy', 'expected_accuracy', 'sampling_error', 'excess_error', 'expected_nll')}
        for model, rows in question_scores.items()}
    state, distill, neural, lexical = [results[c] for c in CYCLES]
    best_neural = max(neural['summary'], key=lambda m: (1-neural['summary'][m]['validation']['error']) if m not in ('I0', 'opinion_nano') else -1)
    candidate_keys = [c+'/'+m for c, r in results.items() for m in r['summary'] if m not in ('I0', 'opinion_nano')]
    best = max(candidate_keys, key=lambda m: summaries[m]['observed_accuracy'])
    best_tree = max([k for k in candidate_keys if not k.startswith('ordered_transformer')], key=lambda m: summaries[m]['observed_accuracy'])
    old = summaries['opinion_nano']; best_score = summaries[best]
    regime_groups = {}
    for model in ('opinion_nano', best):
        regime_groups[model] = []
        for label, low, high in [('50–80%', .5, .8), ('80–99%', .8, .99), ('99–100%', .99, 1.000001)]:
            rows = [r for r in cases[model] if low <= r['confidence_lower'] < high]
            seeds = len({r['seed'] for r in rows}); weight = sum(r['share'] for r in rows)
            regime_groups[model].append({'source_confidence': label, 'share': weight/seeds,
                'observed_accuracy': sum(r['share']*r['observed_accuracy'] for r in rows)/weight,
                'sampling_error_contribution': sum(r['share']*r['sampling_error'] for r in rows)/seeds})
    # The factorial interaction is exploratory and labeled as such.
    nr = read(ROOT/'ordered_transformer_v1/by_question.json')
    lookup = {(r['model'], r['qid']): r for r in nr if r['split'] == 'validation'}
    interaction = np.array([(lookup['actual_conclusion', q]['error']-lookup['actual_arguments', q]['error'])-
        (lookup['random_conclusion', q]['error']-lookup['random_arguments', q]['error']) for q in qs])
    rng = np.random.default_rng(81948); draws = rng.integers(80, size=(5000, 80))
    interaction_result = {'accuracy_gain_difference': float(interaction.mean()), 'ci95': np.quantile(interaction[draws].mean(1), [.025, .975]).tolist(),
        'scope': 'Exploratory additional factorial contrast, not a new model-selection criterion.'}
    json_write(OUT/'summary.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'models': summaries,
        'largest_observed_accuracy_variant': best, 'best_neural_variant': best_neural, 'factorial_interaction': interaction_result,
        'hypothesis_cycles': 4, 'new_tuning_fits': 96, 'new_refits': 26, 'assessment_loaded': False,
        'target_accuracy': .95, 'target_achieved': False, 'report_code_sha256': digest(__file__)})
    json_write(OUT/'by_question.json', dict(question_scores)); json_write(OUT/'source_confidence_regimes.json', dict(cases))
    json_write(OUT/'source_confidence_groups.json', regime_groups)
    # Scientific figure: realized prediction scores and a separate expected-error decomposition.
    model_keys = ['I0', 'opinion_nano', 'ordered_transformer_v1/'+best_neural, best_tree]
    model_labels = ['Ising (19 coefficients)', 'Original transformer (818)', 'Ordered arguments (970)', 'Rich state + option wording (trees)']
    colors = ['#7e8a97', '#4b72a9', '#168a82', '#a55799']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.3), gridspec_kw={'width_ratios': [1.05, 1]}, layout='constrained')
    positions = np.arange(4)[::-1]
    for yy, key, color in zip(positions, model_keys, colors):
        value = 100*summaries[key]['observed_accuracy']; ax.scatter(value, yy, s=65, color=color, zorder=3)
        ax.text(value+.16, yy+.10, f'{value:.2f}%', color=color, fontsize=10)
    ref = noise['summary']['validation']; ref_x = 100*ref['reference_expected_accuracy']; ci = np.array(ref['reference_expected_accuracy_ci95'])*100
    ax.axvspan(*ci, color='#d8dce2', alpha=.6, zorder=0); ax.axvline(ref_x, color='#58616f', linestyle='--', linewidth=1.3)
    ax.axvline(95, color='#b24a3d', linestyle=':', linewidth=1.7)
    ax.text(ref_x, 3.65, 'Full-prompt reference\n91.4% expected', ha='center', va='bottom', fontsize=9, color='#47515e')
    ax.text(95, -.67, '95% target', ha='center', color='#b24a3d', fontsize=9)
    ax.set(yticks=positions, yticklabels=model_labels, xlim=(81.5, 96), ylim=(-.9, 4.3), xlabel='Validation accuracy (%)', title='Prediction scores on all 80 questions')
    ax.grid(axis='x', alpha=.18)
    show = ['opinion_nano', 'ordered_transformer_v1/'+best_neural, best_tree]
    short = ['Original 818', 'Ordered arguments 970', 'Rich state + option wording']
    yy = np.arange(3)[::-1]; floors = np.array([summaries[k]['sampling_error']*100 for k in show]); excess = np.array([summaries[k]['excess_error']*100 for k in show])
    bx.barh(yy, floors, color='#e2ad68', height=.55, label='Source sampling uncertainty')
    bx.barh(yy, excess, left=floors, color='#5b83a4', height=.55, label='Error beyond full-prompt reference')
    for i, a, b in zip(yy, floors, excess):
        bx.text(a/2, i, f'{a:.2f}', ha='center', va='center'); bx.text(a+b/2, i, f'{b:.2f}', ha='center', va='center', color='white')
    bx.set(yticks=yy, yticklabels=short, xlim=(0, 20), ylim=(-1.1, 3.2), xlabel='Expected error (%)', title='Sampling explains part of the gap')
    bx.legend(loc='lower left', fontsize=8.5, frameon=False)
    fig.suptitle('Generalization study: message order helps, sampling limits accuracy', fontsize=15)
    fig.supxlabel('Same 80 reused validation questions. Shading: approximate source reference with a 95% question-bootstrap interval.', fontsize=8)
    fig.savefig('reports/generalization_investigation_320.png', dpi=180)
    plt.close(fig)
    def table(result, names):
        lines = ['| Model | Training accuracy | Validation accuracy | Validation NLL |', '| --- | ---: | ---: | ---: |']
        for name, label in names:
            train, val = result['summary'][name]['train'], result['summary'][name]['validation']
            lines.append(f"| {label} | {pc(1-train['error'])} | {pc(1-val['error'])} | {val['nll']:.5f} |")
        return lines
    lines = ['# Generalization investigation: four hypothesis cycles', '',
        'Completed on the fixed MATH/Llama **320 training /80 validation** cohort. The target was approximately 95% next-opinion accuracy. **That target was not achieved.** The best observed accuracy in this investigation is '+pc(best_score['observed_accuracy'])+'. The new 970-parameter ordered argument transformer reaches '+pc(summaries['ordered_transformer_v1/'+best_neural]['observed_accuracy'])+'.', '',
        'The full source audit estimates a **91.415% best expected accuracy** for a predictor given the source model’s full-prompt response probabilities, under the checked first-token/parsed-response approximation. The 95% target is therefore unsupported for this particular sampled-majority outcome. This does not prevent improvement from 84–85% toward the reference, and it is not a universal ceiling for other targets or datasets.', '',
        '![Observed validation accuracy and expected-error decomposition](generalization_investigation_320.png)', '',
        '## Task and evaluation boundary', '',
        'Predict each agent’s next majority-of-five A/B opinion, not the mathematical answer. There are 81,920 training and 20,480 validation agent updates: 32 known personas, eight updates per question, familiar signed graphs J0–J7. Argument predictors see messages already generated for that update. This is conditional prediction after delivery, not autonomous rollout before message generation.', '',
        'All new hyperparameter/duration choices use an internal 256-fit/64-tune whole-question split drawn only from the 320 training questions, stratified by graph and retained/new origin. Context and message PCA are fitted on 256 during tuning and 320 for final fitting. Every cycle freezes all final models before outer 80 scoring. Neural results average three fresh seed scores; trees use one deterministic fit. No seed ensembling or best-seed reporting.', '',
        '**The 80 validation questions have been reused throughout the project, including to motivate later hypotheses.** Results and paired intervals are exploratory, not a fresh confirmatory test. Pointwise 5,000-draw question-bootstrap intervals do not adjust for multiple experiments or model-selection uncertainty. The 40 reserved assessment conversations remain ungenerated and unused.', '',
        '## Cycle 1: missing state, nonlinear fits, and exact message order', '',
        'Hypothesis: the original state omits useful order/history/context, or small attention misses nonlinear interactions. Six boosted-tree feature variants used the same six-recipe search. “Base” contains current opinions, signed-neighbor summaries and original 16 context features. “Rich” adds actual order, current/past vote confidence, 32 question PCs, known-persona indicators and current/initial global vote summaries. “Arguments” adds 64 PCA8 message summaries by relationship/stance and first/last messages.', '']
    lines += table(state, [('I0', 'Ising (19)'), ('opinion_nano', 'Original transformer (818)'), ('base', 'Base state trees'), ('random_order', 'Independent random order'), ('order', 'Actual order'), ('history', 'Vote/history state'), ('rich', 'Rich state'), ('arguments', 'Rich state + PCA8 arguments')])
    lines += ['', 'Actual order versus random order: **'+gain(contrast(state, 'order', 'random_order'))+'** accuracy. Rich state versus original 818: **'+gain(contrast(state, 'rich', 'opinion_nano'))+'**. Adding compressed argument summaries to rich state does not clearly improve accuracy. These trees are larger models, not parameter-matched substitutes for 818.', '',
        'Revision: actual inbox order is a concrete missing variable. Broader state improves transfer, while simply providing compressed semantics still leaves a large gap.', '',
        '## Source uncertainty: verified on the full 400 conversations', '',
        'The export read saved source logs on a short HPC3 CPU job; it made zero source-model calls. All 400 episode hashes, 115,200 reconstructed prompt hashes and 576,000 finalized raw votes matched. Reconstructing the complete prompts verifies that the order used in these experiments is the actual source order, not an assumed historical order.', '',
        'For a source first-vote probability p of choosing A, the majority-of-five probability is', '',
        r'\[q=M_5(p)=10p^3(1-p)^2+5p^4(1-p)+p^5.\]', '',
        'Even a predictor that knows q exactly must choose between stochastic outcomes. Its best expected accuracy is', '',
        r'\[\mathbb{E}[\max(q,1-q)]=91.415\%,\qquad \mathbb{E}[\min(q,1-q)]=8.585\%.\]', '',
        'The whole-question 95% interval for expected accuracy is **90.617–92.191%**. Its realized accuracy on the saved labels is **91.548%**. The retained 20 and added 60 validation questions give **91.424% and 91.412%** expected accuracy, respectively; the old small-cohort result was not an anomaly.', '',
        'Assumptions and checks: first-token A/B probabilities are recorded at the actual sampling temperature 0.7 and agree across all five repetitions. They approximate parsed complete-response probabilities, with independent request-level sampling. On validation update responses, outside-A/B first-token mass averages 1.54e-7; only 2 of 102,400 initial responses are not bare A/B, and no final vote is invalid. Validation vote and majority calibration-bias intervals include zero. Later-token parsing anomalies are not mathematically bounded by first-token mass alone, so this remains a carefully checked approximation rather than an exact impossibility proof.', '',
        f"For the original 818 model, expected error is {100*(1-old['expected_accuracy']):.3f}%: {100*old['sampling_error']:.3f} points of sampling uncertainty and {100*old['excess_error']:.3f} points beyond the full-prompt reference. Thus sampling explains about {100*old['sampling_error']/(1-old['expected_accuracy']):.1f}% of its expected error, not all of it.", '',
        '## Cycle 2: more precise training targets', '',
        'Hypothesis: replace noisy training majority labels by source-majority probabilities. Only training questions supply teacher labels; probabilities never enter predictor inputs. 38 of 81,920 training labels fall back to the observed outcome due to nontrivial outside-A/B mass, non-bare responses or retries. No case is removed. Hard and soft controls use the same weighted Bernoulli representation, features, search and internal observed-label NLL selection.', '']
    lines += table(distill, [(n, n.replace('_', ' ')) for n in ['order_hard', 'order_soft', 'rich_hard', 'rich_soft', 'arguments_hard', 'arguments_soft']])
    lines += ['', 'Accuracy changes, soft versus hard: '+ '; '.join(f"{f}: {gain(contrast(distill, f+'_soft', f+'_hard'))}" for f in ('order', 'rich', 'arguments'))+'. All three intervals include zero. Rich/argument probability estimates improve in NLL, but more precise labels alone do not solve the accuracy gap. Hard-label controls reproduce cycle 1 exactly.', '',
        'Revision: sampling noise matters, but input representation and conditional response modeling remain limiting. A numerical guard caught probability-polynomial roundoff before the first attempt could evaluate validation; that unscored attempt was archived, corrected and rerun without reusing fits.', '',
        '## Cycle 3: transfer the order signal to a compact transformer', '',
        'The shared pair encoder receives sender state/text, relationship sign and three within-sign position features: normalized rank, first and last. It uses ordinary scaled-dot-product attention. All four conditions have **970 learned parameters**, 24 more than the preceding 946-parameter pair-token model. The text encoder remains frozen MiniLM; its 22.7M pretrained parameters and PCA fitting are separate from the 970 count.', '']
    lines += table(neural, [('random_conclusion', 'Random position, choice only'), ('actual_conclusion', 'Actual position, choice only'), ('random_arguments', 'Random position, arguments'), ('actual_arguments', 'Actual position, arguments')])
    lines += ['', 'Actual order versus random order for arguments: **'+gain(contrast(neural, 'actual_arguments', 'random_arguments'))+'**. Ordered arguments versus original 818: **'+gain(contrast(neural, 'actual_arguments', 'opinion_nano'))+'**. Arguments versus the matched ordered choice-only control: **'+gain(contrast(neural, 'actual_arguments', 'actual_conclusion'))+'**.', '',
        'This is a small positive argument result once actual order is represented. It does not establish that all argument representations work or that the entire remaining gap is positional. The additional exploratory order×argument accuracy interaction is '+f"{100*interaction.mean():+.3f} pp [{100*interaction_result['ci95'][0]:+.3f}, {100*interaction_result['ci95'][1]:+.3f}]"+'.', '',
        '## Cycle 4: preserve explicit option wording', '',
        'Hypothesis: semantic compression blurs simple option-specific details. Eight fixed textual features replace each PCA8 message vector: A/B counts, first/last uppercase mention, explicit option-A/B counts, last explicit option mention, and message length. The same 64 receiver summaries make the lexical model 255-dimensional, matching the rich+PCA8 argument model. A combined variant appends both representations. A final mention can refer to an option being criticized; it is a feature, not a guaranteed extracted recommendation.', '']
    lines += table(lexical, [('lexical', 'Rich state + option wording'), ('combined', 'Rich state + PCA8 + option wording')])
    lines += ['', 'Explicit option wording versus PCA8 argument summaries: **'+gain(contrast(lexical, 'lexical', 'arguments'))+'**. Versus original 818: **'+gain(contrast(lexical, 'lexical', 'opinion_nano'))+'**. The gain over rich state without arguments is uncertain: '+gain(contrast(lexical, 'lexical', 'rich'))+'. This supports a narrow representation issue, not a claim that MiniLM is universally unsuitable.', '',
        '## Where the remaining errors occur', '',
        'A retrospective split by the source model’s own majority confidence explains why improvements remain modest. These source probabilities are diagnostic references, not available inputs to the small predictor or a basis for filtering its primary score.', '',
        '| Source majority confidence | Share of all updates | Original 818 accuracy | Best new tree accuracy | Sampling-error contribution to total |',
        '| --- | ---: | ---: | ---: | ---: |']
    for old_group, new_group in zip(regime_groups['opinion_nano'], regime_groups[best]):
        lines.append(f"| {old_group['source_confidence']} | {pc(old_group['share'])} | {pc(old_group['observed_accuracy'])} | {pc(new_group['observed_accuracy'])} | {100*old_group['sampling_error_contribution']:.3f} pp |")
    low = regime_groups['opinion_nano'][0]
    lines += ['', f"The {100*low['share']:.1f}% of updates with source confidence below 80% contribute {100*low['sampling_error_contribution']:.2f} percentage points of irreducible error by themselves under the checked source approximation—already above a 5% error budget. Most observed improvement occurs in more decisive cases; the small model still misses some of those, leaving room for better representations and more data. Every update remains included in the primary accuracy.", '',
        '## What this establishes and what remains open', '',
        'There is now a verified, specific improvement: represent actual message order. It benefits both a larger tree model and a 970-parameter ordinary-attention model, and it reveals a small matched argument benefit. Richer context and transparent option wording also improve exploratory scores. The strongest simple signal was already captured well by the 19-coefficient Ising model; complexity alone cannot supply omitted information or predict fresh sampling randomness.', '',
        f"The best observed variant still has {100*(1-best_score['observed_accuracy']):.3f}% error. Under the source-probability diagnostic its expected accuracy is {pc(best_score['expected_accuracy'])}, leaving {100*best_score['excess_error']:.3f} predictable percentage points below the full-prompt reference. It closes {100*(old['excess_error']-best_score['excess_error'])/old['excess_error']:.1f}% of the original 818 model's excess expected error. The remaining gap is not fully explained by these experiments.", '',
        'A scientifically useful update law here should predict a response distribution, for example', '',
        r'\[q_i=F_\theta(\text{question, persona, signed ordered messages, available state}),\quad s_i^{t+1}=2\operatorname{Bernoulli}(q_i)-1.\]', '',
        'Keep observed accuracy as a metric, but compare probability loss and excess error above the source reference as well. The independently running 3,200/800 collection can test whether the newly identified state variables transfer with more questions. It was not altered by this investigation. Achieving 95% would require evidence that this noise estimate is wrong or an explicitly different target/protocol; neither is claimed here.', '',
        '## Verification and artifacts', '',
        '**96 tuning fits and 26 final refits completed**, plus the separately recorded failed unscored attempt. All four cycle audits pass. Saved validation predictions replay from checkpoints without class differences; scores and paired intervals independently reproduce. Source/prompt identity, training-only transforms, target isolation, position masks, padding and weighted fractional loss were checked. The prediction/contrast totals and complexity metadata are in the audit files below.', '',
        '- [Hypothesis journal](../experiments/generalization_investigation_320_v1/README.md)',
        '- [Source audit](../runs/generalization_investigation_320_v1/noise_audit_v1/audit.json)',
        '- [State models and audit](../runs/generalization_investigation_320_v1/state_models_v1/completion_audit.json)',
        '- [Probability supervision and audit](../runs/generalization_investigation_320_v1/distillation_v1/completion_audit.json)',
        '- [Ordered transformer and audit](../runs/generalization_investigation_320_v1/ordered_transformer_v1/completion_audit.json)',
        '- [Explicit option wording and audit](../runs/generalization_investigation_320_v1/lexical_v1/completion_audit.json)',
        '- [Consolidated numerical results](../runs/generalization_investigation_320_v1/report_v1/summary.json)',
        '- [Saved source-confidence error regimes](../runs/generalization_investigation_320_v1/report_v1/source_confidence_regimes.json)', '']
    Path('reports/generalization_investigation_320.md').write_text('\n'.join(lines))
    complexity = {}
    for cycle in CYCLES:
        if cycle == 'ordered_transformer_v1': complexity[cycle] = {'parameters': 970, 'refit_models': 12}; continue
        complexity[cycle] = {f.parent.name: {k: read(f)[k] for k in ('tree_count', 'tree_nodes')} for f in (ROOT/cycle/'refit').glob('*/complete.json')}
    json_write(OUT/'complexity.json', complexity)
    outputs = [OUT/p for p in ('summary.json', 'by_question.json', 'source_confidence_regimes.json', 'source_confidence_groups.json', 'complexity.json')]
    outputs += [Path('reports/generalization_investigation_320.md'), Path('reports/generalization_investigation_320.png')]
    json_write(OUT/'report_artifacts.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': provenance,
        'report_code_sha256': digest(__file__), 'outputs': {str(path): digest(path) for path in outputs}, 'assessment_loaded': False})
    print(json.dumps({'best': best, 'best_metrics': best_score, 'best_neural': best_neural,
        'neural_metrics': summaries['ordered_transformer_v1/'+best_neural], 'factorial_interaction': interaction_result}), flush=True)

if __name__ == '__main__': main()
