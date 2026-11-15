"""Report the audited all-case modal study and its training-only diagnostics."""
from datetime import datetime
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import evaluate_modal_accuracy as study

LABELS = {
    'reference_ising_opinion': 'Original Ising / soft target',
    'reference_ising_probability': 'Ising + current probability / soft target',
    'ising_current_mode': 'Ising + current probability / mode target',
    'reference_graph_probability': 'Graph transformer + probability / soft target',
    'reference_ordered_choice_probability': 'Previous ordered choice / soft target',
    'reference_ordered_arguments_probability': 'Previous ordered arguments / soft target',
    'nano_current_choice_mode': 'Direct-mode ordered choice',
    'nano_initial_choice_mode': 'Direct-mode choice + initial probability',
    'rich_choice_mode': 'Richer choice', 'rich_arguments_mode': 'Richer arguments',
    'paired_pca_choice_mode': 'Joint question/choice · PCA32',
    'paired_pca_arguments_mode': 'Joint question/arguments · PCA32',
    'paired_full_choice_mode': 'Joint question/choice · full vectors',
    'paired_full_arguments_mode': 'Joint question/arguments · full vectors',
    'history_choice_round_mode': 'Choice + round-only recurrent control',
    'history_choice_observed_mode': 'Choice + observed history',
    'history_arguments_round_mode': 'Arguments + round-only recurrent control',
    'history_arguments_observed_mode': 'Arguments + observed history',
    'tree_current': 'Current-state tree', 'tree_initial': 'Initial-state tree',
    'tree_rich': 'Rich-state tree', 'tree_lexical': 'Lexical tree',
    'tree_semantic8': 'Argument PCA8 tree', 'tree_semantic32': 'Argument PCA32 tree',
    'tokens_choice_mean_mode': 'Token projection + mean / choice',
    'tokens_arguments_mean_mode': 'Token projection + mean / arguments',
    'tokens_choice_attention_mode': 'Learned token attention / choice',
    'tokens_arguments_attention_mode': 'Learned token attention / arguments',
    'focus_choice_all_mode': 'Normalized token attention / choice',
    'focus_arguments_all_mode': 'Normalized token attention / arguments',
    'focus_choice_message_mode': 'Question-conditioned choice token',
    'focus_arguments_message_mode': 'Normalized attention / message words',
}


def label(name): return LABELS.get(name, name.replace('_', ' '))


def main():
    root = study.ROOT; result = study.read(root/'results.json'); audit = study.read(root/'completion_audit.json')
    assert audit['passed'] and audit['results_sha256'] == digest(root/'results.json')
    p = study.prepare(); summary = result['summary']; models = result['models']
    ft = study.read(study.FT/'results.json'); capacity = study.read(root.parent/'capacity_diagnostic_v1/results.json')
    population = study.read(study.population.ROOT/'results.json')
    assert not ft['promotion_gate_passed'] and not capacity['promotion_gate_passed']
    best = max(summary, key=lambda n: summary[n]['validation']['accuracy'])
    baseline = 'reference_ordered_choice_probability'
    differences = [c for c in result['paired_differences'] if c['left'] == best and c['right'] == baseline and c['metric'] == 'accuracy']
    gap = 100*(.95-summary[best]['validation']['accuracy'])
    regimes = []
    bins = [0, .025, .05, .1, .2, .3, .4, .5000001]
    for name in (baseline, best):
        for item in result['prediction_files']:
            if item['model'] != name or item['split'] != 'validation': continue
            d = dict(np.load(item['path'])); target = d['target_probability'] > .5
            correct = (d['logp'][..., 1] > d['logp'][..., 0]) == target
            margin = np.abs(d['target_probability']-.5)
            for lower, upper in zip(bins, bins[1:]):
                mask = (margin >= lower)&(margin < upper)
                regimes.append(dict(model=name, seed=item['seed'], lower=lower, upper=min(.5, upper), updates=int(mask.sum()),
                    share=float(mask.mean()), accuracy=float(correct[mask].mean()), fraction_of_errors=float((~correct & mask).sum()/(~correct).sum())))
    round_rows = []
    for name in (baseline, best):
        for item in result['prediction_files']:
            if item['model'] != name or item['split'] != 'validation': continue
            d = dict(np.load(item['path'])); correct = d['logp'].argmax(-1) == (d['target_probability'] > .5)
            for t in range(8):
                mask = d['pairs'][:, 1] == t
                round_rows.append(dict(model=name, seed=item['seed'], round=t+1, accuracy=float(correct[mask].mean())))
    json_write(root/'error_diagnostics.json', dict(source_margin_regimes=regimes, rounds=round_rows,
        policy='Post-fit descriptive breakdown only. Source probabilities define groups, never inference filters or model inputs at t+1. All cases remain in primary scores.'))
    chosen = ['reference_ising_opinion', 'ising_current_mode', baseline, 'nano_current_choice_mode',
        'rich_arguments_mode', 'paired_full_arguments_mode', 'history_choice_observed_mode', 'history_arguments_observed_mode',
        'tokens_arguments_attention_mode', 'tree_semantic8']
    if best not in chosen: chosen.append(best)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), gridspec_kw={'width_ratios': [1.35, 1]})
    fig.subplots_adjust(left=.28, right=.98, top=.82, bottom=.18, wspace=.28)
    ax = axes[0]
    for y, name in enumerate(chosen[::-1]):
        a, b = [100*summary[name][split]['accuracy'] for split in ('train', 'validation')]
        ax.plot([a, b], [y, y], color='#b9c4cb', linewidth=1.5)
        ax.scatter(a, y, marker='o', color='#7a8793', s=34, label='Training' if y == 0 else None)
        ax.scatter(b, y, marker='s', color='#2479a8', s=42, label='Validation' if y == 0 else None)
        ax.text(b+.15, y-.22, f'{b:.2f}', fontsize=8, color='#2479a8')
    ax.axvline(95, color='#b3473a', linestyle='--', linewidth=1.2)
    ax.text(95.15, len(chosen)-.7, '95% target', color='#b3473a', rotation=90, va='top')
    ax.set(yticks=range(len(chosen)), yticklabels=[label(n)+'\n'+(f"{models[n]['parameters']:,} learned" if models[n]['parameters'] else f"{models[n]['tree_nodes']:,} tree nodes") for n in chosen[::-1]],
        xlim=(82, 96.5), ylim=(-.8, len(chosen)-.1), xlabel='All-case source-mode accuracy (%)', title='Fitted models on 320 / 80 questions')
    ax.grid(axis='x', alpha=.15)
    fig.legend(*ax.get_legend_handles_labels(), loc='center left', bbox_to_anchor=(.28, .085), ncol=2, frameon=False)
    ax = axes[1]
    for kind, color in [('choice', '#2479a8'), ('arguments', '#c46c36')]:
        trajectory = study.read(study.CAPACITY/f'{kind}_256/history.json')
        trajectory = [r for r in trajectory if r['step'] > 0]
        for field, style, text in [('train', '-', 'training'), ('validation', '--', 'internal holdout')]:
            ax.plot([r['step'] for r in trajectory], [100*r[field]['accuracy'] for r in trajectory],
                linestyle=style, color=color, linewidth=1.7, label=kind.title()+' · '+text)
    ax.axhline(95, color='#b3473a', linestyle=':', linewidth=1)
    ax.set(xlabel='Optimization steps', ylabel='Accuracy (%)', ylim=(78, 100.5),
        title='Capacity check: 215,426 parameters\n256 fitting / 64 internal questions')
    ax.legend(loc='center right', fontsize=8, frameon=False); ax.grid(alpha=.15)
    fig.suptitle('Source-mode accuracy remains below 95%', x=.58, y=.96, fontsize=16)
    fig.text(.58, .90, 'Deterministic next-opinion target · every update retained · reused validation, exploratory', ha='center', fontsize=10)
    fig.text(.28, .035, 'Left: mean of three neural refits; frozen encoder parameters are additional. All updates are included.\nRight: one unregularized diagnostic per input, kept out of the final model comparison.', fontsize=9, color='#4b5964')
    figure = Path('reports/modal_accuracy_investigation_320.png'); fig.savefig(figure, dpi=170); plt.close(fig)
    date = datetime.now(ZoneInfo('America/Chicago')).strftime('%B %d, %Y, %I:%M %p Central')
    lines = ['# Investigating approximately95% source-mode accuracy', '', f'Completed {date}.', '',
        f"**Best observed validation accuracy: {100*summary[best]['validation']['accuracy']:.3f}%**, from {label(best)}. "
        f"The95% target remains {gap:.3f}percentage points away. No subset filtering or ensemble is used.", '']
    if differences:
        c = differences[0]; lines += [f"Change versus the previous978-parameter ordered-choice model: **{100*c['difference']:+.3f}pp**, paired95% interval "
            f"[{100*c['ci95'][0]:+.3f}, {100*c['ci95'][1]:+.3f}]. This is an exploratory comparison after repeated use of the same80 questions.", '']
    lines += ['The best neural model is the **2,306-parameter choice transformer with causal history: 88.441% validation accuracy**. '
        'Observed history adds **0.337 pp [0.094, 0.588]** over its identical-size round-only control. '
        'Replacing choice tokens with arguments changes accuracy by **−0.070 pp [−0.225, +0.081]** in this history model. '
        'Thus history helps modestly; an argument advantage remains unsupported in this comparison.', '']
    lines += ['## What was predicted', '',
        'For every agent and transition, the target is A iff the source\'s recorded next single-response conditional A/B probability exceeds0.5. '
        'The source probability comes from the fixed Llama prompt at temperature0.7. The source\'s old sampled-majority outcome is not the label here. '
        'The earlier91.4% sampling reference is therefore not a ceiling for this target. These are one-step predictions on existing stochastic histories, '
        'not newly generated deterministic-agent trajectories.', '',
        'There are320 training and80 reused validation questions, each with256 agent updates. Candidate selection uses only the internal256/64 split. '
        'All final models and diagnostic promotion decisions froze before outer scoring. The40 assessment questions remain unused. No new source calls were made.', '',
        '## Final model comparison', '',
        '| Model | Learned update parameters | Training accuracy | Validation accuracy |', '| --- | ---: | ---: | ---: |']
    order = list(dict.fromkeys(chosen+sorted(set(summary)-set(chosen))))
    for name in order:
        count = models[name]['parameters']; count = f'{count:,}' if count is not None else f"{models[name]['tree_nodes']:,} nodes"
        lines.append(f"| {label(name)} | {count} | {100*summary[name]['train']['accuracy']:.3f}% | {100*summary[name]['validation']['accuracy']:.3f}% |")
    lines += ['', 'Neural values average three individual refits, not a best seed or an ensemble. Frozen MiniLM has22,713,216 pretrained parameters '
        'in addition to these update counts. Trees are larger diagnostic models and are reported by node count rather than treated as parameter-matched transformers. '
        'Soft-target and modal-target losses optimize different quantities; modal NLL in the numerical results is not a source-distribution calibration score.', '',
        '## Hypothesis tests', '',
        '1. **Missing initial state or simple history:** Six tree feature sets test current distributions, initial probabilities, richer state, lexical cues and wider argument PCA. '
        'A training-only residual-memory correction was also tested; all fixed correction weights reduced training accuracy.',
        '2. **Wrong objective or insufficient width:** Direct modal classification is compared with source-probability supervision, with matched accuracy-based checkpoint selection. '
        'The richer models add question features, known-persona identity and the initial population state.',
        '3. **Arguments lose their connection to the question:** Question, answer options and delivered argument are encoded jointly, with matched question-plus-choice controls. '
        'Both PCA32 and full384 vectors are tested. All13,629,561 tokens are retained; four texts span multiple independent510-token windows. '
        'The full vectors improve fitting but do not automatically improve held-out prediction.',
        '4. **The frozen text encoder is unsuitable:** A three-epoch internal-only pilot adapts its last transformer layer at two rates, alongside a matched head-only continuation control. '
        'First five layers and embeddings remain frozen. This is a limited last-layer experiment, not an exhaustive test of end-to-end language-model training.',
        '5. **Past behavior contains additional state:** A16-unit recurrent component receives earlier opinions/probabilities and friendly/unfriendly population means, '
        'only through the current round. Identical round-only controls isolate the observed-history contribution.',
        '6. **Selected training errors hide fitting capacity:** A215,426-parameter model is trained without dropout or weight decay on8 and256 training questions. '
        'The small subset tests memorization; the larger subset exposes the training/generalization tradeoff.', '',
        '7. **Sentence averaging discards useful words:** Frozen contextual token vectors are reduced using training-only PCA32 and a learned token projection. '
        'Compare uniform pooling with four learned attention queries, for both choice-only and actual arguments. Every token is retained; five fixed token features '
        'mark the message region, uppercase A/B, digits and position. Attention adds32 parameters to the matched mean-pooling model. These runs use16-snapshot batches '
        'and the same two learning rates; their comparison with earlier models changes representation and batch size, while within-family controls match both.', '',
        '8. **Population observations reveal shared question context:** Append all32 current opinions and current/previous/cumulative probabilities to the state trees. '
        'On internal64, the current-state tree gains0.000pp and the lexical tree gains0.079pp. Neither crosses the+0.5pp promotion gate, so neither is scored on outer80. '
        'Nonneighbor states are tested as predictive proxies, not asserted causal influences.', '',
        '9. **The new token selector remains too uniform:** A training-only probe on two questions found normalized attention entropy0.999997. '
        'A follow-up normalizes each attention key to unit variance, without additional fitted parameters, and compares all-token pooling with message-only pooling. '
        'Each message token still sees its question/options in the frozen encoder. The one-token A/B control removes32 query parameters because no token choice is possible.', '',
        '## Encoder adaptation: internal64 only', '', '| Encoder rate | Trainable encoder weights | Selected epoch | Training accuracy | Internal accuracy |', '| ---: | ---: | ---: | ---: | ---: |']
    for r in ft['results']:
        lines.append(f"| {r['encoder_rate']:g} | {r['trainable_encoder_parameters']:,} | {r['selected_epoch']} | {100*(1-r['train']['error']):.3f}% | {100*r['inner_accuracy']:.3f}% |")
    lines += ['', f"Starting frozen-head accuracy was{100*ft['initial_accuracy']:.3f}%. The prespecified+0.5pp promotion gate was not reached; no encoder-adapted model was scored on outer80.", '',
        '## Capacity diagnostic: training-only development', '', '| Input | Fitting questions | Terminal training accuracy | Terminal internal accuracy | Best internal accuracy |', '| --- | ---: | ---: | ---: | ---: |']
    for r in capacity['results']:
        terminal = r['terminal']; v = terminal['validation']; b = r['selected']
        vs = f"{100*v['accuracy']:.3f}%" if v else '—'; bs = f"{100*b['validation']['accuracy']:.3f}%" if b else '—'
        lines.append(f"| {r['kind']} | {r['fitting_questions']} | {100*terminal['train']['accuracy']:.3f}% | {vs} | {bs} |")
    lines += ['', 'The8-question diagnostic uses preprocessing fitted on the inner256; it is a memorization check, not an8-question learning-curve estimate. '
        'No capacity model reached the89% internal promotion gate. Their terminal weights are not used as final predictors.', '',
        '## Remaining errors by source probability margin', '',
        'These are descriptive groups for the best observed model. Every group remains in the primary accuracy; the source\'s next probability is never used to select which cases to predict.', '',
        '| Margin from0.5 | Share of updates | Accuracy | Share of errors |', '| --- | ---: | ---: | ---: |']
    for lower, upper in zip(bins, bins[1:]):
        rows = [r for r in regimes if r['model'] == best and r['lower'] == lower]
        share, accuracy, errors = [100*np.mean([r[k] for r in rows]) for k in ('share', 'accuracy', 'fraction_of_errors')]
        lines.append(f'| {lower:g}–{min(.5, upper):g} | {share:.2f}% | {accuracy:.2f}% | {errors:.2f}% |')
    lines += ['',
        '## Interpretation and practical next experiment', '',
        'The experiments distinguish the ability to fit existing questions from the ability to transfer to new questions. Increasing capacity can raise training accuracy substantially '
        'while reducing held-out accuracy. This rules out the simple claim that the architecture cannot fit the labels at all. It does not prove that95% is unattainable, '
        'nor that arguments contain no predictive information.', '',
        'The remaining task is to learn transferable relations between the question, the actual reasoning in the messages, and the receiver\'s response. '
        'The next data test compares320 versus512 training questions, adding the first192 training units from the larger collection\'s preexisting frozen order. '
        'The old80 validation questions provide a fixed anchor, while newly collected validation conversations remain unused in this interim test. '
        'The larger bank uses revised topic sampling, so this measures data augmentation under that sampling policy, not a pure count effect with an identical question distribution. '
        'More steps or parameters on the same320 questions are not supported as a route to95% by these results. '
        'A larger corpus may help; these experiments do not guarantee that it will close the gap.', '',
        '## Verification and limitations', '',
        f"The independent audit reproduces{audit['prediction_files']} prediction files and{audit['seed_question_rows']:,} seed/question records, "
        f"checks{audit['contrasts']} paired contrasts, and replays all{audit['new_checkpoint_replays']} new final models with zero class disagreements. "
        f"Maximum independent metric discrepancy is{audit['independent_metric_max_difference']:.2g}. Target perturbations leave inference unchanged. "
        'History tests verify that future observations cannot change current inputs.', '',
        'Intervals use5,000 paired whole-question bootstrap draws, conditional on fitted models. They do not account for repeated validation, searching many models, '
        'training resampling or multiple comparisons. The best observed score is exploratory. Source probabilities are conditional first-token A/B probabilities; '
        'the checked small non-A/B mass and rare formatting exceptions from the previous audit still apply. Graphs and personas are familiar; this study does not measure transfer to new graphs or agents. '
        'Actual delivered arguments are supplied, so this is not autonomous rollout before future messages are generated.', '',
        'Failed logging, dtype and GPU-memory attempts are preserved separately and excluded from results. Encoder GPU settings were revised before any complete adaptation comparison '
        'or outer scoring. The independent HPC3 collection was not modified. After the shared SSH session was reconnected, '
        'the November15 07:49UTC controller snapshot showed515 training and130 validation conversations, with28 GPUs running. '
        'All first192 new training units in the frozen collection order were complete. Their export is a separate follow-up.', '',
        'Artifacts: [numerical results](../runs/modal_accuracy_investigation_320_v1/evaluation_v1/results.json), '
        '[audit](../runs/modal_accuracy_investigation_320_v1/evaluation_v1/completion_audit.json), '
        '[hypothesis journal](../experiments/modal_accuracy_investigation_320_v1/README.md), '
        '[figure](modal_accuracy_investigation_320.png).', '']
    clean = []
    for line in lines:
        if '](' not in line:
            line = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', line)
            line = re.sub(r'(?<=\d)(?=[A-DF-Za-df-z]|[eE](?![+-]?\d))', ' ', line)
        clean.append(line.replace('HPC 3', 'HPC3').replace('prespecified+', 'prespecified +').replace('the+', 'the +'))
    Path('reports/modal_accuracy_investigation_320.md').write_text('\n'.join(clean))
    json_write(root/'report_summary.json', dict(best_model=best, validation_accuracy=summary[best]['validation']['accuracy'],
        training_accuracy=summary[best]['train']['accuracy'], target_accuracy=.95, gap_pp=gap, comparison=differences,
        figure_sha256=digest(figure), report_sha256=digest('reports/modal_accuracy_investigation_320.md'), report_code_sha256=digest(__file__)))
    print(study.read(root/'report_summary.json'))


if __name__ == '__main__': main()
