"""Readable report and standalone scientific figure from audited frozen results."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import pair_token_attention_study as study

def report():
    cfg = study.config(); root = Path(cfg['output']); result = study.read(root/'results/summary.json')
    audit = study.read(root/'completion_audit.json'); assert audit['status'] == 'passed'
    values = result['summary']; parameters = result['parameters']
    started = min(datetime.fromisoformat(study.read(p)['created_utc']) for p in (root/'tune').glob('*/manifest.json'))
    finished = datetime.fromisoformat(study.read(root/'models_frozen.json')['frozen_utc'])
    elapsed_minutes = (finished-started).total_seconds()/60
    local_completion = datetime.fromisoformat(result['completed_utc']).astimezone(ZoneInfo('America/Chicago'))
    widths = ['pico_pca8', 'nano_pca8', 'pico_full', 'nano_full']
    labels = ['Width 4 / PCA8', 'Width 8 / PCA8', 'Width 4 / full384', 'Width 8 / full384']
    def acc(name, split='validation'): return 100*(1-values[name][split]['error'])
    def contrast(left, right, metric):
        return next(r for r in result['paired_differences'] if (r['left'], r['right'], r['metric']) == (left, right, metric))
    def gain(row, scale=1):
        decimals = 3 if scale == 100 else 5
        return f"{-scale*row['mean_difference']:+.{decimals}f} [{-scale*row['ci95'][1]:+.{decimals}f}, {-scale*row['ci95'][0]:+.{decimals}f}]"

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelsize': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), gridspec_kw={'wspace': .26})
    series = [('signed_', 'conclusion', 'Previous signed: choice', '#8793a2', 'o', -.21),
              ('signed_', 'arguments', 'Previous signed: arguments', '#b25c3b', 'o', -.07),
              ('pair_', 'conclusion', 'Ordinary attention: choice', '#4576b5', 's', .07),
              ('pair_', 'arguments', 'Ordinary attention: arguments', '#1c8b76', 's', .21)]
    for prefix, condition, label, color, marker, offset in series:
        names = [prefix+w+'_'+condition for w in widths]
        for ax, metric in zip(axes, ['accuracy', 'nll']):
            y = [acc(n) if metric == 'accuracy' else values[n]['validation']['nll'] for n in names]
            ax.scatter(np.arange(4)+offset, y, label=label, color=color, marker=marker, s=46, zorder=3)
    for ax, metric in zip(axes, ['accuracy', 'nll']):
        for baseline, label, color in [('I0', 'Ising19', '#7a6483'), ('opinion_nano', 'Original opinion818', '#444444')]:
            y = acc(baseline) if metric == 'accuracy' else values[baseline]['validation']['nll']
            ax.axhline(y, color=color, linestyle='--', linewidth=1, alpha=.8, label=label)
        ax.set_xticks(range(4), [x.replace(' / ', '\n') for x in labels]); ax.set_xlim(-.5, 3.5)
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    axes[0].set_ylabel('Validation accuracy (%)  ↑'); axes[1].set_ylabel('Validation NLL  ↓')
    axes[0].set_title('A/B prediction accuracy', loc='left', weight='bold', pad=12)
    axes[1].set_title('Probability prediction quality', loc='left', weight='bold', pad=12)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=10)
    fig.suptitle('Does ordinary attention improve the argument model?', fontsize=16, weight='bold', y=.99)
    fig.subplots_adjust(top=.73, bottom=.19, left=.08, right=.97)
    fig.text(.08, .045, 'Same 320 training / 80 validation questions · Mean of 3 fresh training seeds\n'
             'Reused validation: exploratory comparison. Frozen MiniLM encoder; assessment questions unused.',
             fontsize=9, color='#4c5663')
    plot = Path('reports/pair_token_attention_320.png'); fig.savefig(plot, dpi=180, facecolor='white')
    fig.savefig(plot.with_suffix('.pdf'), facecolor='white'); plt.close(fig)

    lines = ['# Ordinary attention with pair-specific friendship inputs', '',
        f"Completed {local_completion:%B %d, %Y at %I:%M %p} Central. All 72 tuning fits and 24 fresh-seed refits are complete; "
        f"recorded fitting totals {result['new_fit_seconds']/60:.1f} worker-minutes, with {elapsed_minutes:.1f} minutes "
        'elapsed from the first fit to model freezing. Independent model sizes run concurrently.', '',
        'The new architecture encodes the sender state and the receiver–sender relationship together, '
        'then uses ordinary scaled dot-product attention. The source conversations, frozen text vectors, '
        'question/persona context, opinions, graph, tuning grid and training seeds match the previous study. '
        'Old predictions are reused without modification.', '',
        'The ordinary-attention replacement provides no clear improvement. With compressed arguments, accuracy changes '
        'by only +0.007 / +0.015 percentage points from the previous width-four/eight argument models; both paired intervals '
        'span zero. The width-eight probability loss is essentially unchanged, while width-four NLL worsens. '
        'Full-vector argument models have lower validation accuracy point estimates than before, although those architecture '
        'difference intervals span zero. Neither compressed model has a clear argument advantage over its new choice-only control.', '',
        'The full-vector models retain the overfitting pattern: arguments raise training accuracy by about 0.68 /1.02 points '
        'but lower validation accuracy by 0.83 /0.64 points versus their new matched controls. Replacing the special '
        'attention mechanisms with shared relationship tokens therefore does not resolve the generalization problem in this experiment. '
        'This is evidence about this architecture bundle and training protocol, not a proof that arguments carry no predictive information.', '',
        '![Comparison](pair_token_attention_320.png)', '',
        '## Validation comparison', '',
        'Each row reports three-seed mean scores. Higher accuracy and lower NLL are better. '
        'The architecture comparison changes token encoding and removes three special attention mechanisms together.', '',
        '| Hidden / text dimensions | Parameters, previous → new | Input | Previous accuracy | New accuracy | Previous NLL | New NLL |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for w, label in zip(widths, labels):
        for cond in ('conclusion', 'arguments'):
            old, new = 'signed_'+w+'_'+cond, 'pair_'+w+'_'+cond
            lines.append(f"| {label} | {parameters[old]:,} → {parameters[new]:,} | {'Choice only' if cond=='conclusion' else 'Arguments'} | "
                         f"{acc(old):.3f}% | {acc(new):.3f}% | {values[old]['validation']['nll']:.5f} | {values[new]['validation']['nll']:.5f} |")
    lines += ['', '| Earlier baseline | Parameters | Training accuracy | Validation accuracy | Validation NLL |',
              '| --- | ---: | ---: | ---: | ---: |']
    for n, label in [('I0', 'Ising'), ('opinion_pico', 'Opinion/context, width 4'), ('opinion_nano', 'Opinion/context, width 8')]:
        lines.append(f"| {label} | {parameters[n]:,} | {acc(n,'train'):.3f}% | {acc(n):.3f}% | {values[n]['validation']['nll']:.5f} |")
    lines += ['', '## Paired changes and training fit', '',
        'Intervals below are 95% paired question bootstrap intervals, averaging the three seed scores before resampling '
        'the 80 validation questions. Positive values mean improvement for the new model or argument condition. '
        'These are descriptive pointwise intervals conditional on fitted models; they do not include tuning uncertainty '
        'or resampled training seeds.', '',
        '| Model | New versus previous argument accuracy gain, pp | New versus previous argument NLL reduction | Arguments versus new choice control: accuracy gain, pp | Arguments versus new choice control: NLL reduction |',
        '| --- | --- | --- | --- | --- |']
    for w, label in zip(widths, labels):
        new, old, ctrl = 'pair_'+w+'_arguments', 'signed_'+w+'_arguments', 'pair_'+w+'_conclusion'
        lines.append(f"| {label} | {gain(contrast(new,old,'error'),100)} | {gain(contrast(new,old,'nll'))} | "
                     f"{gain(contrast(new,ctrl,'error'),100)} | {gain(contrast(new,ctrl,'nll'))} |")
    lines += ['', '| Model | Previous choice train accuracy | Previous argument train accuracy | New choice train accuracy | New argument train accuracy |',
              '| --- | ---: | ---: | ---: | ---: |']
    for w, label in zip(widths, labels):
        names = ['signed_'+w+'_conclusion', 'signed_'+w+'_arguments', 'pair_'+w+'_conclusion', 'pair_'+w+'_arguments']
        lines.append('| '+label+' | '+' | '.join(f'{acc(n,"train"):.3f}%' for n in names)+' |')
    lines += ['', '## Exact architecture and comparison boundary', '',
        r'$$h_i=E([\mathrm{onehot}(s_i),c_i,\log(1+d_i^+),\log(1+d_i^-),\log(1+d_i)]),\qquad n_i=\mathrm{LN}(h_i)$$', '',
        r'$$u_j=n_j+W_{\rm text}\widetilde e_j,\qquad z_{i\leftarrow j}=\mathrm{LN}(\mathrm{GELU}(W_{\rm pair}[u_j;\tau_{ij}]+b_{\rm pair}))$$', '',
        r'$$q_i=W_Q n_i,\quad k_{ij}=W_K z_{i\leftarrow j},\quad v_{ij}=W_V z_{i\leftarrow j},\quad '
        r'\alpha_{ij}=\frac{\exp(q_i^\top k_{ij}/\sqrt{d_k})}{\sum_{\ell\in\mathcal N(i)}\exp(q_i^\top k_{i\ell}/\sqrt{d_k})}$$', '',
        r'$$m_i=W_O\operatorname{concat}_{\rm heads}\left(\sum_{j\in\mathcal N(i)}\alpha_{ij}v_{ij}\right).$$', '',
        'The same residual, dropout, feed-forward block and classifier follow. Isolated agents get zero social contribution. '
        'There is one shared pair encoder and no learned receiver/sender IDs. Friendship is an input value +1 or −1. '
        'Different receivers can therefore have different tokens for the same sender. The graph still determines who can communicate.', '',
        'The new attention has no edge-type additive bias, signed value multiplier or degree-dependent social multiplier. '
        'Graph degree still enters the agent-state encoder. The pair encoder has a linear layer, GELU and learned LayerNorm; '
        'this is the nonlinear refinement discussed in chat, not the earlier 322-parameter linear sketch. '
        'Width-four models match their old parameter counts; width-eight models have 64 extra parameters.', '',
        'Only these small predictors are trained. MiniLM has 22,713,216 pretrained, frozen parameters. '
        'PCA8 and the existing 16 context features are also frozen, fitted using training inputs. '
        'The comparison retains ordinary PCA8 to isolate the architecture change from the separate within-question PCA experiment. '
        'Each model forecasts the next majority-of-five opinion after the actual source messages are available; '
        'this is conditional one-step prediction, not autonomous conversation simulation.', '',
        '## Training protocol', '',
        'Training uses 320 questions /81,920 agent transitions; validation uses 80 questions /20,480 transitions. '
        'There are eight familiar signed graphs, 32 personas and eight opinion updates per conversation. '
        'The 40 reserved assessment conversations are ungenerated, unloaded and unscored.', '',
        'Each architecture and condition receives the same nine learning-rate/output-penalty combinations and validation stopping '
        'policy as the old experiment. Candidate selection uses seed 42 and lowest validation NLL. Settings and duration freeze '
        'before refits at seeds 142, 143 and 144; every seed contributes equally. The output classifier starts at zero, '
        'and every shared initial parameter matches the same-width previous architecture at that seed. '
        'Training used the faster measured CPU backend with four independent workers, four Torch threads per worker, batch 32, AdamW, dropout 0.1, '
        'gradient clipping 1, validation every 512 steps, a 32,768-step cap and patience 10.', '',
        '| Width / text | Input | Selected candidate | Learning rate | Output penalty | Refit updates |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in study.read(root/'selection.json')['choices']:
        candidate = next(c for c in study.base.candidates(cfg) if c['id'] == r['candidate'])
        lines.append(f"| {r['width']} | {r['condition']} | {r['candidate']} | {candidate['learning_rate']} | "
                     f"{candidate['penalty']} | {r['refit_steps']:,} |")
    lines += ['', '## Offline text sensitivity', '',
        'At evaluation only, messages are replaced by another round from the same question, sender and current stance. '
        'The same substitutions as the prior study are used. Receiver state, full history and round are not matched, '
        'and donors can be from later rounds. This checks content sensitivity, not causal argument quality or usable prospective inputs.', '',
        '| Model | Actual versus substituted accuracy gain, pp | Mean probability change, pp |',
        '| --- | --- | ---: |']
    for w, label in zip(widths, labels):
        name = 'pair_'+w+'_arguments'
        sensitivity = np.mean([r['probability_mae'] for r in result['text_sensitivity'] if r['model'] == name])*100
        lines.append(f"| {label} | {gain(contrast(name,name+'_matched','error'),100)} | {sensitivity:.3f} |")
    lines += ['', '## Verification and artifacts', '',
        f"Five focused tests pass. The independent final audit reproduces {audit['prediction_arrays_recomputed']} prediction arrays "
        f"and {audit['paired_contrasts_recomputed']} paired contrasts; maximum aggregate discrepancy is {audit['maximum_metric_difference']:.3g}. "
        f"It verifies {audit['source_episodes_verified']} source episode hashes, {audit['prior_artifacts_verified']} prior artifacts "
        f"and {audit['frozen_artifacts_verified']} new frozen artifacts. All 24 trained models have active text and relationship gradients, "
        'changed projection/token weights and matching dense/packed attention predictions. Prior artifacts and the independent HPC3 collection remain unchanged.', '',
        '- [Experiment protocol and commands](../experiments/pair_token_attention_320_v1/README.md)',
        '- [Machine-readable results](../runs/pair_token_attention_320_v1/results/summary.json)',
        '- [Independent audit](../runs/pair_token_attention_320_v1/completion_audit.json)',
        '- [PDF figure](pair_token_attention_320.pdf)', '',
        'This reused-validation comparison cannot establish performance on untouched questions or unseen graphs. '
        'It tests the specified architecture and frozen text representation; it does not establish that arguments are intrinsically uninformative.', '']
    Path('reports/pair_token_attention_320.md').write_text('\n'.join(lines))
    json_write(root/'report_manifest.json', {'results_sha256': digest(root/'results/summary.json'),
        'audit_sha256': digest(root/'completion_audit.json'), 'reporting_code_sha256': digest(__file__),
        'files': {str(p): digest(p) for p in [plot, plot.with_suffix('.pdf'), Path('reports/pair_token_attention_320.md')]}})
    print('Report and figure generated from audited results.', flush=True)

if __name__ == '__main__': report()
