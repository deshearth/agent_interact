"""Training-only test of whether the whole observed population reveals latent question context."""
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import modal_neural_study as core

ROOT = Path('runs/modal_accuracy_investigation_320_v1/population_screen_v1')


def population_features(spins, probabilities, pairs):
    e, t = pairs.T
    mean = probabilities.cumsum(1)[e, t]/(t[:, None]+1)
    lag = probabilities[e, np.maximum(t-1, 0)]
    return np.concatenate([spins[e, t], probabilities[e, t], mean, lag], -1).astype(np.float32)


def run():
    ROOT.mkdir(exist_ok=True); p = core.screen.validate(); common = dict(np.load(core.state.config()['common']))
    probability = np.load(Path(core.dist.config()['probability_reference'])/'probabilities.npz')['single']
    prototype = population_features(common['spins'], probability, common['pairs'])
    x = np.broadcast_to(prototype[:, None], (len(prototype), 32, 128)).reshape(-1, 128)
    labels = dict(np.load(core.screen.ROOT/'labels.npz')); fit = np.isin(labels['qids'], p['inner']['fit']); val = np.isin(labels['qids'], p['inner']['validation'])
    protocol = dict(code_sha256=digest(__file__), test_sha256=digest('tests/test_modal_population_screen.py'), inner=p['inner'], fold=p['fold'],
        screen_protocol_sha256=digest(core.screen.ROOT/'protocol.json'), settings=core.screen.SETTINGS, variants=['current', 'lexical'], promotion_gain=.005,
        policy='Append all32 current opinions, current probabilities, cumulative observed mean probabilities and previous probabilities to each receiver. '
            'These are predictive proxies for shared question context; this is not a claim of causal influence from nonneighbors. '
            'Current and lexical state trees use exactly their previous fixed training settings. Select on internal64 only, and promote only for at least0.5pp '
            'accuracy gain over the corresponding original internal score. No outer80 scoring, answer keys or future inputs.',
        source_calls=0, assessment_loaded=False, outer_validation_scored=False)
    if (ROOT/'protocol.json').exists(): assert core.read(ROOT/'protocol.json') == protocol
    else: json_write(ROOT/'protocol.json', protocol)
    results = []
    for variant in protocol['variants']:
        folder = ROOT/variant; folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists(): results.append(core.read(folder/'complete.json')); continue
        original = np.load(core.screen.ROOT/'features/inner'/f'{variant}.npy', mmap_mode='r')
        features = np.concatenate([original, x], -1)
        model = HistGradientBoostingClassifier(**core.screen.SETTINGS, early_stopping=True)
        best = None; selected = None; history = []
        with threadpool_limits(limits=2):
            model.fit(features[fit], labels['target'][fit], X_val=features[val], y_val=labels['target'][val])
            for iteration, prediction in enumerate(model.staged_predict_proba(features[val]), 1):
                pr = np.clip(prediction[:, 1], 1e-10, 1-1e-10); y = labels['target'][val]
                accuracy = float(((pr > .5) == y).mean()); nll = float((-y*np.log(pr)-(1-y)*np.log1p(-pr)).mean())
                history.append(dict(iteration=iteration, accuracy=accuracy, modal_nll=nll)); key = -accuracy, nll, iteration
                if best is None or key < best: best = key; selected = pr.copy()
        reference = core.read(core.screen.ROOT/'screen'/variant/'complete.json')['inner_accuracy']
        np.save(folder/'inner_probability.npy', selected); json_write(folder/'history.json', history)
        record = dict(variant=variant, inner_accuracy=-best[0], inner_nll=best[1], selected_iterations=best[2], original_inner_accuracy=reference,
            gain=-best[0]-reference, promotion_gate_passed=-best[0] >= reference+protocol['promotion_gain'], completed_utc=core.now(),
            artifacts={f.name: digest(f) for f in folder.iterdir() if f.is_file()})
        json_write(folder/'complete.json', record); results.append(record); print(record, flush=True)
    json_write(ROOT/'results.json', dict(results=results, promotion_gate_passed=any(r['promotion_gate_passed'] for r in results),
        completed_utc=core.now(), outer_validation_scored=False, assessment_loaded=False))


if __name__ == '__main__': run()
