"""Audit the two training-only promotion screens, including actual weight replay."""
from pathlib import Path
import copy
import os
import sys
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write
from scripts import evaluate_modal_accuracy as evaluation
from scripts import modal_encoder_finetune as ft
from scripts import modal_capacity_diagnostic as capacity


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoModel, AutoTokenizer
    from types import SimpleNamespace
    torch.set_num_threads(2); paired = evaluation.paired; data = paired.Data('inner')
    p = paired.validate(paired.config()); fit, val = [data.ids(p['inner'][key]) for key in ('fit', 'validation')]
    results = paired.read(ft.ROOT/'results.json'); protocol = paired.read(ft.ROOT/'protocol.json')
    for path, sha in protocol['code_files'].items(): assert digest(path) == sha, path
    assert not results['outer_validation_scored'] and not results['assessment_loaded']
    assert results['promotion_gate_passed'] == (max(r['inner_accuracy'] for r in results['results']) >= results['initial_accuracy']+ft.SETTINGS['promotion_gain'])
    parent = Path(paired.config()['output']); e, t = data.common['pairs'].T
    message_ids = np.load(parent/'data/message_ids.npy')[e, t]
    sample = val[:8]; selected_ids = message_ids[sample]
    texts = paired.read(parent/'data/texts.json'); cfg = paired.config()
    tokenizer = AutoTokenizer.from_pretrained(cfg['encoder_snapshot'], local_files_only=True)
    unique = np.unique(selected_ids); tokens = [[] for _ in texts]
    parsed = tokenizer([texts[i] for i in unique], add_special_tokens=False, truncation=False, verbose=False)['input_ids']
    for i, value in zip(unique, parsed): tokens[int(i)] = value
    token_spec = SimpleNamespace(pad=tokenizer.pad_token_id, cls=tokenizer.cls_token_id, sep=tokenizer.sep_token_id)
    encoder = AutoModel.from_pretrained(cfg['encoder_snapshot'], local_files_only=True).eval()
    initial_last = {k: v.clone() for k, v in encoder.encoder.layer[-1].state_dict().items()}
    max_metric, max_replay, disagreements = 0., 0., 0
    replay_records = []
    for record in results['results']:
        folder = ft.ROOT/f"arm{record['arm']}"
        for name, sha in record['artifacts'].items(): assert digest(folder/name) == sha, name
        for split, ids, path in [('train', fit, 'train_logp.npy'), ('inner', val, 'inner_logp.npy')]:
            lp = np.load(folder/path); target = data.source[ids] > .5
            got = dict(error=float((lp.argmax(-1) != target).mean()), nll=float(-np.where(target, lp[..., 1], lp[..., 0]).astype(float).mean()),
                       brier_two_class=float((2*(np.exp(lp[..., 1].astype(float))-target)**2).mean()))
            expected = record['train'] if split == 'train' else dict(error=1-record['inner_accuracy'], nll=record['inner_nll'])
            for metric in expected.keys() & got.keys():
                delta = abs(got[metric]-expected[metric]); max_metric = max(max_metric, delta); assert delta < 1e-12
        weights = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        encoder.encoder.layer[-1].load_state_dict(weights['encoder_last_layer'])
        encoder_delta = max(float((weights['encoder_last_layer'][k]-initial_last[k]).abs().max()) for k in initial_last)
        assert (encoder_delta == 0) if record['encoder_rate'] == 0 else (encoder_delta > 0)
        head = paired.core.make_model(cfg, 'paired_full_arguments'); head.load_state_dict(weights['head']); head.eval()
        batch = data.batch(sample, 'mode'); values = {k: batch[k] for k in ft.KEYS}
        with torch.inference_mode():
            encoded = ft.encode_ids(encoder, selected_ids, tokens, token_spec, 'cpu', ft.SETTINGS['token_batch_size'])
            values['text_arguments392'] = torch.cat([encoded, batch['text_arguments392'][..., -8:]], -1)
            score = paired.core.score_model(head, values)
            lp = torch.stack([-F.softplus(score), -F.softplus(-score)], -1).numpy()
        saved = np.load(folder/'inner_logp.npy')[:len(sample)]
        delta = float(np.abs(lp-saved).max()); max_replay = max(max_replay, delta)
        mismatches = int((lp.argmax(-1) != saved.argmax(-1)).sum()); disagreements += mismatches
        assert delta < 2e-4 and mismatches == 0
        replay_records.append(dict(arm=record['arm'], updates=int(score.numel()), max_logp_difference=delta, encoder_last_layer_max_change=encoder_delta))
    c = paired.read(capacity.ROOT/'results.json'); cp = paired.read(capacity.ROOT/'protocol.json')
    assert cp['code_sha256'] == digest(capacity.__file__) and not c['outer_validation_scored']
    capacity_replays = 0
    for record in c['results']:
        folder = capacity.ROOT/f"{record['kind']}_{record['fitting_questions']}"
        for name, sha in record['artifacts'].items(): assert digest(folder/name) == sha
        runtime = copy.deepcopy(cfg); name = 'paired_full_'+record['kind']; runtime['models'][name]['width'] = cp['settings']['width']; runtime['training']['dropout'] = 0.
        head = paired.core.make_model(runtime, name)
        head.load_state_dict(torch.load(folder/'terminal.pt', map_location='cpu', weights_only=False)['model'])
        ids = data.ids(cp['subset8']) if record['fitting_questions'] == 8 else fit
        for split, selected in [('train', ids)]+([('validation', val)] if record['fitting_questions'] == 256 else []):
            lp = evaluation.predict(head, data, selected); got, _ = evaluation.metrics(lp, data.source[selected], data.qids[selected])
            expected = record['terminal'][split]
            for metric in evaluation.METRICS:
                delta = abs(got[metric]-expected[metric]); max_metric = max(max_metric, delta); assert delta < 1e-12
        capacity_replays += 1
    out = evaluation.ROOT.parent/'diagnostics_v1/completion_audit.json'
    json_write(out, dict(passed=True, completed_utc=paired.now(), audit_code_sha256=digest(__file__),
        encoder_results_sha256=digest(ft.ROOT/'results.json'), capacity_results_sha256=digest(capacity.ROOT/'results.json'),
        encoder_replays=replay_records, encoder_replay_max_logp_difference=max_replay, encoder_replay_class_disagreements=disagreements,
        capacity_terminal_models_replayed=capacity_replays, metric_max_difference=max_metric, outer_validation_scored=False, assessment_loaded=False))
    print(paired.read(out))


if __name__ == '__main__': main()
