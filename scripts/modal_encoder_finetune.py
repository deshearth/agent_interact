"""Training-only test of adapting the last MiniLM layer for source-mode prediction."""
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest, json_write, seed_all
from scripts import modal_question_arguments as paired

ROOT = Path('runs/modal_accuracy_investigation_320_v1/encoder_finetune_screen_v1')
SETTINGS = dict(encoder_rates=[0., .00001, .00003], head_rate=.0001, epochs=3, batch_graphs=4,
                token_batch_size=32, seed=42, warmup_steps=32, weight_decay=.001, gradient_clip=1., promotion_gain=.005)
KEYS = ['s', 'context', 'J', 'current_probability', 'initial_probability', 'question32', 'persona32',
        'initial_population', 'neighbors', 'signs', 'positions']


def encode_ids(encoder, requested, tokens, token_spec, device, token_batch=64):
    """Exact complete-window token-sum pooling; differentiable through the encoder."""
    shape = requested.shape; unique, inverse = np.unique(requested.ravel(), return_inverse=True)
    chunks = [(j, values[start:start+510]) for j, text_id in enumerate(unique) if text_id
              for values in [tokens[int(text_id)]] for start in range(0, len(values), 510)]
    sums = torch.zeros((len(unique), encoder.config.hidden_size), device=device)
    for begin in range(0, len(chunks), token_batch):
        pack = chunks[begin:begin+token_batch]
        # Four padded sequence shapes and one batch shape bound MPS graph-cache memory.
        width = min(512, ((max(len(t) for j, t in pack)+2+127)//128)*128)
        ids = torch.full((token_batch, width), token_spec.pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids); pooling = torch.zeros_like(ids)
        ids[:, 0] = token_spec.cls; ids[:, 1] = token_spec.sep; mask[:, :2] = 1
        for j, (_, values) in enumerate(pack):
            ids[j, :len(values)+2] = torch.tensor([token_spec.cls]+values+[token_spec.sep], device=device)
            mask[j, :len(values)+2] = 1; pooling[j, 1:len(values)+1] = 1
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        values = (hidden*pooling[..., None]).sum(1)[:len(pack)]
        sums = sums.index_add(0, torch.tensor([j for j, values in pack], device=device), values)
    output = F.normalize(sums, dim=-1)
    return output[torch.as_tensor(inverse, device=device)].reshape(*shape, encoder.config.hidden_size)


def run():
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoModel, AutoTokenizer
    cfg = paired.config(); parent_root = Path(cfg['output']); p = paired.validate(cfg)
    frozen = paired.read(parent_root/'models_frozen.json')
    for path, sha in frozen['artifacts'].items(): assert digest(path) == sha, path
    selected = next(c for c in paired.read(parent_root/'selection.json')['choices'] if c['width'] == 'paired_full_arguments')
    checkpoint_path = Path(selected['tuning_directory'])/'best.pt'
    selected_record = paired.read(checkpoint_path.parent/'manifest.json')
    assert selected_record['phase'] == 'tune' and selected_record['train']['questions'] == 256
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(settings=SETTINGS, source_protocol_sha256=digest(parent_root/'protocol.json'),
        source_models_sha256=digest(parent_root/'models_frozen.json'), initial_checkpoint=str(checkpoint_path), initial_checkpoint_sha256=digest(checkpoint_path),
        code_files={str(f): digest(f) for f in [Path(__file__), Path('tests/test_modal_encoder_finetune.py'), Path(paired.__file__), Path(paired.core.__file__)]},
        fold=p['fold'], inner=p['inner'], policy='Inner256/64 only. Warm-start the selected inner256 full-vector argument head. Compare head-only continuation '
            'against jointly adapting the last MiniLM transformer layer at two rates. First five layers, embeddings and other encoder parameters stay frozen. '
            'Every model sees the same three epochs and warmup, and selects mode accuracy then NLL on internal64. Complete tokens, no next-state inputs. '
            'No outer80 scoring. Promote only if inner accuracy improves by at least0.5pp over the frozen head at initialization.',
        assessment_loaded=False, source_calls=0, outer_validation_scored=False)
    if (ROOT/'protocol.json').exists(): assert paired.read(ROOT/'protocol.json') == protocol
    else: json_write(ROOT/'protocol.json', protocol)
    torch.set_num_threads(2); device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    data = paired.Data('inner'); fit, validation = [data.ids(p['inner'][k]) for k in ('fit', 'validation')]
    assert len(set(data.qids[fit])) == 256 and len(set(data.qids[validation])) == 64
    message_ids = np.load(parent_root/'data/message_ids.npy'); e, t = data.common['pairs'].T
    all_ids = message_ids[e, t]; frozen_vectors = np.load(parent_root/'data/vectors.npy', mmap_mode='r')
    tokenizer = AutoTokenizer.from_pretrained(cfg['encoder_snapshot'], local_files_only=True)
    text = paired.read(parent_root/'data/texts.json')
    tokens = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)['input_ids']; tokens[0] = []
    spec = SimpleNamespace(pad=tokenizer.pad_token_id, cls=tokenizer.cls_token_id, sep=tokenizer.sep_token_id)
    initial = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
    results = []
    for arm, rate in enumerate(SETTINGS['encoder_rates']):
        folder = ROOT/f'arm{arm}'; folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists(): results.append(paired.read(folder/'complete.json')); continue
        seed_all(SETTINGS['seed']); head = paired.core.make_model(cfg, 'paired_full_arguments').to(device); head.load_state_dict(initial)
        encoder = AutoModel.from_pretrained(cfg['encoder_snapshot'], local_files_only=True).to(device).eval()
        for parameter in encoder.parameters(): parameter.requires_grad_(False)
        if rate:
            for parameter in encoder.encoder.layer[-1].parameters(): parameter.requires_grad_(True)
        trainable = [v for v in encoder.parameters() if v.requires_grad]
        with torch.inference_mode():
            sample_ids = all_ids[fit[:2]]
            encoded = encode_ids(encoder, sample_ids, tokens, spec, device).cpu().numpy()
            difference = float(np.max(np.abs(encoded-frozen_vectors[sample_ids])))
            assert difference < 5e-6
        optimizer = torch.optim.AdamW([dict(params=head.parameters(), lr=SETTINGS['head_rate'], base_lr=SETTINGS['head_rate'])]
            + ([dict(params=trainable, lr=rate, base_lr=rate)] if trainable else []), weight_decay=SETTINGS['weight_decay'])
        def inputs(ids):
            values = data.batch(ids, 'mode'); batch = {k: values[k].to(device) for k in KEYS}
            embeddings = encode_ids(encoder, all_ids[ids], tokens, spec, device, SETTINGS['token_batch_size']) if rate else torch.as_tensor(np.array(frozen_vectors[all_ids[ids]]), device=device)
            lexical = values['text_arguments392'][..., -8:].to(device)
            batch['text_arguments392'] = torch.cat([embeddings, lexical], -1)
            return batch, values['target'].to(device)
        def assess(ids):
            head.eval(); encoder.eval(); arrays = []
            with torch.inference_mode():
                for begin in range(0, len(ids), 16):
                    batch, target = inputs(ids[begin:begin+16]); score = paired.core.score_model(head, batch)
                    arrays.append(torch.stack([-F.softplus(score), -F.softplus(-score)], -1).cpu().numpy())
            logp = np.concatenate(arrays); metrics, rows = paired.core.dist.base.metrics(logp, data.y[ids], data.qids[ids])
            return metrics, logp
        history = []; best = None; steps = 0; start = time.monotonic()
        for epoch in range(SETTINGS['epochs']+1):
            if epoch:
                order = np.random.default_rng(SETTINGS['seed']+epoch).permutation(fit)
                for begin in range(0, len(order), SETTINGS['batch_graphs']):
                    head.train(); encoder.eval(); optimizer.zero_grad(set_to_none=True)
                    batch, target = inputs(order[begin:begin+SETTINGS['batch_graphs']]); score = paired.core.score_model(head, batch)
                    loss = F.binary_cross_entropy_with_logits(score, target)+.001*score.square().mean()
                    assert torch.isfinite(loss); loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(head.parameters())+trainable, SETTINGS['gradient_clip'], error_if_nonfinite=True)
                    for group in optimizer.param_groups: group['lr'] = group['base_lr']*min((steps+1)/SETTINGS['warmup_steps'], 1.)
                    optimizer.step(); steps += 1
                    if steps % 16 == 0:
                        json_write(ROOT/'progress.json', dict(arm=arm, encoder_rate=rate, epoch=epoch, step=steps, updated_utc=paired.now()))
                        if device == 'mps': torch.mps.empty_cache()
            metrics, logp = assess(validation); key = metrics['error'], metrics['nll']
            history.append(dict(epoch=epoch, step=steps, validation=metrics, seconds=time.monotonic()-start))
            if best is None or key < best:
                best = key
                torch.save(dict(head={k: v.detach().cpu() for k, v in head.state_dict().items()},
                    encoder_last_layer={k: v.detach().cpu() for k, v in encoder.encoder.layer[-1].state_dict().items()}, epoch=epoch, step=steps), folder/'best.pt')
                np.save(folder/'inner_logp.npy', logp)
            json_write(folder/'history.json', history)
            print({'arm': arm, 'encoder_rate': rate, 'epoch': epoch, 'inner_accuracy': 1-metrics['error'], 'seconds': time.monotonic()-start}, flush=True)
        selected_weights = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        head.load_state_dict(selected_weights['head']); encoder.encoder.layer[-1].load_state_dict(selected_weights['encoder_last_layer'])
        train_metrics, train_logp = assess(fit); np.save(folder/'train_logp.npy', train_logp)
        record = dict(arm=arm, encoder_rate=rate, selected_epoch=selected_weights['epoch'], selected_step=selected_weights['step'],
            inner_accuracy=1-best[0], inner_nll=best[1], train=train_metrics, seconds=time.monotonic()-start,
            trainable_head_parameters=sum(v.numel() for v in head.parameters()), trainable_encoder_parameters=sum(v.numel() for v in trainable),
            initial_embedding_max_difference=difference, device=device, completed_utc=paired.now(), artifacts={f.name: digest(f) for f in folder.iterdir() if f.is_file()})
        json_write(folder/'complete.json', record); results.append(record)
        del optimizer, encoder, head
        if device == 'mps': torch.mps.empty_cache()
    initial_accuracy = 1-paired.read(ROOT/'arm0/history.json')[0]['validation']['error']
    best = max(results, key=lambda r: (r['inner_accuracy'], -r['inner_nll']))
    json_write(ROOT/'results.json', dict(completed_utc=paired.now(), initial_accuracy=initial_accuracy, results=results,
        selected_arm=best['arm'], promotion_gate_passed=best['inner_accuracy'] >= initial_accuracy+SETTINGS['promotion_gain'],
        outer_validation_scored=False, assessment_loaded=False))
    json_write(ROOT/'progress.json', dict(phase='complete', updated_utc=paired.now()))


if __name__ == '__main__': run()
