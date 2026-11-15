import numpy as np
import torch
import time
from pathlib import Path
from scripts import modal_neural_study as study


def batch():
    graph = np.array([[[0, 1, -1], [1, 0, 1], [-1, 1, 0]]], dtype=np.int8)
    neighbors, signs = study.dist.ordered.pair.neighbor_slots(graph)
    return dict(s=torch.tensor([[1., -1., 1.]]), context=torch.randn(1, 3, 16), J=torch.tensor(graph, dtype=torch.float32),
        current_probability=torch.rand(1, 3), initial_probability=torch.rand(1, 3),
        question32=torch.randn(1, 3, 32), persona32=torch.randn(1, 3, 32), initial_population=torch.rand(1, 3, 32),
        text_arguments8=torch.randn(1, 3, 8), text_conclusion8=torch.randn(1, 3, 8),
        text_arguments40=torch.randn(1, 3, 40), text_conclusion40=torch.randn(1, 3, 40),
        target=torch.rand(1, 3), target_soft=torch.rand(1, 3), target_mode=torch.ones(1, 3),
        neighbors=torch.tensor(neighbors), signs=torch.tensor(signs), positions=torch.randn(1, 3, 2, 3))


def test_nano_parameter_counts_and_paired_argument_initialization():
    cfg = study.config()
    assert sum(p.numel() for p in study.make_model(cfg, 'nano_current_choice', 42).parameters()) == 978
    assert sum(p.numel() for p in study.make_model(cfg, 'nano_initial_choice', 42).parameters()) == 986
    a, b = [study.make_model(cfg, n, 42) for n in ('rich_choice', 'rich_arguments')]
    for x, y in zip(a.parameters(), b.parameters()): torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_future_targets_and_undeclared_text_never_enter_predictions():
    cfg = study.config(); cfg['training']['dropout'] = 0.; values = batch()
    for name in cfg['models']:
        model = study.make_model(cfg, name, 11).eval()
        with torch.no_grad(): model.classifier[-1].weight[1] = torch.linspace(-.5, .5, model.classifier[-1].in_features)
        first = study.score_model(model, values); changed = dict(values)
        for key in ('target', 'target_soft', 'target_mode'): changed[key] = 1-changed[key]
        torch.testing.assert_close(first, study.score_model(model, changed), atol=0, rtol=0)
        changed = dict(values); changed['text_arguments8'] = -values['text_arguments8']; changed['text_arguments40'] = -values['text_arguments40']
        other = study.score_model(model, changed)
        if cfg['models'][name]['text'] == 'arguments': assert (first-other).abs().max() > 1e-7
        else: torch.testing.assert_close(first, other, atol=0, rtol=0)


def test_modal_label_and_soft_training_target_are_separate():
    obj = study.Data.__new__(study.Data); obj.condition = 'soft'
    obj.values = {'target_soft': torch.tensor([[.49, .51, .9]]), 'target_mode': torch.tensor([[0., 1., 1.]]), 'target': torch.zeros(1, 3)}
    torch.testing.assert_close(obj.batch([0])['target'], obj.values['target_soft'])
    torch.testing.assert_close(obj.batch([0], 'mode')['target'], obj.values['target_mode'])


def test_complete_training_loop_writes_finished_artifacts(tmp_path):
    cfg = study.config(); cfg['output'] = str(tmp_path)
    cfg['training'].update(max_steps=2, validation_every_steps=1, batch_size=2, evaluation_batch_size=2)
    study.json_write(tmp_path/'protocol.json', {'synthetic': True}); study.json_write(tmp_path/'encoding.json', {})
    obj = study.Data.__new__(study.Data); obj.condition = 'mode'; obj.device = 'cpu'
    obj.qids = np.array(['fit', 'fit', 'inner', 'inner']); obj.y = np.array([[0, 1, 1], [1, 0, 1]]*2)
    obj.values = {k: v.repeat((4,)+(1,)*(v.ndim-1)) for k, v in batch().items()}
    obj.values['target_mode'] = torch.as_tensor(obj.y.astype(np.float32))
    obj.values['target_soft'] = obj.values['target_mode']*.8+.1
    protocol = {'fold': {'train_questions': ['fit'], 'validation_questions': ['inner']}}
    item = dict(phase='tune', width='nano_current_choice', condition='mode', candidate='c0', seed=42)
    with study.adapted(tmp_path/'worker_progress.json'):
        folder, record = study.trainer.train_one(cfg, protocol, obj, item, time.time()+30)
    assert record['status'] == 'complete' and record['steps'] == 2
    assert (folder/'best.pt').exists() and (folder/'validation_logp.npy').exists()
    assert record['validation']['questions'] == 1
