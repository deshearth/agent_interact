import torch
from scripts import modal_token_focus as study


def test_key_normalization_keeps_initial_pooling_and_padding_safe():
    torch.manual_seed(42); x = torch.randn(3, 12, 32); mask = torch.arange(12)[None] < torch.tensor([10, 3, 0])[:, None]
    q = torch.zeros(4, 8, requires_grad=True)
    expected = study.tokens.pool_tokens(x, mask)
    actual = study.focused_pool(x, mask, q)
    torch.testing.assert_close(expected, actual, atol=2e-7, rtol=0)
    changed = x.clone(); changed[~mask] = 1000
    torch.testing.assert_close(actual, study.focused_pool(changed, mask, q), atol=0, rtol=0)
    actual[:, 0].sum().backward(); assert q.grad.abs().sum() > 0


def test_message_control_has_no_dead_query_and_real_batch_uses_message_mask():
    torch.set_num_threads(2); cfg = study.config(); data = study.tokens.Data('inner')
    p = study.tokens.read(study.tokens.ROOT/'protocol.json'); ids = data.ids(p['inner']['fit'])[:2]
    for name, spec in cfg['models'].items():
        data.text_kind = spec['text']; batch = data.batch(ids, 'mode'); model = study.make_model(cfg, name, 42)
        with torch.no_grad(): model.classifier[-1].weight[1].fill_(.03)
        z = study.core.score_model(model, batch); loss = torch.nn.functional.binary_cross_entropy_with_logits(z, batch['target']); loss.backward()
        assert torch.isfinite(loss) and model.token_projection.weight.grad.abs().sum() > 0
        if spec['message_only'] and spec['text'] == 'conclusion':
            assert model.token_query is None
            assert ((batch['token_features'][..., 32] > .5)&batch['token_mask']).sum(1).eq(1).all()
