import torch
from scripts import modal_token_pooling as study


def test_attention_starts_as_mean_pooling_and_ignores_padding():
    torch.manual_seed(42); x = torch.randn(3, 10, 32); mask = torch.arange(10)[None] < torch.tensor([7, 4, 0])[:, None]
    q = torch.zeros(4, 8, requires_grad=True)
    expected = study.pool_tokens(x, mask)
    actual = study.pool_tokens(x, mask, q)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=0)
    changed = x.clone(); changed[~mask] = 1000
    torch.testing.assert_close(study.pool_tokens(changed, mask, q), actual, atol=0, rtol=0)
    assert not actual[2].any()
    actual[:, 0].sum().backward(); assert q.grad.abs().sum() > 0


def test_added_token_features_use_only_supplied_text_and_positions():
    text = 'Question: A?\nOption A: 3\nOption B: 4\nMessage: B 4'
    offsets = [(text.index('A?'), text.index('A?')+1), (len(text)-3, len(text)-2), (len(text)-1, len(text))]
    f = study.extra_features(text, offsets)
    assert f[:, 0].tolist() == [0, 1, 1]
    assert f[:, 1].tolist() == [1, 0, 0]
    assert f[:, 2].tolist() == [0, 1, 0]
    assert f[:, 3].tolist() == [0, 0, 1]


def test_full_pooling_model_forward_backward_on_padded_messages():
    from scripts import modal_neural_study as core
    torch.set_num_threads(2); data = core.Data('inner'); p = core.validate(core.config()); ids = data.ids(p['inner']['fit'])[:2]
    batch = data.batch(ids, 'mode'); batch.update(token_features=torch.randn(5, 12, 37),
        token_mask=torch.tensor([[True]*8+[False]*4]*5), token_inverse=torch.randint(0, 5, batch['s'].shape))
    cfg = study.config()
    for name in cfg['models']:
        model = study.make_model(cfg, name, 42)
        with torch.no_grad(): model.classifier[-1].weight[1].fill_(.05)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(core.score_model(model, batch), batch['target'])
        loss.backward(); assert torch.isfinite(loss)
        assert model.token_projection.weight.grad.abs().sum() > 0
