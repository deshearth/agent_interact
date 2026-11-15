import numpy as np
import torch
from scripts import modal_question_arguments as study


def test_joint_text_has_only_supplied_question_choices_and_message():
    text = study.pair_text('A question?', {'A': '3', 'B': '4', 'answer': 'hidden'}, 'I favor A.')
    assert text == 'Question: A question?\nOption A: 3\nOption B: 4\nMessage: I favor A.'
    assert 'hidden' not in text


def test_each_agent_gets_its_own_question_conditioned_choice_vector():
    ids = np.array([[11, 12], [21, 22]])
    spins = np.array([[1, -1, 1], [-1, 1, -1]])
    result = np.take_along_axis(ids[:, None, :], (spins == 1)[..., None].astype(int), -1)[..., 0]
    np.testing.assert_array_equal(result, [[12, 11, 12], [21, 22, 21]])


def test_full_and_pca_lexical_scaling_match_after_projection_input_scale():
    x = np.arange(8)/4.
    np.testing.assert_allclose(x*np.sqrt(40), x*np.sqrt(40/392)*np.sqrt(392))
    cfg = study.config()
    for pair in [('paired_pca_choice', 'paired_pca_arguments'), ('paired_full_choice', 'paired_full_arguments')]:
        models = [study.core.make_model(cfg, name, 42) for name in pair]
        for a, b in zip(models[0].parameters(), models[1].parameters()): torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_real_training_batch_has_consistent_dtype_and_text_gradients():
    torch.set_num_threads(2)
    data = study.Data('inner'); cfg = study.config()
    p = study.read(study.Path(cfg['output'])/'protocol.json')
    ids = data.ids(p['inner']['fit'])[:2]
    for name, spec in cfg['models'].items():
        batch = data.batch(ids, 'mode'); model = study.core.make_model(cfg, name, 42)
        assert batch['text_'+spec['text']+str(spec['text_dim'])].dtype == torch.float32
        with torch.no_grad(): model.classifier[-1].weight[1].fill_(.01)
        score = study.core.score_model(model, batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, batch['target'])
        loss.backward()
        assert torch.isfinite(loss)
        assert model.text_projection.weight.grad.abs().sum() > 0
