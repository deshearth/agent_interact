import numpy as np
from scripts.generalization_lexical import text_features

def test_option_letters_not_indefinite_articles_or_word_parts():
    v = text_features('A good argument is an AB test. I prefer option B to option A.')
    np.testing.assert_allclose(v[:7], [np.log(3), np.log(2), 1, 1, np.log(2), np.log(2), 1])
    v = text_features('a cat by a tree; no option identified')
    np.testing.assert_array_equal(v[:7], np.zeros(7))

def test_last_explicit_recommendation_and_empty_message():
    v = text_features('Option A first; my answer is (B).')
    assert v[2] == 1 and v[3] == -1 and v[6] == -1
    np.testing.assert_array_equal(text_features(''), np.zeros(8))
