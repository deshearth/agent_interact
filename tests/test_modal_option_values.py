import numpy as np
from scripts.modal_option_values import numeric_features


def test_equivalent_fractions_and_last_value():
    f=numeric_features(r'Compute 2, then \\frac{2}{4}. The answer is \\boxed{0.5}.',{'A':r'\frac{1}{2}','B':'2'})
    np.testing.assert_allclose(f[:2],np.log1p([2,1]),atol=1e-7)
    assert f[2]==f[3]==f[6]==1


def test_labels_swap_without_truth_or_solution():
    text='The value is -3.5, while 7/2 is positive.'
    a=numeric_features(text,{'A':'-7/2','B':'3.5'});b=numeric_features(text,{'B':'-7/2','A':'3.5'})
    np.testing.assert_array_equal(a[[0,1]],b[[1,0]])
    assert a[2]==-b[2] and a[3]==-b[3] and a[7]==b[7]


def test_literal_boundaries_and_zero_denominator():
    f=numeric_features('x12 and y2.5 are identifiers; 12/0 is undefined. Answer: 2.',{'A':'2','B':'12'})
    np.testing.assert_allclose(f[:2],np.log1p([1,0]),atol=1e-7)
    assert f[3]==1
