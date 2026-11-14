import numpy as np

from scripts.analyze_pooled_rule_limits import group_inputs,grouped_floors


def test_empirical_floors_match_a_known_collision_example():
    x=np.array([[1,2],[1,2],[1,3],[1,3]],dtype=float)
    inverse,counts=group_inputs(x)
    p=np.array([.1,.9,.3,.3]); weights=np.full(4,.25)
    r=grouped_floors(p,inverse,counts,weights)
    np.testing.assert_allclose(r['minimum_probability_mae'],.2)
    np.testing.assert_allclose(r['minimum_probability_rmse'],np.sqrt(.08))
    np.testing.assert_allclose(r['minimum_teacher_label_disagreement'],.25)
    np.testing.assert_allclose(r['minimum_teacher_kl'],.5*(.1*np.log(.1/.5)+.9*np.log(.9/.5)))
    assert r['distinct_pooled_groups']==2 and r['example_group_count']==2
    np.testing.assert_allclose(r['largest_group_probability_range'],.8)


def test_weighted_median_and_majority_use_probability_mass():
    inverse,counts=group_inputs(np.ones((3,2)))
    p=np.array([.1,.6,.9]); w=np.array([.05,.1,.85])
    r=grouped_floors(p,inverse,counts,w)
    np.testing.assert_allclose(r['minimum_probability_mae'],.07)
    np.testing.assert_allclose(r['minimum_teacher_label_disagreement'],.05)
    order=np.array([2,0,1])
    other=grouped_floors(p[order],inverse[order],counts,w[order])
    for k in ['minimum_probability_mae','minimum_probability_rmse','minimum_teacher_kl','minimum_teacher_label_disagreement']:
        np.testing.assert_allclose(r[k],other[k])


def test_unique_inputs_have_zero_empirical_floor():
    x=np.column_stack([np.ones(5),np.arange(5)])
    inverse,counts=group_inputs(x)
    r=grouped_floors(np.array([.01,.2,.5,.8,.99]),inverse,counts,np.full(5,.2))
    for key in ['minimum_probability_mae','minimum_probability_rmse','minimum_teacher_kl','minimum_teacher_label_disagreement']:
        np.testing.assert_allclose(r[key],0,atol=1e-14)


def test_exact_grouping_does_not_merge_nearby_features():
    x=np.array([[1,0],[1,-0.],[1,1e-12],[1,1e-10]])
    inverse,counts=group_inputs(x)
    assert inverse[0]==inverse[1]
    assert len(counts)==3
