def test_proposed_sqrt_metric():
    from segmetric.metrics.core import _proposed_sqrt
    from segmetric.data.synthetic_cases import create_shifted_pair
    true_label, pred_label = create_shifted_pair()
    score = _proposed_sqrt(true_label, pred_label)
    assert isinstance(score, float) or isinstance(score, int)