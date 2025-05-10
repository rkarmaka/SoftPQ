def test_create_circle_case():
    from segmetric.data.synthetic_cases import create_circle_case
    mask = create_circle_case()
    assert mask.shape == (64, 64)
    assert mask.max() == 1