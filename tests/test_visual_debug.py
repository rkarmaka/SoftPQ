def test_visual_shifted_case():
    import matplotlib.pyplot as plt
    from segmetric.data.synthetic_cases import create_shifted_pair
    true_mask, pred_mask = create_shifted_pair(shift=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(true_mask)
    plt.title("True")
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask)
    plt.title("Pred")
    plt.show()