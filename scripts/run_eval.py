def main():
    from segmetric.metrics.core import _proposed_sqrt
    from segmetric.data.synthetic_cases import create_shifted_pair
    true_mask, pred_mask = create_shifted_pair(shift=(8, 5))
    score = _proposed_sqrt(true_mask, pred_mask)
    print("Proposed Score:", score)


if __name__ == '__main__':
    main()