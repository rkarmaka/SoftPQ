def main():
    from metrics.softpq import SoftPQ
    from data import synthetic_cases
    true_mask = synthetic_cases.create_circle_mask((256, 256), 32)
    pred_mask = synthetic_cases.create_circle_mask((256, 256), 32, center=(125, 125))
    softpq = SoftPQ(iou_high=0.5, iou_low=0.1)
    score = softpq.evaluate(true_mask, pred_mask)
    print("SoftPQ Score:", score)


if __name__ == '__main__':
    main()