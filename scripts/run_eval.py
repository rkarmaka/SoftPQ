def main():
    from metrics.softpq import SoftPQ
    import data.synthetic_cases as synthetic_cases
    true_mask, pred_mask = synthetic_cases.create_paired_circles((256, 256), (32, 16), shift_x=25)
    softpq = SoftPQ(iou_high=0.5, iou_low=0.1)
    score = softpq.evaluate(true_mask, pred_mask)
    print("SoftPQ Score:", score)


if __name__ == '__main__':
    main()