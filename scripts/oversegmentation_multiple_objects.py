import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import metrics.core as metrics
from metrics.softpq import SoftPQ
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases
from data.synthetic_cases import relabel_segments_fixed_groups


def evaluate_oversegmentation_effect(
    k_range=range(10, 0, -1),
    data_case='single_circle',
    iou_high=0.5,
    iou_low_values=np.arange(0.5, 0.0, -0.1),
    output_dir='output/figures',
    output_filename='oversegmentation_k_effect_full.png'
):
    os.makedirs(output_dir, exist_ok=True)

    # Load base synthetic data
    if data_case == 'single_circle':
        base_mask = synthetic_cases.create_circle_mask((256, 256), 32)
        second_mask = np.zeros_like(base_mask)
        # Create multiple circles
        second_mask = np.zeros_like(base_mask)
        centers = np.array([(50,50), (100,50), (150,50), (75,50), (125,50), (175,50), (200,50), (225,50), 
                            (50,200), (100,200), (150,200), (75,200), (125,200), (175,200), (200,200), (225,200)])
        i = 11
        for center in centers:
            circle_mask = i*synthetic_cases.create_circle_mask((256, 256), 10, center=center)
            second_mask = second_mask+circle_mask
            i+=1
    elif data_case == 'paired_circles':
        base_mask = synthetic_cases.create_paired_circles((256, 256), (32, 16), shift_x=25)
        second_mask = np.zeros_like(base_mask)
    else:
        raise ValueError(f"Unknown data_case: {data_case}")

    ground_truth = label(base_mask+second_mask)

    

    pq_values = []
    f1_values = []
    map_values = []
    SoftPQ_curves = {round(l, 2): [] for l in iou_low_values}

    for k in k_range:
        oversegmented = synthetic_cases.create_n_oversegments_from_circle(base_mask, n=k, offset=10)
        predicted = relabel_segments_fixed_groups(oversegmented, k)
        predicted = predicted+second_mask
        predicted_labels = label(predicted)

        # Standard metrics
        eval = metrics.evaluate_segmentation(ground_truth, predicted_labels, thresh=iou_high)
        pq = eval['panoptic_quality']
        f1 = eval['f1']
        mAP = metrics.average_precision(ground_truth, predicted_labels)[0].mean()

        pq_values.append(pq)
        f1_values.append(f1)
        map_values.append(mAP)

        # SoftPQ for different IoU lows
        for iou_low in iou_low_values:
            softpq = SoftPQ(iou_high=iou_high, iou_low=iou_low)
            softpq_score = softpq.evaluate(ground_truth, predicted_labels)
            SoftPQ_curves[round(iou_low, 2)].append(softpq_score)

    # Plot
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'legend.frameon': False
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Core metrics
    ax.plot(k_range, pq_values, label='PQ (IoU > 0.5)', color='black', linestyle='-', linewidth=2.5)
    ax.plot(k_range, f1_values, label='F1 Score', linestyle=':', marker='x', color='#1f77b4')
    ax.plot(k_range, map_values, label='mAP', linestyle='-.', marker='o', color='#ff7f0e')

    # SoftPQ curves
    for iou_low, values in SoftPQ_curves.items():
        label_str = f"SoftPQ (IoU 0.5–{iou_low:.2f})"
        ax.plot(k_range, values, label=label_str, linestyle='--', marker='.', alpha=0.8)

    ax.set_title('Effect of Oversegmentation (k) on PQ, SoftPQ, F1, and mAP')
    ax.set_xlabel('Number of Oversegments (k)')
    ax.set_ylabel('Score')

    # Shared legend below
    fig.legend(
        handles=ax.lines,
        labels=[line.get_label() for line in ax.lines],
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.1)
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    # Save with metadata
    metadata = {
        'Title': 'Oversegmentation Metric Robustness',
        'Author': 'Ranit Karmakar',
        'Description': 'Evaluates PQ, SoftPQ (varied IoU low), F1, and mAP under increasing oversegmentation (k)',
        'Keywords': 'segmentation, oversegmentation, PQ, SoftPQ, F1, mAP, robustness'
    }

    save_path = os.path.join(output_dir, output_filename)
    fig.savefig(save_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Plot saved to: {save_path}")


if __name__ == '__main__':
    evaluate_oversegmentation_effect(
        k_range=range(10, 0, -1),
        data_case='single_circle',
        iou_low_values=np.arange(0.5, 0.0, -0.1),
        output_filename='oversegmentation_k_effect_full.png'
    )