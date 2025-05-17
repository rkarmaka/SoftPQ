import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.measure import label
import metrics.core as metrics
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases
from data.synthetic_cases import relabel_segments_fixed_groups, create_multi_circle_image, simulate_incremental_oversegmentation


def evaluate_oversegmentation_effect(
    k_range,
    iou_high=0.5,
    iou_low=0.1,
    output_dir='output/figures',
    output_filename='oversegmentation_k_effect.png'
):
    os.makedirs(output_dir, exist_ok=True)

    # Load base synthetic data
    base_mask = create_multi_circle_image(grid_size=5, image_size=256, radius=10)
    ground_truth = (base_mask>0).astype(np.uint8)
    ground_truth_labels = label(ground_truth)


    pq_values = []
    SoftPQ_values = []
    f1_values = []
    mAP_values = []

    for k in k_range:
        # Create oversegmented prediction
        predicted = simulate_incremental_oversegmentation(base_mask, k)
        predicted_labels = label(predicted)

        # Evaluate PQ and SoftPQ
        pq = metrics.evaluate_segmentation(ground_truth_labels, predicted_labels)['panoptic_quality']
        SoftPQ = metrics._proposed_sqrt(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)
        f1 = metrics.evaluate_segmentation(ground_truth_labels, predicted_labels)['f1']
        mAP = metrics.average_precision(ground_truth_labels, predicted_labels)[0].mean()

        pq_values.append(pq)
        SoftPQ_values.append(SoftPQ)
        f1_values.append(f1)
        mAP_values.append(mAP)
    
    # Convert x-axis to % oversegmentation (k / total_objects)
    total_objects = 25
    x_percent = np.array(k_range) / total_objects * 100

    # Convert scores to percentages
    pq_array = np.array(pq_values)
    SoftPQ_array = np.array(SoftPQ_values)
    f1_array = np.array(f1_values)
    mAP_array = np.array(mAP_values)

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

    # Fill between PQ and SoftPQ (now in percentage)
    ax.fill_between(x_percent, np.minimum(pq_array, SoftPQ_array), np.maximum(pq_array, SoftPQ_array),
                    color='lightgray', alpha=0.4, label='Gap between PQ and SoftPQ')

    # Replot lines
    ax.plot(x_percent, pq_array, label='PQ (IoU > 0.5)', color='black', linestyle='-', linewidth=2.5)
    ax.plot(x_percent, SoftPQ_array, label=f'SoftPQ (IoU 0.5–{iou_low})', linestyle='--', marker='o', color='#d62728', alpha=0.9)
    ax.plot(x_percent, mAP_array, label='mAP', linestyle='--', marker='o', color='#1f77b4', alpha=0.9)
    ax.plot(x_percent, f1_array, label='F1', linestyle='--', marker='o', color='#ff7f0e', alpha=0.9)

    ax.set_title('Effect of Oversegmentation on Segmentation Quality')
    ax.set_xlabel('Oversegmentation (% of objects)')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)

    # Legend (include shaded area)
    from matplotlib.patches import Patch
    gap_patch = Patch(facecolor='lightgray', edgecolor='gray', alpha=0.4, label='Gap between PQ and SoftPQ')

    fig.legend(
        handles=[*ax.lines, gap_patch],
        labels=[line.get_label() for line in ax.lines] + [gap_patch.get_label()],
        loc='lower center',
        ncol=6,
        bbox_to_anchor=(0.5, 0.1)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)


    # Save with metadata
    metadata = {
        'Title': 'Oversegmentation k Robustness',
        'Author': 'Ranit Karmakar',
        'Description': f'Evaluates the effect of varying number of segments (k) on PQ and SoftPQ (IoU high = {iou_high}, low = {iou_low})',
        'Keywords': 'segmentation, oversegmentation, PQ, SoftPQ, robustness, k-split'
    }

    save_path = os.path.join(output_dir, output_filename)
    fig.savefig(save_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Plot saved to: {save_path}")


if __name__ == '__main__':
    evaluate_oversegmentation_effect(
        k_range=range(1,26),
        iou_high=0.5,
        iou_low=0.05,
        output_filename='oversegmentation_effect.png'
    )
