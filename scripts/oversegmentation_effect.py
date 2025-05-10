import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import metrics.core as metrics
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases
from data.synthetic_cases import relabel_segments_fixed_groups


def evaluate_oversegmentation_effect(
    k_range=range(10,0,-1),
    data_case='single_circle',
    iou_high=0.5,
    iou_low=0.1,
    output_dir='output/figures',
    output_filename='oversegmentation_k_effect.png'
):
    os.makedirs(output_dir, exist_ok=True)

    # Load base synthetic data
    if data_case == 'single_circle':
        circle_mask = synthetic_cases.create_circle_mask((256, 256), 32)
        base_mask = synthetic_cases.create_n_oversegments_from_circle(circle_mask, n=10, offset=10)
        ground_truth = (base_mask>0).astype(np.uint8)
        ground_truth_labels = label(ground_truth)
    else:
        raise ValueError(f"Unknown data_case: {data_case}")


    pq_values = []
    gpq_values = []

    for k in k_range:
        # Create oversegmented prediction
        predicted = relabel_segments_fixed_groups(base_mask, k)
        predicted_labels = label(predicted)

        # Evaluate PQ and gPQ
        pq = metrics.evaluate_segmentation(ground_truth_labels, predicted_labels)['panoptic_quality']
        gpq = metrics._proposed_sqrt(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)

        pq_values.append(pq)
        gpq_values.append(gpq)

    # Plot
    plt.rcParams.update({
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

    ax.plot(k_range, pq_values, label='PQ (IoU > 0.5)', color='black', linestyle='-', linewidth=2.5)
    ax.plot(k_range, gpq_values, label=f'gPQ (IoU 0.5–{iou_low})', linestyle='--', marker='o', color='#d62728', alpha=0.9)

    k = k_range[0]
    ax.set_title(f'Effect of Oversegmentation ({k}) on PQ and gPQ')
    ax.set_xlabel(f'Number of Oversegments ({k})')
    ax.set_ylabel('Score')

    # Shared legend below
    fig.legend(
        handles=ax.lines,
        labels=[line.get_label() for line in ax.lines],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.1)
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Save with metadata
    metadata = {
        'Title': 'Oversegmentation k Robustness',
        'Author': 'Ranit Karmakar',
        'Description': f'Evaluates the effect of varying number of segments (k) on PQ and gPQ (IoU high = {iou_high}, low = {iou_low})',
        'Keywords': 'segmentation, oversegmentation, PQ, gPQ, robustness, k-split'
    }

    save_path = os.path.join(output_dir, output_filename)
    fig.savefig(save_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Plot saved to: {save_path}")


if __name__ == '__main__':
    evaluate_oversegmentation_effect(
        k_range=range(10,0,-1),
        data_case='single_circle',
        iou_high=0.5,
        iou_low=0.05,
        output_filename='oversegmentation_k_effect.png'
    )
