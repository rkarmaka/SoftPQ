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
    k_range=range(10, 0, -1),
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
        ground_truth = (base_mask > 0).astype(np.uint8)
        ground_truth_labels = label(ground_truth, connectivity=1)
    else:
        raise ValueError(f"Unknown data_case: {data_case}")

    pq_values = []
    SoftPQ_values = []
    SoftPQ_log_values = []
    SoftPQ_linear_values = []

    for k in k_range:
        # Create oversegmented prediction
        predicted = relabel_segments_fixed_groups(base_mask, k)
        predicted_labels = label(predicted, connectivity=1)

        # Evaluate PQ and SoftPQ variants
        pq = metrics.evaluate_segmentation(ground_truth_labels, predicted_labels)['panoptic_quality']
        SoftPQ = metrics._proposed_sqrt(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)
        SoftPQ_log = metrics._proposed_log(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)
        SoftPQ_linear = metrics._proposed_linear(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)

        pq_values.append(pq)
        SoftPQ_values.append(SoftPQ)
        SoftPQ_log_values.append(SoftPQ_log)
        SoftPQ_linear_values.append(SoftPQ_linear)

    print("[DEBUG] SoftPQ Log values:", SoftPQ_log_values)

    # Plotting setup
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

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

    # LEFT PLOT: PQ and SoftPQ metrics vs oversegmentation factor
    ax_left.plot(k_range, pq_values, label='PQ (IoU > 0.5)', color='black', linestyle='-', linewidth=2.5)
    ax_left.plot(k_range, SoftPQ_values, label=f'SoftPQ Square Root Scaling (IoU {iou_low}–{iou_high})',
                 linestyle='--', marker='o', color='#d62728', alpha=0.9)
    ax_left.plot(k_range, SoftPQ_log_values, label=f'SoftPQ Log Scaling (IoU {iou_low}–{iou_high})',
                 linestyle='--', marker='o', color='#1f77b4', alpha=0.9)
    ax_left.plot(k_range, SoftPQ_linear_values, label=f'SoftPQ Linear Scaling (IoU {iou_low}–{iou_high})',
                 linestyle='--', marker='o', color='#ff7f0e', alpha=0.9)

    ax_left.set_title('Effect of Oversegmentation on PQ and SoftPQ')
    ax_left.set_xlabel('Number of Oversegments')
    ax_left.set_ylabel('Score')
    ax_left.legend(loc='upper right')

    # RIGHT PLOT: Scaling functions
    x_vals = np.linspace(1.1, 10, 100)  # start from 1.1 to avoid log(1) = 0
    x_plot = x_vals

    ax_right.plot(x_plot, 1 / np.log(x_plot), label='1 / log(n)', color='#1f77b4', linestyle='--')
    ax_right.plot(x_plot, 1 / np.sqrt(x_plot), label='1 / sqrt(n)', color='#d62728', linestyle='--')
    ax_right.plot(x_plot, 1 / x_plot, label='1 / n', color='#ff7f0e', linestyle='--')

    ax_right.set_title('Scaling Functions vs n (where y = 1/f(n))')
    ax_right.set_xlabel('n')
    ax_right.set_ylabel('Scaling Value')
    ax_right.set_xlim(0, 10)
    ax_right.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Save figure with metadata
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
        k_range=range(10, 0, -1),
        data_case='single_circle',
        iou_high=0.5,
        iou_low=0.05,
        output_filename='scaling_factor_effect_2.png'
    )
