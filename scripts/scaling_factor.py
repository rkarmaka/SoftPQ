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
    gpq_log_values = []
    gpq_linear_values = []

    for k in k_range:
        # Create oversegmented prediction
        predicted = relabel_segments_fixed_groups(base_mask, k)
        predicted_labels = label(predicted)

        # Evaluate PQ and gPQ
        pq = metrics.evaluate_segmentation(ground_truth_labels, predicted_labels)['panoptic_quality']
        gpq = metrics._proposed_sqrt(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)
        gpq_log = metrics._proposed_log(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)
        gpq_linear = metrics._proposed_linear(ground_truth_labels, predicted_labels, iou_high=iou_high, iou_low=iou_low)

        pq_values.append(pq)
        gpq_values.append(gpq)
        gpq_log_values.append(gpq_log)
        gpq_linear_values.append(gpq_linear)
    
    print(gpq_log_values)

    # Plot
    plt.rcParams.update({
        'font.family': 'Times New Roman', 
        'font.size': 16,
        'figure.figsize': (10, 6),
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'legend.frameon': False
    })

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

    # LEFT: PQ and gPQ
    ax_left.plot(k_range, pq_values, label='PQ (IoU > 0.5)', color='black', linestyle='-', linewidth=2.5)
    ax_left.plot(k_range, gpq_values, label=f'gPQ Square Root Scaling (IoU 0.5–{iou_low})', linestyle='--', marker='o', color='#d62728', alpha=0.9)
    ax_left.plot(k_range, gpq_log_values, label=f'gPQ Log Scaling (IoU 0.5–{iou_low})', linestyle='--', marker='o', color='#1f77b4', alpha=0.9)
    ax_left.plot(k_range, gpq_linear_values, label=f'gPQ Linear Scaling (IoU 0.5–{iou_low})', linestyle='--', marker='o', color='#ff7f0e', alpha=0.9)

    k = k_range[0]
    ax_left.set_title(f'Effect of Oversegmentation ({k}) on PQ and gPQ')
    ax_left.set_xlabel(f'Number of Oversegments ({k})')
    ax_left.set_ylabel('Score')
    ax_left.legend(loc='upper right')

    # RIGHT: Scaling Functions (plotted vs x but evaluated as x+1)
    x_vals = np.linspace(0, max(k_range), 100)
    x_plot = x_vals+1  # This is for axis (starts at 0)
    x_eval = x_vals + 1  # This is for function computation (starts at 1)

    ax_right.plot(x_plot, 1 / np.log(x_eval), label='1 / log(x + 1)', color='#1f77b4', linestyle='--')
    ax_right.plot(x_plot, 1 / np.sqrt(x_eval), label='1 / sqrt(x + 1)', color='#d62728', linestyle='--')
    ax_right.plot(x_plot, 1 / (x_eval), label='1 / (x + 1)', color='#ff7f0e', linestyle='--')

    ax_right.set_title('Scaling Functions vs x (where y = 1/f(x + 1))')
    ax_right.set_xlabel('x')
    ax_right.set_ylabel('Scaling Value')
    ax_right.set_xlim(0, max(k_range))
    ax_right.set_ylim(0, 1.5)
    ax_right.legend(loc='upper right')

    # # Shared Legend
    # fig.legend(
    #     handles=ax_left.lines + ax_right.lines,
    #     labels=[line.get_label() for line in ax_left.lines + ax_right.lines],
    #     loc='lower center',
    #     ncol=4,
    #     bbox_to_anchor=(0.5, 0)
    # )

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
        output_filename='scaling_factor_effect.png'
    )
