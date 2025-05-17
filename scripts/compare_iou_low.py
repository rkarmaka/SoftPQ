import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import metrics.core as metrics
from metrics.softpq import SoftPQ
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases
from skimage.measure import label

def get_pq_softpq_data(operation, data_case, iou_high, iou_low_range, num_iterations):
    # Load synthetic test case
    if data_case == 'single_circle':
        ground_truth = synthetic_cases.create_circle_mask((256, 256), 32)
        ground_truth_labels = label(ground_truth)
        predicted_mask = synthetic_cases.create_circle_mask((256, 256), 32)
    elif data_case == 'paired_circles':
        ground_truth = synthetic_cases.create_paired_circles((256, 256), (32, 16), shift_x=25)
        ground_truth_labels = label(ground_truth)
        predicted_mask = synthetic_cases.create_paired_circles((256, 256), (32, 16), shift_x=25)
    else:
        raise ValueError(f"Unknown data_case: {data_case}")

    # Evaluate baseline PQ
    pq_values = []
    predicted_mask_copy = predicted_mask.copy()
    for _ in range(num_iterations):
        pq = metrics.evaluate_segmentation(ground_truth, predicted_mask_copy)['panoptic_quality']
        pq_values.append(pq)
        predicted_mask_copy = utils.erode_dilate_mask(predicted_mask_copy, operation=operation, kernel_size=1)

    # Evaluate SoftPQ
    SoftPQ_curves = {}
    for iou_low in iou_low_range:
        predicted_mask_copy = predicted_mask.copy()
        SoftPQ_values = []
        for _ in range(num_iterations):
            predicted_mask_copy_labels = label(predicted_mask_copy)
            softpq = SoftPQ(iou_high=iou_high, iou_low=iou_low)
            softpq_score = softpq.evaluate(ground_truth_labels, predicted_mask_copy_labels)
            SoftPQ_values.append(softpq_score)
            predicted_mask_copy = utils.erode_dilate_mask(predicted_mask_copy, operation=operation, kernel_size=1)
        SoftPQ_curves[round(iou_low, 2)] = SoftPQ_values

    return pq_values, SoftPQ_curves


def plot_side_by_side_panels(data_case='paired_circles', output_dir='output/figures', output_filename='combined_spq_panels.png'):
    os.makedirs(output_dir, exist_ok=True)

    iou_low_range = np.arange(0.5, 0.0, -0.05)
    num_iterations = 24
    iou_high = 0.5

    operations = ['erode', 'dilate']
    titles = ['Erosion: PQ vs SoftPQ', 'Dilation: PQ vs SoftPQ']

    # Set up plot
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'figure.figsize': (14, 6),
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

    

    fig, axes = plt.subplots(1, 2, sharey=True)

    for i, operation in enumerate(operations):
        ax = axes[i]
        pq_vals, spq_curves = get_pq_softpq_data(
            operation=operation,
            data_case=data_case,
            iou_high=iou_high,
            iou_low_range=iou_low_range,
            num_iterations=num_iterations
        )

        ax.plot(pq_vals, label='PQ (IoU > 0.5)', color='black', linestyle='-', linewidth=2.5)
        for iou_low, values in spq_curves.items():
            label_str = f"SoftPQ (IoU 0.5–{iou_low:.2f})"
            ax.plot(values, label=label_str, linestyle='--', marker='.', alpha=0.8)

        ax.set_title(titles[i])
        ax.set_xlabel('Iteration')
        if i == 0:
            ax.set_ylabel('Score')

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save figure
    save_path = os.path.join(output_dir, output_filename)
    metadata = {
        'Title': 'PQ vs SoftPQ Side-by-Side Comparison',
        'Author': 'Ranit Karmakar',
        'Description': 'Left: Erosion. Right: Dilation. SoftPQ with varying IoU low thresholds',
        'Keywords': 'segmentation, PQ, SoftPQ, erosion, dilation, IoU'
    }
    fig.savefig(save_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Combined plot saved to: {save_path}")


if __name__ == '__main__':
    plot_side_by_side_panels(data_case='paired_circles', output_filename='combined_spq_panels_low_iou_paired_circles_2.png')