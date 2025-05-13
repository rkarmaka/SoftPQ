import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import metrics.core as metrics
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases
from skimage.measure import label

def compare_pq_vs_gpq_varying_iou_low(
    operation='dilate',
    data_case='single_circle',
    iou_high=0.5,
    iou_low_range=np.arange(0.5, 0.0, -0.05),
    num_iterations=24,
    output_dir='output/figures',
    output_filename='pq_vs_gpq_varying_ioulow.png'
):
    os.makedirs(output_dir, exist_ok=True)

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

    # Evaluate gPQ for each iou_low
    gpq_curves = {}
    for iou_low in iou_low_range:
        predicted_mask_copy = predicted_mask.copy()
        gpq_values = []
        for _ in range(num_iterations):
            predicted_mask_copy_labels = label(predicted_mask_copy)
            gpq = metrics._proposed_sqrt(ground_truth_labels, predicted_mask_copy_labels, iou_high=iou_high, iou_low=iou_low)
            gpq_values.append(gpq)
            predicted_mask_copy = utils.erode_dilate_mask(predicted_mask_copy, operation=operation, kernel_size=1)
        gpq_curves[round(iou_low, 2)] = gpq_values

    # Plot Setup
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
    ax.plot(pq_values, label='PQ (IoU > 0.5)', color='black', linestyle='-', linewidth=2.5)

    for iou_low, values in gpq_curves.items():
        l = f"gPQ (IoU 0.5–{iou_low:.2f})"
        ax.plot(values, label=l, linestyle='--', marker='.', alpha=0.8)

    ax.set_title('PQ vs gPQ with Varying IoU Low Thresholds')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    
    # Move legend below plot
    ax.legend(
        handles=ax.lines,
        labels=[line.get_label() for line in ax.lines],
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, -0.35)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)


    # Save with metadata
    metadata = {
        'Title': 'PQ vs gPQ Robustness',
        'Author': 'Ranit Karmakar',
        'Description': f'Comparison of Panoptic Quality vs Generalized PQ using varying IoU low thresholds ({operation})',
        'Keywords': 'segmentation, PQ, gPQ, iou_low, robustness'
    }
    save_path = os.path.join(output_dir, output_filename)
    fig.savefig(save_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Plot saved to: {save_path}")


if __name__ == '__main__':
    compare_pq_vs_gpq_varying_iou_low(
        operation='erode',  # or 'erode'
        data_case='paired_circles',
        output_filename='spq_erosion_low_iou.png'
    )

    compare_pq_vs_gpq_varying_iou_low(
        operation='dilate',
        data_case='paired_circles',
        output_filename='spq_dilation_low_iou.png'
    )
