import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))


import matplotlib.pyplot as plt
import pandas as pd
import metrics.core as metrics
from metrics.softpq import SoftPQ
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases
from skimage.measure import label


def evaluate_dilation_robustness(
    data_case='single_circle',
    iou_threshold=0.5,
    num_iterations=24,
    output_dir='output/figures',
    output_filename='dilation_undersegmentation_plot.png'
):
    # Create output directory if it doesn't exist
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

    scores = []
    for i in range(1, num_iterations + 1):
        predicted_mask_labels = label(predicted_mask)
        score = metrics.evaluate_segmentation(
            ground_truth_labels, predicted_mask_labels, thresh=iou_threshold
        )
        softpq = SoftPQ()
        softpq_score = softpq.evaluate(ground_truth_labels, predicted_mask_labels)
        mAP = metrics.average_precision(ground_truth_labels, predicted_mask_labels)[0].mean()

        scores.append({
            'iteration': i,
            'f1': score['f1'],
            'mAP': mAP,
            'panoptic_quality': score['panoptic_quality'],
            'softpq': softpq_score
        })

        predicted_mask = utils.erode_dilate_mask(
            predicted_mask, operation='dilate', kernel_size=1
        )

    scores_df = pd.DataFrame(scores)

    # Define plot setup
    metric_order = ['F1 Score', 'mAP', 'Panoptic Quality', 'SoftPQ']
    metrics_to_plot = {
        'F1 Score': scores_df.f1,
        'mAP': scores_df.mAP,
        'Panoptic Quality': scores_df.panoptic_quality,
        'SoftPQ': scores_df.softpq
    }
    custom_colors = {
        'F1 Score': '#1f77b4',
        'mAP': '#ff7f0e',
        'Panoptic Quality': '#2ca02c',
        'SoftPQ': '#d62728'
    }

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'figure.figsize': (12, 5),
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'legend.frameon': False
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Left plot: metric values
    for name in metric_order:
        axes[0].plot(metrics_to_plot[name], label=name, color=custom_colors[name], marker='.')

    axes[0].set_title('Metric Evolution with Progressive Dilation')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Score')

    # Right plot: delta values
    for name in metric_order:
        axes[1].plot(metrics_to_plot[name].diff(), label=name, color=custom_colors[name], marker='.')

    axes[1].set_title('Change in Metric Values')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Δ Score')

    # Shared legend below
    fig.legend(metric_order, loc='lower center', ncol=len(metric_order), bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    # Save to output directory
    metadata = {
        'Title': 'Dilation Robustness Plot',
        'Author': 'Ranit Karmakar',
        'Description': 'Evaluation of segmentation robustness to progressive dilation — comparing Panoptic Quality vs Generalized PQ',
        'Keywords': 'segmentation, robustness, PQ, metrics, evaluation, undersegmentation'
    }
    save_path = os.path.join(output_dir, output_filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[✓] Plot saved to: {save_path}")


if __name__ == '__main__':
    # Example run
    evaluate_dilation_robustness(
        data_case='single_circle',
        iou_threshold=0.5,
        num_iterations=24,
        output_dir='output/figures',
        output_filename='dilation_undersegmentation_single_circle.png'
    )

    evaluate_dilation_robustness(
        data_case='paired_circles',
        iou_threshold=0.5,
        num_iterations=24,
        output_dir='output/figures',
        output_filename='dilation_undersegmentation_paired_circle.png'
    )
