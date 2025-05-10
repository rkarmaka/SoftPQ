import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import matplotlib.pyplot as plt
import pandas as pd
import metrics.core as metrics
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases


def evaluate_erosion_robustness(
    data_case='single_circle',
    iou_threshold=0.5,
    num_iterations=24,
    output_dir='output/figures',
    output_filename='erosion_undersegmentation_plot.png'
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load synthetic test case
    if data_case == 'single_circle':
        ground_truth = synthetic_cases.create_circle_mask((256, 256), 32)
        predicted_mask = synthetic_cases.create_circle_mask((256, 256), 32)
    elif data_case == 'paired_circles':
        ground_truth = synthetic_cases.create_paired_circles((256, 256), (32, 16), shift_x=25)
        predicted_mask = synthetic_cases.create_paired_circles((256, 256), (32, 16), shift_x=25)
    else:
        raise ValueError(f"Unknown data_case: {data_case}")

    scores = []
    for i in range(1, num_iterations + 1):
        score = metrics.evaluate_segmentation(
            ground_truth, predicted_mask, thresh=iou_threshold
        )
        pq_modified = metrics._proposed_sqrt(ground_truth, predicted_mask)
        mAP = metrics.average_precision(ground_truth, predicted_mask)[0].mean()

        scores.append({
            'iteration': i,
            'f1': score['f1'],
            'mAP': mAP,
            'panoptic_quality': score['panoptic_quality'],
            'pq_modified': pq_modified
        })

        predicted_mask = utils.erode_dilate_mask(
            predicted_mask, operation='erode', kernel_size=1
        )

    scores_df = pd.DataFrame(scores)

    # Define plot setup
    metric_order = ['F1 Score', 'mAP', 'Panoptic Quality', 'gPQ']
    metrics_to_plot = {
        'F1 Score': scores_df.f1,
        'mAP': scores_df.mAP,
        'Panoptic Quality': scores_df.panoptic_quality,
        'gPQ': scores_df.pq_modified
    }
    custom_colors = {
        'F1 Score': '#1f77b4',
        'mAP': '#ff7f0e',
        'Panoptic Quality': '#2ca02c',
        'gPQ': '#d62728'
    }

    plt.rcParams.update({
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

    axes[0].set_title('Metric Evolution with Progressive Erosion')
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

    # Save to output directory with metadata
    metadata = {
        'Title': 'Erosion Robustness Plot',
        'Author': 'Ranit Karmakar',
        'Description': 'Evaluation of segmentation robustness to progressive erosion — comparing Panoptic Quality vs Generalized PQ',
        'Keywords': 'segmentation, robustness, PQ, metrics, evaluation, undersegmentation'
    }
    save_path = os.path.join(output_dir, output_filename)
    fig.savefig(save_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Plot saved to: {save_path}")


if __name__ == '__main__':
    evaluate_erosion_robustness(
        data_case='single_circle',
        iou_threshold=0.5,
        num_iterations=24,
        output_dir='output/figures',
        output_filename='erosion_undersegmentation_single_circle.png'
    )

    evaluate_erosion_robustness(
        data_case='paired_circles',
        iou_threshold=0.5,
        num_iterations=24,
        output_dir='output/figures',
        output_filename='erosion_undersegmentation_paired_circle.png'
    )
