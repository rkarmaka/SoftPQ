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


def run_progressive_segmentation(
    operation='dilate',
    data_case='single_circle',
    iou_threshold=0.5,
    num_iterations=24
):
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
        score = metrics.evaluate_segmentation(ground_truth, predicted_mask, thresh=iou_threshold)
        softpq = SoftPQ()
        pq_modified = softpq.evaluate(ground_truth, predicted_mask)
        mAP = metrics.average_precision(ground_truth, predicted_mask)[0].mean()

        scores.append({
            'iteration': i,
            'f1': score['f1'],
            'mAP': mAP,
            'panoptic_quality': score['panoptic_quality'],
            'pq_modified': pq_modified
        })

        predicted_mask = utils.erode_dilate_mask(predicted_mask, operation=operation, kernel_size=1)

    return pd.DataFrame(scores)


def compare_erosion_dilation_plot():
    output_dir = 'output/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Run both tests
    df_dilate = run_progressive_segmentation(operation='dilate')
    df_erode = run_progressive_segmentation(operation='erode')

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Metric selection and color
    metrics_to_plot = ['panoptic_quality', 'pq_modified']
    display_names = {'panoptic_quality': 'PQ', 'pq_modified': 'SoftPQ'}
    colors = {'PQ': '#2ca02c', 'SoftPQ': '#d62728'}

    # Dilation plot
    for m in metrics_to_plot:
        name = display_names[m]
        axes[0].plot(df_dilate[m], label=f'{name} (Dilation)', color=colors[name], marker='.')

    axes[0].set_title('Progressive Dilation (Undersegmentation)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Score')

    # Erosion plot
    for m in metrics_to_plot:
        name = display_names[m]
        axes[1].plot(df_erode[m], label=f'{name} (Erosion)', color=colors[name], linestyle='--', marker='.')

    axes[1].set_title('Progressive Erosion (Oversegmentation)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Score')

    # Shared legend
    all_labels = [
        f'{display_names[m]} (Dilation)' for m in metrics_to_plot
    ] + [
        f'{display_names[m]} (Erosion)' for m in metrics_to_plot
    ]
    fig.legend(all_labels, loc='lower center', ncol=2 * len(metrics_to_plot), bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    # Metadata + Save
    metadata = {
        'Title': 'Dilation vs Erosion - Undersegmentation Comparison',
        'Author': 'Ranit Karmakar',
        'Description': 'Comparison of PQ and SoftPQ under progressive dilation and erosion to evaluate robustness under under- and oversegmentation',
        'Keywords': 'segmentation, evaluation, panoptic quality, dilation, erosion, robustness'
    }
    save_path = os.path.join(output_dir, 'comparison_undersegmentation.png')
    fig.savefig(save_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Comparison plot saved to: {save_path}")


if __name__ == '__main__':
    compare_erosion_dilation_plot()
