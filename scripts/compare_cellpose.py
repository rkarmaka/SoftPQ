import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import matplotlib.pyplot as plt
import pandas as pd
import metrics.core as metrics
import metrics.softpq as softpq
import metrics.utils as utils

def plot_pq_vs_pqstar(df, output_path='pq_vs_softpq_scatter.png'):

    x = df['pq']
    y = df['softpq']
    diff = y - x

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'figure.figsize': (12, 5),
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'legend.frameon': False,
        'lines.markersize': 6
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

    # Left: PQ vs SoftPQ
    ax1.scatter(x, y, color='#1f77b4', alpha=0.8, label='SoftPQ vs PQ')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)  # y=x line
    ax1.set_title('PQ vs SoftPQ')
    ax1.set_xlabel('PQ')
    ax1.set_ylabel('SoftPQ')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Right: PQ vs (SoftPQ - PQ)
    ax2.scatter(x, diff, color='#ff7f0e', alpha=0.8, label='SoftPQ - PQ')
    ax2.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax2.set_title('Difference: SoftPQ - PQ')
    ax2.set_xlabel('PQ')
    ax2.set_ylabel('Difference')
    ax2.set_xlim(0, 1)

    # Shared legend below
    fig.legend(
        handles=[line for ax in [ax1, ax2] for line in ax.collections],
        labels=[col.get_label() for ax in [ax1, ax2] for col in ax.collections],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    metadata = {
        'Title': 'PQ vs SoftPQ Scatter',
        'Author': 'Ranit Karmakar',
        'Description': 'Scatter plot comparing PQ and SoftPQ, and the difference (SoftPQ - PQ)',
        'Keywords': 'segmentation, PQ, SoftPQ, comparison, scatter'
    }

    fig.savefig(output_path, dpi=300, metadata=metadata)
    plt.close(fig)

    print(f"[✓] Scatter plots saved to: {output_path}")


if __name__ == "__main__":
    # Load the CSV file
    csv_path = "data/cellpose.csv"
    df = pd.read_csv(csv_path)

    plot_pq_vs_pqstar(df, output_path="output/figures/pq_vs_softpq_scatter.png")


