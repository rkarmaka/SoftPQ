<!-- Env name: SoftPQ-env -->
# SoftPQ: Robust Instance Segmentation Evaluation via Soft Matching and Tunable Thresholds
**SoftPQ** is a generalization of the Panoptic Quality (PQ) metric that introduces soft matching and tunable IoU thresholds to provide a more flexible and interpretable evaluation framework for instance segmentation.

<!-- This repository contains the reference implementation used in our NeurIPS 2025 submission, along with example usage and evaluation scripts. -->

## Overview
Conventional segmentation metrics (e.g., F1, mAP, IoU, PQ) rely on fixed overlap thresholds and binary matching logic, which can obscure differences in error types and hinder iterative model development.

SoftPQ addresses this by:
- Using a pair of IoU thresholds to define a graded match region.
- Applying a sublinear penalty to fragmented or ambiguous predictions.
- Retaining compatibility with standard PQ when thresholds are fixed at 0.5.

## Key Features
- Tunable IoU thresholds to control the softness of matching.
- Penalty functions (e.g., sqrt, log, linear) to modulate the impact of soft matches.
- Improved interpretability and robustness in cases of over-/under-segmentation.
- Compatible with standard segmentation pipelines.

## Installation
```python
git clone https://github.com/rkarmaka/SoftPQ
cd SoftPQ
pip install -r requirements.txt
```

## Usage
```python
from softpq import SoftPQ

metric = SoftPQ(iou_high=0.5, iou_low=0.05, method='sqrt')
score = metric.evaluate(ground_truth_mask, predicted_mask)
```



