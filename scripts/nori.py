import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))

import metrics.core as metrics
import metrics.utils as utils
import cv2 as cv
from skimage.measure import label

auto = cv.imread('/Users/ranit/Research/github/gPQ/data/C1_Day2_MAP1_auto.png', cv.IMREAD_GRAYSCALE)
manual = cv.imread('/Users/ranit/Research/github/gPQ/data/C1_Day2_MAP1_manual.png', cv.IMREAD_GRAYSCALE)

auto = label(auto)
manual = label(manual)

print(metrics._proposed_sqrt(manual, auto))

print(metrics.SoftPQ(iou_high=0.5, iou_low=0.05, method='sqrt', prioritize_underseg=False).evaluate(manual, auto))

print(metrics.SoftPQ(iou_high=0.5, iou_low=0.05, method='sqrt', prioritize_underseg=True).evaluate(manual, auto))