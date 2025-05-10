# import cv2 as cv
# import matplotlib.pyplot as plt
# from skimage.measure import label
# import numpy as np
# import pandas as pd
# from scipy.ndimage import shift
# from DIOS.evaluation import evaluate_segmentation, average_precision, compute_standard_pq, _proposed_sqrt, compute_f1_score

import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(""))


import matplotlib.pyplot as plt
import pandas as pd
import metrics.core as metrics
import metrics.utils as utils
import data.synthetic_cases as synthetic_cases

# create circular structuring element
def create_circular_se(radius):
    se = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    se[mask] = 1
    return se.astype('uint8')


# # Load images (Underseg  - Case 3) - Erosion
# path = "../_DATA"
# gt_img = cv.imread(f'{path}/underseg/gt_underseg.png', cv.IMREAD_GRAYSCALE)
# gt_img = (gt_img > 100).astype('uint8')
# gt_labels = label(gt_img)

# pl_img = cv.imread(f'{path}/underseg/gt_underseg.png', cv.IMREAD_GRAYSCALE)
# pl_img = (pl_img > 100).astype('uint8')
# pl_labels = label(pl_img)



scores = evaluate_segmentation(gt_labels, pl_labels)
pq_score_NIPS = scores['panoptic_quality']

data_dilate = []

entry = {'Standard_PQ': pq_score_NIPS,
        'Modified_PQ_50': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.5),
        'Modified_PQ_45': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.45),
        'Modified_PQ_40': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.40),
        'Modified_PQ_35': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.35),
        'Modified_PQ_30': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.30),
        'Modified_PQ_25': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.25),
        'Modified_PQ_20': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.20),
        'Modified_PQ_15': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.15),
        'Modified_PQ_10': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.10),
        'Modified_PQ_05': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.05),
        }
    
data_dilate.append(entry)
# print(f"Modified PQ*: {pq_star_score:.4f}")

for i in range(1, 21):
    # gt_img = cv.erode(gt_img, np.ones((3, 3), np.uint8), iterations=1)
    se = create_circular_se(1)
    pl_img = cv.dilate(pl_img, se, iterations=1)
    pl_labels = label(pl_img)
    
    # Compute Modified PQ* with tau1 = tau2 = 0.5 and w = 1.0
    # pq_star_score = _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.05)

    scores = evaluate_segmentation(gt_labels, pl_labels)
    pq_score_NIPS = scores['panoptic_quality']

    
    # print(f"F1 Score: {f1_score:.4f}")
    # print(f"Standard PQ - NIPS: {pq_score_NIPS:.4f}")
    # print(f"Modified PQ*: {pq_star_score:.4f}")
    # print(f"mAP: {mAP:.4f}")

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(gt_labels, cmap='gray')
    # plt.title(i)
    # plt.subplot(122)
    # plt.imshow(pl_labels, cmap='gray')
    # plt.title(i)

    # store in a dataframe
    entry = {'Standard_PQ': pq_score_NIPS,
             'Modified_PQ_50': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.5),
             'Modified_PQ_45': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.45),
             'Modified_PQ_40': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.40),
             'Modified_PQ_35': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.35),
             'Modified_PQ_30': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.30),
             'Modified_PQ_25': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.25),
             'Modified_PQ_20': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.20),
             'Modified_PQ_15': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.15),
             'Modified_PQ_10': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.10),
             'Modified_PQ_05': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.05),
             }
    
    data_dilate.append(entry)
    # print(f"Modified PQ*: {pq_star_score:.4f}")


pl_img = cv.imread(f'{path}/underseg/gt_underseg.png', cv.IMREAD_GRAYSCALE)
pl_img = (pl_img > 100).astype('uint8')

data_erode = []

for i in range(1, 21):
    # gt_img = cv.erode(gt_img, np.ones((3, 3), np.uint8), iterations=1)
    se = create_circular_se(1)
    pl_img = cv.erode(pl_img, se, iterations=1)
    pl_labels = label(pl_img)
    
    # Compute Modified PQ* with tau1 = tau2 = 0.5 and w = 1.0
    # pq_star_score = _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.05)

    scores = evaluate_segmentation(gt_labels, pl_labels)
    pq_score_NIPS = scores['panoptic_quality']

    
    # print(f"F1 Score: {f1_score:.4f}")
    # print(f"Standard PQ - NIPS: {pq_score_NIPS:.4f}")
    # print(f"Modified PQ*: {pq_star_score:.4f}")
    # print(f"mAP: {mAP:.4f}")

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(gt_labels, cmap='gray')
    # plt.title(i)
    # plt.subplot(122)
    # plt.imshow(pl_labels, cmap='gray')
    # plt.title(i)

    # store in a dataframe
    entry = {'Standard_PQ': pq_score_NIPS,
             'Modified_PQ_50': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.5),
             'Modified_PQ_45': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.45),
             'Modified_PQ_40': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.40),
             'Modified_PQ_35': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.35),
             'Modified_PQ_30': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.30),
             'Modified_PQ_25': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.25),
             'Modified_PQ_20': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.20),
             'Modified_PQ_15': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.15),
             'Modified_PQ_10': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.10),
             'Modified_PQ_05': _proposed_sqrt(gt_labels, pl_labels, iou_high=0.5, iou_low=0.05),
             }
    
    data_erode.append(entry)



# reverse order of the list
data_dilate.reverse()

data = data_dilate + data_erode

df = pd.DataFrame(data)