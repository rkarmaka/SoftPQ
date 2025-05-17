
from skimage import io, img_as_ubyte
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label

import os
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2 as cv





#########################################################################################
# SoftPQ Metric
#########################################################################################
class SoftPQ:
    """
    This class implements the SoftPQ metric.
    It is a generalized version of the PQ metric that introduces soft-matching.
    
    Control parameters:
        iou_high: float, default=0.5 -> Uses to identify the hard matches
        iou_low: float, default=0.05 -> Uses to identify the soft matches
        method: str, default='sqrt' -> Uses to identify the penalty function
        prioritize_underseg: bool, default=False -> Prioritize undersegmentation

    The metric is computed as follows:
        - Compute the intersection over union (IoU) matrix between the ground truth and the predicted masks
        - Use the iou_high to identify the hard matches
        - If undersegmentation is prioritized, compute softmatches for the predicted masks
        - If oversegmentation is prioritized, compute softmatches for the ground truth masks
        - Use the iou_low to identify the soft matches
        - Define the penalty function
        - Based on soft matches, compute modified mean true score
        - Compute the true positives
        - Compute modified false positives and false negatives
        - Compute the F1 score
        - Compute the soft PQ score
    """

    def __init__(self, iou_high=0.5, iou_low=0.05, method='sqrt', prioritize_underseg=False):
        """
        Initialize the SoftPQ class

        Args:
            iou_high: float, default=0.5 -> Uses to identify the hard matches
            iou_low: float, default=0.05 -> Uses to identify the soft matches
            method: str, default='sqrt' -> Uses to identify the penalty function
            prioritize_underseg: bool, default=False -> Prioritize undersegmentation
        """
        self.iou_high = iou_high
        self.iou_low = iou_low
        self.method = method
        self.prioritize_underseg = prioritize_underseg

    def _compute_f1_score(self, true_pos, false_pos, false_neg):
        """
        Compute the F1 score

        Args:
            true_pos: int -> Number of true positives
            false_pos: int -> Number of false positives
            false_neg: int -> Number of false negatives

        Returns:
            f1_score: float -> F1 score
        """
        return 2 * true_pos / (2 * true_pos + false_pos + false_neg)

    def _compute_label_overlap(self, ground_truth, prediction):
        """
        Compute the label overlap between the ground truth and the predicted masks

        Args:
            ground_truth: ND-array, int -> Ground truth masks
            prediction: ND-array, int -> Predicted masks
        
        Returns:
            overlap_matrix: ND-array, int -> Label overlap matrix
        """
        ground_truth = ground_truth.ravel()
        prediction = prediction.ravel()
        overlap_matrix = np.zeros((1 + ground_truth.max(), 1 + prediction.max()), dtype=np.uint)
        for i in range(len(ground_truth)):
            overlap_matrix[ground_truth[i], prediction[i]] += 1
        return overlap_matrix

    def _compute_iou_matrix(self, ground_truth_labels, prediction_labels):
        """
        Compute the IoU matrix between the ground truth and the predicted masks

        Args:
            ground_truth_labels: ND-array, int -> Ground truth labels
            prediction_labels: ND-array, int -> Predicted labels
        
        Returns:
            iou_matrix: ND-array, int -> IoU matrix
        """
        overlap = self._compute_label_overlap(ground_truth_labels, prediction_labels)
        pred_pixel_counts = np.sum(overlap, axis=0, keepdims=True)
        gt_pixel_counts = np.sum(overlap, axis=1, keepdims=True)
        union = pred_pixel_counts + gt_pixel_counts - overlap
        iou_matrix = overlap / union
        with np.errstate(divide='ignore', invalid='ignore'):
            iou_matrix = np.true_divide(overlap, union)
            iou_matrix[~np.isfinite(iou_matrix)] = 0.0

        return iou_matrix

    def _penalty(self, count, mode='sqrt'):
        """
        Compute the penalty function

        Args:
            count: int -> Number of soft matches
            mode: str, default='sqrt' -> Uses to identify the penalty function

        Returns:
            penalty_map[mode]: float -> Penalty function
        """
        penalty_map = {
            'sqrt': np.sqrt(count + 1),
            'log': np.maximum(1.0, np.log(count + 1)),
            'linear': count + 1
        }
        if mode not in penalty_map:
            raise ValueError(f"Unknown penalty mode: {mode}")
        return penalty_map[mode]


    def _compute_mean_true_score(self, iou_matrix, iou_thresh_high, iou_thresh_low, prioritize_underseg=False):
        """
        Compute the mean true score

        Args:
            iou_matrix: ND-array, int -> IoU matrix
            iou_thresh_high: float -> IoU threshold for hard matches
            iou_thresh_low: float -> IoU threshold for soft matches
            prioritize_underseg: bool, default=False -> Prioritize undersegmentation

        Returns:
            soft_pq_score: float -> Mean true score
            num_soft_matches: int -> Number of soft matches
        """
        if prioritize_underseg:
            iou_matrix = iou_matrix.T  # transpose to prioritize predicted-to-GT matches

        soft_pq_score = 0.0
        num_soft_matches = 0
        iou_core = iou_matrix[1:, 1:]
        num_rows, _ = iou_core.shape

        for row_index in range(num_rows):
            iou_row = iou_core[row_index]
            soft_match_mask = (iou_row > iou_thresh_low) & (iou_row < iou_thresh_high)
            num_soft_for_row = np.count_nonzero(soft_match_mask)
            matched_mask = iou_row > iou_thresh_high

            if np.any(matched_mask):
                if num_soft_for_row > 0:
                    num_soft_matches += num_soft_for_row
                penalty = self._penalty(num_soft_for_row, self.method)
                soft_pq_score += iou_row[matched_mask][0] + np.sum(iou_row[soft_match_mask] / penalty)
            else:
                penalty = self._penalty(num_soft_for_row, self.method)
                soft_pq_score += np.sum(iou_row[soft_match_mask] / penalty)

        return soft_pq_score, num_soft_matches

    def _count_true_positives(self, iou_matrix, iou_threshold):
        """
        Count the true positives

        Args:
            iou_matrix: ND-array, int -> IoU matrix
            iou_threshold: float -> IoU threshold
        
        Returns:
            true_positives: int -> Number of true positives
        """
        iou_core = iou_matrix[1:, 1:]
        num_gt, num_pred = iou_core.shape
        max_matches = min(num_gt, num_pred)

        cost_matrix = -(iou_core >= iou_threshold).astype(float) - iou_core / (2 * max_matches)
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

        assert len(gt_indices) == len(pred_indices) == max_matches

        valid_matches = iou_core[gt_indices, pred_indices] >= iou_threshold
        return np.count_nonzero(valid_matches)

    def evaluate(self, ground_truth_mask, predicted_mask):
        """
        Evaluate the SoftPQ metric

        Args:
            ground_truth_mask: ND-array, int -> Ground truth mask
            predicted_mask: ND-array, int -> Predicted mask
        
        Returns:
            soft_pq_score: float -> SoftPQ score
        """
        gt_labels = label(ground_truth_mask)
        pred_labels = label(predicted_mask)

        iou_matrix = self._compute_iou_matrix(gt_labels, pred_labels)
        num_gt_objects, num_pred_objects = iou_matrix[1:, 1:].shape

        if num_gt_objects == 0 or num_pred_objects == 0:
            return 0.0

        mean_true_score, num_soft_matches = self._compute_mean_true_score(
            iou_matrix,
            self.iou_high,
            self.iou_low,
            self.prioritize_underseg
        )

        true_positives = self._count_true_positives(iou_matrix, self.iou_high)

        if self.prioritize_underseg:
            false_positives = max(num_pred_objects - true_positives, 0)
            false_negatives = max(num_gt_objects - true_positives - num_soft_matches, 0)
        else:
            false_positives = max(num_pred_objects - true_positives - num_soft_matches, 0)
            false_negatives = max(num_gt_objects - true_positives, 0)

        f1_score = self._compute_f1_score(true_positives, false_positives, false_negatives)

        normalization_factor = true_positives if true_positives > 0 else max(num_gt_objects, 1)
        mean_true_score /= normalization_factor

        if f1_score == 0:
            return mean_true_score
        else:
            return mean_true_score*f1_score






#########################################################################################
# Other metrics
#########################################################################################

def getIoUvsThreshold(prediction_filepath, groud_truth_filepath):
  '''
  This function calculates the IoU score for a given prediction and ground truth image at different thresholds.
  '''
  prediction = io.imread(prediction_filepath)
  ground_truth_image = img_as_ubyte(io.imread(groud_truth_filepath, as_gray=True), force_copy=True)

  threshold_list = []
  IoU_scores_list = []

  for threshold in range(0,256):
    # Convert to 8-bit for calculating the IoU
    mask = img_as_ubyte(prediction, force_copy=True)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0

    # Intersection over Union metric
    intersection = np.logical_and(ground_truth_image, np.squeeze(mask))
    union = np.logical_or(ground_truth_image, np.squeeze(mask))
    iou_score = np.sum(intersection) / np.sum(union)

    threshold_list.append(threshold)
    IoU_scores_list.append(iou_score)

  return (threshold_list, IoU_scores_list)



def evaluate_segmentation(y_true, y_pred, thresh=0.5):
    '''
    This function evaluates the segmentation performance of a given ground truth and predicted label images.
    '''
    y_true = _create_labeled_mask(y_true)
    y_pred = _create_labeled_mask(y_pred)
    
    overlap = _label_overlap(y_true, y_pred)

    score = matching(y_true, y_pred, thresh=thresh)

    return score



def _safe_divide(x,y, eps=1e-10):
    '''
    This function computes a safe divide which returns 0 if y is zero.
    '''
    if np.isscalar(x) and np.isscalar(y):
        return x/y if np.abs(y)>eps else 0.0
    else:
        out = np.zeros(np.broadcast(x,y).shape, np.float32)
        np.divide(x,y, out=out, where=np.abs(y)>eps)
        return out

def _label_overlap(x, y):
    '''
    This function computes the overlap between two label images.
    '''
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def intersection_over_union(overlap):
    '''
    This function computes the intersection over union between two label images.
    '''
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))



def matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images.

    Currently, the following metrics are implemented:

    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp), false positives (fp), and false negatives (fn)
    whether their intersection over union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives

    * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects

    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    thresh: float
        threshold for matching criterion (default 0.5)
    criterion: string
        matching criterion (default IoU)
    report_matches: bool
        if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')

    Returns
    -------
    Matching object with different metrics as attributes

    Examples
    --------
    >>> y_true = np.zeros((100,100), np.uint16)
    >>> y_true[10:20,10:20] = 1
    >>> y_pred = np.roll(y_true,5,axis = 0)

    >>> stats = matching(y_true, y_pred)
    >>> print(stats)
    Matching(criterion='iou', thresh=0.5, fp=1, tp=0, fn=1, precision=0, recall=0, accuracy=0, f1=0, n_true=1, n_pred=1, mean_true_score=0.0, mean_matched_score=0.0, panoptic_quality=0.0)

    """

    overlap = _label_overlap(y_true, y_pred)
    scores = intersection_over_union(overlap)

    # print(scores)
    
    # ignoring background
    scores = scores[1:,1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        # not_trivial = n_matched > 0 and np.any(scores >= thr)
        not_trivial = n_matched > 0
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2*n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind,pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        # assert tp+fp == n_pred
        # assert tp+fn == n_true

        # the score sum over all matched objects (tp)
        sum_matched_score = np.sum(scores[true_ind,pred_ind][match_ok]) if not_trivial else 0.0

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score    = _safe_divide(sum_matched_score, n_true)
        panoptic_quality   = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

        stats_dict = dict (
            criterion          = criterion,
            thresh             = thr,
            fp                 = fp,
            tp                 = tp,
            fn                 = fn,
            precision          = np.round(precision(tp,fp,fn),4),
            recall             = np.round(recall(tp,fp,fn),4),
            accuracy           = np.round(accuracy(tp,fp,fn),4),
            f1                 = np.round(f1(tp,fp,fn),4),
            n_true             = n_true,
            n_pred             = n_pred,
            mean_true_score    = np.round(mean_true_score,4),
            mean_matched_score = np.round(mean_matched_score,4),
            panoptic_quality   = np.round(panoptic_quality,4),
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update (
                    # int() to be json serializable
                    matched_pairs  = tuple((int(map_rev_true[i]),int(map_rev_pred[j])) for i,j in zip(1+true_ind,1+pred_ind)),
                    matched_scores = tuple(scores[true_ind,pred_ind]),
                    matched_tps    = tuple(map(int,np.flatnonzero(match_ok))),
                )
            else:
                stats_dict.update (
                    matched_pairs  = (),
                    matched_scores = (),
                    matched_tps    = (),
                )
        return stats_dict
        # return namedtuple('Matching',stats_dict.keys())(*stats_dict.values())

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single,thresh))


def precision(tp,fp,fn):
    return tp/(tp+fp) if tp > 0 else 0
def recall(tp,fp,fn):
    return tp/(tp+fn) if tp > 0 else 0
def accuracy(tp,fp,fn):
    return tp/(tp+fp+fn) if tp > 0 else 0
def f1(tp,fp,fn):
    # also known as "dice coefficient"
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0





def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ 
    Average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)): 
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)): 
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]): 
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]): 
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]): 
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]): 
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn



def _f1(tp,fp,fn):
    # also known as "dice coefficient"
    # print(tp)
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0

def compute_iou_matrix(gt_labels, pred_labels):
    """
    Compute the IoU matrix between ground truth and predicted labels.
    """
    n_gt = gt_labels.max()
    n_pred = pred_labels.max()
    iou_matrix = np.zeros((n_gt, n_pred))
    
    for i in range(1, n_gt + 1):
        gt_mask = (gt_labels == i)
        gt_area = gt_mask.sum()
        for j in range(1, n_pred + 1):
            pred_mask = (pred_labels == j)
            pred_area = pred_mask.sum()
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = gt_area + pred_area - intersection
            if union > 0:
                iou_matrix[i - 1, j - 1] = intersection / union
    return iou_matrix

def compute_standard_pq(gt_labels, pred_labels, iou_threshold=0.5):
    """
    Compute the standard Panoptic Quality (PQ) metric.
    """
    iou_matrix = compute_iou_matrix(gt_labels, pred_labels)
    n_gt, n_pred = iou_matrix.shape

    # Keep track of matched ground truth and predicted segments
    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)

    TP = 0
    sum_iou = 0.0

    # Match segments with IoU >= threshold
    matches = []
    for i in range(n_gt):
        for j in range(n_pred):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j, iou_matrix[i, j]))

    # Sort matches by IoU in descending order
    matches.sort(key=lambda x: x[2], reverse=True)

    for i, j, iou in matches:
        if not gt_matched[i] and not pred_matched[j]:
            gt_matched[i] = True
            pred_matched[j] = True
            TP += 1
            sum_iou += iou

    FP = np.sum(~pred_matched)
    FN = np.sum(~gt_matched)

    denominator = TP + 0.5 * FP + 0.5 * FN
    if denominator == 0:
        pq = 1.0
    else:
        pq = sum_iou / denominator

    return pq, TP, FP, FN

def compute_f1_score(TP, FP, FN):
    """
    Compute F1 score given TP, FP, FN.
    """
    denominator = 2 * TP + FP + FN
    if denominator == 0:
        f1 = 1.0
    else:
        f1 = 2 * TP / denominator
    return f1

# def _proposed_sqrt(true_label, pred_label, iou_high=0.5, iou_low=0.05):
#     iou = _intersection_over_union(true_label, pred_label)
#     modified_mts = 0
#     n_overseg = 0
#     n_underseg = 0
#     n_gt, n_seg = iou[1:,1:].shape
#     n_matched = min(n_gt, n_seg)

#     # print(np.round(iou,2))

#     for i in range(1,n_gt+1):
#         iou_temp = iou[i,1:]
#         overseg_count_temp = ((iou_temp>iou_low)&(iou_temp<iou_high)).sum()
#         # print(overseg_count_temp)
#         if iou_temp[iou_temp>iou_high].size>0:
#             if overseg_count_temp>0:
#                 # print(iou_temp)
#                 n_overseg+=overseg_count_temp
#             else:
#                 overseg_count_temp = 0
#         if iou_temp[iou_temp>iou_high].size>0:
#             modified_mts_temp = iou_temp[iou_temp>iou_high][0] + np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/np.sqrt(overseg_count_temp+1))
#         else:
#             modified_mts_temp = np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/np.sqrt(overseg_count_temp+1))

#         # print(modified_mts_temp)

#         modified_mts+=modified_mts_temp

#     # modified_mts = modified_mts/n_gt

#     # print(modified_mts)

#     scores = iou[1:,1:]
#     costs = -(scores >= iou_high).astype(float) - scores / (2*n_matched)
#     true_ind, pred_ind = linear_sum_assignment(costs)
#     assert n_matched == len(true_ind) == len(pred_ind)
#     match_ok = scores[true_ind,pred_ind] >= iou_high
#     tp = np.count_nonzero(match_ok)
    
#     if tp != 0:
#         modified_mts = modified_mts/tp
#     else:
#         modified_mts = modified_mts/n_gt

#     fp = max(n_seg-tp-n_overseg,0)

#     fn = max(n_gt-tp-n_underseg,0)
#     f1 = _f1(tp, fp, fn)

#     # print(fp, fn, tp, n_overseg)

#     if f1 == 0:
#         return modified_mts
#     else:
#         return modified_mts*f1
    


def _proposed_sqrt(true_label, pred_label, iou_high=0.5, iou_low=0.05):
    iou = _intersection_over_union(true_label, pred_label)
    modified_mts = 0
    n_overseg = 0
    n_underseg = 0
    n_gt, n_seg = iou[1:,1:].shape
    n_matched = min(n_gt, n_seg)

    # print(np.round(iou,2))

    for i in range(1,n_gt+1):
        iou_temp = iou[i,1:]
        overseg_count_temp = ((iou_temp>iou_low)&(iou_temp<iou_high)).sum()
        # print(overseg_count_temp)
        if iou_temp[iou_temp>iou_high].size>0:
            if overseg_count_temp>0:
                # print(iou_temp)
                n_overseg+=overseg_count_temp
            else:
                overseg_count_temp = 0
        if iou_temp[iou_temp>iou_high].size>0:
            modified_mts_temp = iou_temp[iou_temp>iou_high][0] + np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/np.sqrt(overseg_count_temp+1))
        else:
            modified_mts_temp = np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/np.sqrt(overseg_count_temp+1))

        # print(modified_mts_temp)

        modified_mts+=modified_mts_temp

    # modified_mts = modified_mts/n_gt

    # print(modified_mts)

    scores = iou[1:,1:]
    costs = -(scores >= iou_high).astype(float) - scores / (2*n_matched)
    true_ind, pred_ind = linear_sum_assignment(costs)
    assert n_matched == len(true_ind) == len(pred_ind)
    match_ok = scores[true_ind,pred_ind] >= iou_high
    tp = np.count_nonzero(match_ok)
    
    if tp != 0:
        modified_mts = modified_mts/tp
    else:
        modified_mts = modified_mts/n_gt

    fp = max(n_seg-tp-n_overseg,0)

    fn = max(n_gt-tp-n_underseg,0)
    f1 = _f1(tp, fp, fn)

    # print(fp, fn, tp, n_overseg)

    if f1 == 0:
        return modified_mts
    else:
        return modified_mts*f1
    


def _proposed_log(true_label, pred_label, iou_high=0.5, iou_low=0.05):
    iou = _intersection_over_union(true_label, pred_label)
    modified_mts = 0
    n_overseg = 0
    n_underseg = 0
    n_gt, n_seg = iou[1:,1:].shape
    n_matched = min(n_gt, n_seg)

    # print(np.round(iou,2))

    for i in range(1,n_gt+1):
        iou_temp = iou[i,1:]
        overseg_count_temp = ((iou_temp>iou_low)&(iou_temp<iou_high)).sum()
        # print(overseg_count_temp)
        if iou_temp[iou_temp>iou_high].size>0:
            if overseg_count_temp>0:
                # print(iou_temp)
                n_overseg+=overseg_count_temp
            else:
                overseg_count_temp = 0
        if iou_temp[iou_temp>iou_high].size>0:
            factor = max(1.0, np.log(overseg_count_temp+1))
            modified_mts_temp = iou_temp[iou_temp>iou_high][0] + np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/factor)
        else:
            factor = max(1.0, np.log(overseg_count_temp+1))
            modified_mts_temp = np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/factor)

        # print(modified_mts_temp)
        # print(overseg_count_temp)

        modified_mts+=modified_mts_temp

    # modified_mts = modified_mts/n_gt

    # print(modified_mts)

    scores = iou[1:,1:]
    costs = -(scores >= iou_high).astype(float) - scores / (2*n_matched)
    true_ind, pred_ind = linear_sum_assignment(costs)
    assert n_matched == len(true_ind) == len(pred_ind)
    match_ok = scores[true_ind,pred_ind] >= iou_high
    tp = np.count_nonzero(match_ok)
    
    if tp != 0:
        modified_mts = modified_mts/tp
    else:
        modified_mts = modified_mts/n_gt

    fp = max(n_seg-tp-n_overseg,0)

    fn = max(n_gt-tp-n_underseg,0)
    f1 = _f1(tp, fp, fn)

    # print(fp, fn, tp, n_overseg)

    if f1 == 0:
        return modified_mts
    else:
        return modified_mts*f1
    


def _proposed_linear(true_label, pred_label, iou_high=0.5, iou_low=0.05):
    iou = _intersection_over_union(true_label, pred_label)
    modified_mts = 0
    n_overseg = 0
    n_underseg = 0
    n_gt, n_seg = iou[1:,1:].shape
    n_matched = min(n_gt, n_seg)

    # print(np.round(iou,2))

    for i in range(1,n_gt+1):
        iou_temp = iou[i,1:]
        overseg_count_temp = ((iou_temp>iou_low)&(iou_temp<iou_high)).sum()
        # print(overseg_count_temp)
        if iou_temp[iou_temp>iou_high].size>0:
            if overseg_count_temp>0:
                # print(iou_temp)
                n_overseg+=overseg_count_temp
            else:
                overseg_count_temp = 0
        if iou_temp[iou_temp>iou_high].size>0:
            modified_mts_temp = iou_temp[iou_temp>iou_high][0] + np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/(overseg_count_temp+1))
        else:
            modified_mts_temp = np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/(overseg_count_temp+1))

        # print(modified_mts_temp)

        modified_mts+=modified_mts_temp

    # modified_mts = modified_mts/n_gt

    # print(modified_mts)

    scores = iou[1:,1:]
    costs = -(scores >= iou_high).astype(float) - scores / (2*n_matched)
    true_ind, pred_ind = linear_sum_assignment(costs)
    assert n_matched == len(true_ind) == len(pred_ind)
    match_ok = scores[true_ind,pred_ind] >= iou_high
    tp = np.count_nonzero(match_ok)
    
    if tp != 0:
        modified_mts = modified_mts/tp
    else:
        modified_mts = modified_mts/n_gt

    fp = max(n_seg-tp-n_overseg,0)

    fn = max(n_gt-tp-n_underseg,0)
    f1 = _f1(tp, fp, fn)

    # print(fp, fn, tp, n_overseg)

    if f1 == 0:
        return modified_mts
    else:
        return modified_mts*f1

########################################################
# From NeurIPS
########################################################

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    
    return iou


def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    
    true_ind, pred_ind = linear_sum_assignment(costs)


    
    match_ok = iou[true_ind, pred_ind] >= th
    # print(iou[true_ind, pred_ind])
    tp = match_ok.sum()

    n_min = min(iou.shape[0], iou.shape[1])
    costs = -((iou < th)&(iou > 0)).astype(float) - iou / (2*n_min)
    
    true_ind, pred_ind = linear_sum_assignment(costs)
    # print(iou[true_ind, pred_ind])
    
    return tp



def _create_labeled_mask(mask):
    '''
    This function creates a labeled mask from a binary mask.
    '''
    if mask.dtype == 'bool':
        mask=mask.astype('uint8')
    
    return label(mask)















    


# def _proposed_sqrt(true_label, pred_label, iou_high=0.5, iou_low=0.05):
#     iou = _intersection_over_union(true_label, pred_label)
#     modified_mts = 0
#     n_overseg = 0
#     n_underseg = 0
#     n_gt, n_seg = iou[1:,1:].shape
#     n_matched = min(n_gt, n_seg)

#     # print(np.round(iou,2))

#     for i in range(1,n_gt+1):
#         iou_temp = iou[i,1:]
#         overseg_count_temp = ((iou_temp>iou_low)&(iou_temp<iou_high)).sum()
#         # print(overseg_count_temp)
#         if iou_temp[iou_temp>iou_high].size>0:
#             if overseg_count_temp>0:
#                 # print(iou_temp)
#                 n_overseg+=overseg_count_temp
#             else:
#                 overseg_count_temp = 0
#         if iou_temp[iou_temp>iou_high].size>0:
#             modified_mts_temp = iou_temp[iou_temp>iou_high][0] + np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/np.sqrt(overseg_count_temp+1))
#         else:
#             modified_mts_temp = np.sum((iou_temp[((iou_temp>iou_low)&(iou_temp<=iou_high))])/np.sqrt(overseg_count_temp+1))

#         # print(modified_mts_temp)

#         modified_mts+=modified_mts_temp

#     # modified_mts = modified_mts/n_gt

#     # print(modified_mts)

#     scores = iou[1:,1:]
#     costs = -(scores >= iou_high).astype(float) - scores / (2*n_matched)
#     true_ind, pred_ind = linear_sum_assignment(costs)
#     assert n_matched == len(true_ind) == len(pred_ind)
#     match_ok = scores[true_ind,pred_ind] >= iou_high
#     tp = np.count_nonzero(match_ok)
    
#     if tp != 0:
#         modified_mts = modified_mts/tp
#     else:
#         modified_mts = modified_mts/n_gt

#     fp = max(n_seg-tp-n_overseg,0)

#     fn = max(n_gt-tp-n_underseg,0)
#     f1 = _f1(tp, fp, fn)

#     # print(fp, fn, tp, n_overseg)

#     if f1 == 0:
#         return modified_mts
#     else:
#         return modified_mts*f1