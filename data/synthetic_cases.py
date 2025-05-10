import numpy as np
import cv2 as cv

def create_circle_mask(image_size: tuple, radius: int, center: tuple = None) -> np.ndarray:
    """
    Create a binary mask with a white filled circle.

    Args:
        image_size (tuple): (height, width) of the output image.
        radius (int): Radius of the circle.
        center (tuple, optional): (x, y) center of the circle. 
                                  If None, defaults to center of the image.

    Returns:
        np.ndarray: Binary image (uint8) with circle as 255 (white), background as 0 (black).
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    if center is None:
        center = (width // 2, height // 2)

    cv.circle(mask, center, radius, 1, thickness=-1)
    return mask

def create_paired_circles(image_size: tuple, radius: tuple, center: tuple = None, shift_x: int = None) -> np.ndarray:
    """
    Create a pair of circles with different radii.

    Args:
        image_size (tuple): (height, width) of the output image.
        radius (tuple): (radius1, radius2) of the circles.
        center (tuple, optional): (x, y) center of the circles. 
                                  If None, defaults to center of the image.
        shift_x (int, optional): Shift of the circles along x-axis.
                                  If None, defaults to (radius1+radius2)//2.
    
    Returns:
        np.ndarray: Binary image (uint8) with two circles as 255 (white), background as 0 (black).
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if center is None:
        center = (width // 2, height // 2)

    # Circles centers shall be shifted by shift_x
    if shift_x is None:
        shift_x = (radius[0]+radius[1])//2
    
    # Create two circles
    cv.circle(mask, (center[0] - shift_x, center[1]), radius[0], 1, thickness=-1)
    cv.circle(mask, (center[0] + shift_x, center[1]), radius[1], 1, thickness=-1)
    return mask



def create_n_oversegments_from_circle(gt_mask: np.ndarray, n: int = 3, offset: int = 3) -> np.ndarray:
    """
    Create an oversegmented prediction mask with `n` overlapping segments that cover a single GT circle.

    Args:
        gt_mask (np.ndarray): Ground truth binary mask (single circle).
        n (int): Number of oversegmented components to generate.
        offset (int): Pixel offset between oversegments.

    Returns:
        np.ndarray: Labeled prediction mask (uint8) with `n` oversegments (values: 1 to n).
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    height, width = gt_mask.shape
    pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)

    # Estimate circle center and radius from GT
    contours, _ = cv.findContours(gt_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in ground truth mask.")
    (x, y), radius = cv.minEnclosingCircle(contours[0])
    center = (int(x), int(y))
    radius = int(radius * 0.95)  # slightly reduce radius for variability

    # Generate n overlapping circles with varying x-offsets
    for i in range(n):
        dx = int((i - n // 2) * offset)
        segment_center = (center[0] + dx, center[1])
        cv.circle(pred_mask, segment_center, radius, i + 1, -1)  # label i+1

    return pred_mask

def relabel_segments_fixed_groups(img: np.ndarray, k: int) -> np.ndarray:
    """
    Relabel the input mask into `k` groups by evenly distributing the original labels.

    Args:
        img (np.ndarray): Input mask with original labels (e.g., 1–6).
        k (int): Desired number of output segments (e.g., 2, 3, ..., 6).

    Returns:
        np.ndarray: New predicted mask with `k` segment labels (1 to k), covering all original pixels.
    """
    pred = np.zeros_like(img, dtype=np.uint8)
    labels = np.unique(img)
    labels = labels[labels > 0]  # skip background

    # Split labels into k nearly equal groups
    groups = np.array_split(labels, k)

    for new_label, group in enumerate(groups, start=1):
        for original_label in group:
            pred[img == original_label] = new_label

    return pred