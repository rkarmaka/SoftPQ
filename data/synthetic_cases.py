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


def create_multi_circle_image(grid_size=3, image_size=256, radius=20):
    """
    Create a synthetic image with square number of circles in a grid.
    Returns:
        - image: synthetic grayscale image with circles
        - mask: labeled ground truth mask (each circle has a unique label)
    """
    mask = np.zeros((image_size, image_size), dtype=np.int32)

    step = image_size // (grid_size + 1)
    label = 1

    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            center = (j * step, i * step)
            gt = create_circle_mask((image_size, image_size), radius=radius, center=center)
            pred = create_n_oversegments_from_circle(gt, n=2, offset=5)
            pred[pred==1] = label
            pred[pred==2] = label+1
            mask = mask + pred
            label += 2

    return mask

def simulate_incremental_oversegmentation(mask, num_oversegment):
    """
    Simulates prediction by revealing one fragment at a time.

    Args:
        mask: np.ndarray (int), fragmented ground truth
        num_oversegment: int, how many individual segments to reveal

    Returns:
        prediction_mask: np.ndarray with:
            - 1: the currently exposed fragment(s)
            - 2: everything else merged
    """
    prediction_mask = np.zeros_like(mask, dtype=np.uint8)

    all_labels = np.unique(mask)
    all_labels = all_labels[all_labels != 0]  # exclude background

    reveal_labels = sorted(all_labels)[::2][:num_oversegment]
    other_labels = set(all_labels) - set(reveal_labels)

    for label in reveal_labels:
        prediction_mask[mask == label] = 1

    for label in other_labels:
        prediction_mask[mask == label] = 2

    return prediction_mask
