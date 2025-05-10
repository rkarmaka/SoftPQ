import cv2 as cv
import numpy as np


def erode_dilate_mask(mask, operation='erode', kernel_size=1):
    kernel = create_circular_se(kernel_size)
    if operation == 'erode':
        return cv.erode(mask, kernel)
    elif operation == 'dilate':
        return cv.dilate(mask, kernel)
    else:
        raise ValueError("operation must be 'erode' or 'dilate'")
    

# create circular structuring element
def create_circular_se(radius):
    se = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    se[mask] = 1
    return se.astype('uint8')


# def _create_labeled_mask(mask):
#     '''
#     This function creates a labeled mask from a binary mask.
#     '''
#     if mask.dtype == 'bool':
#         mask=mask.astype('uint8')
    
#     return label(mask)