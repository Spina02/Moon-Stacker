# median stacking
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from const import *
from metrics import *

def median_stack(images):
    # Stack images
    stack = np.dstack(images)
    # Calculate median
    median = np.median(stack, axis=2).astype(np.uint8)

    return median

def sigma_clipping(images, sigma=3):
    if DEBUG: print('Applying sigma clipping...')
    stack = np.dstack(images)
    median = np.median(stack, axis=2)
    std = np.std(stack, axis=2)

    # Apply sigma clipping
    mask = np.abs(stack - median[:, :, np.newaxis]) < sigma * std[:, :, np.newaxis]
    
    # Replace outliers with the median value instead of zero
    clipped_stack = np.where(mask, stack, median[:, :, np.newaxis])
    
    # Recalculate the median after sigma clipping
    median_clipped = np.median(clipped_stack, axis=2).astype(np.uint8)

    return median_clipped

def calculate_weights(images, method='snr'):
    if method == 'snr':
        weights = [calculate_snr(image) for image in images]
    elif method == 'contrast':
        weights = [calculate_contrast(image) for image in images]
    elif method == 'sharpness':
        weights = [calculate_sharpness(image) for image in images]
    elif method == 'variance':
        weights = [calculate_variance(image) for image in images]
    elif method == 'entropy':
        weights = [calculate_entropy(image) for image in images]
    else:
        raise ValueError("Unknown method for calculating weights")
    return weights

def weighted_average_stack(images, method='snr'):
    weights = calculate_weights(images, method)
    weighted_sum = np.zeros_like(images[0], dtype=np.float64)
    total_weight = np.sum(weights)

    for image, weight in zip(images, weights):
        weighted_sum += image * weight

    weighted_average = (weighted_sum / total_weight).astype(np.uint8)
    return weighted_average