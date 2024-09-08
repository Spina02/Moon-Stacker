# median stacking
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from const import *
from metrics import *

def median_stack(images):
    channels = cv2.split(images[0])
    stacked_channels = [np.median([cv2.split(img)[i] for img in images], axis=0).astype(np.uint8) for i in range(3)]
    stacked_image = cv2.merge(stacked_channels)
    return stacked_image

def sigma_clipping(images, sigma=3):
    channels = cv2.split(images[0])
    stacked_channels = []
    for i in range(3):
        channel_stack = np.array([cv2.split(img)[i] for img in images])
        mean = np.mean(channel_stack, axis=0)
        std = np.std(channel_stack, axis=0)
        mask = np.abs(channel_stack - mean) < sigma * std
        clipped_images = np.where(mask, channel_stack, mean)
        stacked_channel = np.mean(clipped_images, axis=0).astype(np.uint8)
        stacked_channels.append(stacked_channel)
    stacked_image = cv2.merge(stacked_channels)
    return stacked_image

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
    channels = cv2.split(images[0])
    stacked_channels = []
    for i in range(3):
        channel_stack = np.array([cv2.split(img)[i] for img in images])
        weights = calculate_weights(channel_stack, method)
        weighted_sum = np.zeros_like(channel_stack[0], dtype=np.float64)
        total_weight = np.sum(weights)
        for channel, weight in zip(channel_stack, weights):
            weighted_sum += channel * weight
        stacked_channel = (weighted_sum / total_weight).astype(np.uint8)
        stacked_channels.append(stacked_channel)
    stacked_image = cv2.merge(stacked_channels)
    return stacked_image