# median stacking
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import *
from metrics import *


def median_stack(images):
    # Verifica il numero di canali della prima immagine
    num_channels = 1 if len(images[0].shape) == 2 else images[0].shape[2]
    img_type = images[0].dtype

    if num_channels == 1:
        # Immagini in scala di grigi
        stacked_image = np.median(images, axis=0).astype(img_type)
    else:
        # Immagini a colori
        stacked_channels = [np.median([cv2.split(img)[i] for img in images], axis=0).astype(img_type) for i in range(num_channels)]
        stacked_image = cv2.merge(stacked_channels)
    
    return stacked_image

def sigma_clipping(images, sigma=3):
    num_channels = 1 if len(images[0].shape) == 2 else images[0].shape[2]
    img_type = images[0].dtype
    stacked_channels = []
    for i in range(num_channels):
        channel_stack = np.array([cv2.split(img)[i] for img in images])
        mean = np.mean(channel_stack, axis=0)
        std = np.std(channel_stack, axis=0)
        mask = np.abs(channel_stack - mean) < sigma * std
        clipped_images = np.where(mask, channel_stack, mean)
        stacked_channel = np.mean(clipped_images, axis=0).astype(img_type)
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
    elif method == 'edge_sharpness':
        weights = [edge_sharpness(image) for image in images]
    elif method == 'composite':
        weights = [composite_metric(image) for image in images]
    else:
        raise ValueError("Unknown method for calculating weights")
    return weights

def weighted_average_stack(images, method='snr'):
    img_type = images[0].dtype
    # Calculate the weights for each image
    weights = calculate_weights(images, method)

    # Set to zero the lowest 10% of the weights
    weights = np.array(weights)
    weights[np.argsort(weights)[:len(weights)//10]] = 0
    weights = weights / np.sum(weights)

    num_channels = 1 if len(images[0].shape) == 2 else images[0].shape[2]
    stacked_channels = []
    for i in range(num_channels):
        # Extract the i-th channel from each image
        channel_stack = np.array([cv2.split(img)[i] for img in images])

        # Calculate the weighted sum of the channels
        weighted_sum = np.zeros_like(channel_stack[0], dtype=np.float64)
        for channel, weight in zip(channel_stack, weights):
            weighted_sum += channel * weight

        # Normalize the weighted sum to 8-bit range
        stacked_channel = (weighted_sum / np.sum(weights)).astype(img_type)
        stacked_channels.append(stacked_channel)

    # Combine the channels into a single image
    stacked_image = cv2.merge(stacked_channels)
    return stacked_image