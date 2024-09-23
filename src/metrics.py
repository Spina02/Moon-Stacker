from config import *
import numpy as np
import cv2
from image import to_8bit

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(image_0, image):
    ssim_value, _ = ssim(image_0, image, full=True)
    return ssim_value

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) != 2 else image
    return gray.std()

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) != 2 else image
    return np.mean(gray)

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) != 2 else image
    laplacian = cv2.Laplacian(to_8bit(gray), cv2.CV_64F)
    return laplacian.var()

def normalize(value, min_value, max_value):
    # Normalize the value to [0, 10]
    return (value - min_value) / (max_value - min_value) *10

def evaluate_improvement(image_0, image):
    ssim_value = calculate_ssim(image_0, image)
    contrast_0 = calculate_contrast(image_0)
    contrast = calculate_contrast(image)
    brightness_0 = calculate_brightness(image_0)
    brightness = calculate_brightness(image)
    sharpness_0 = calculate_sharpness(image_0)
    sharpness = calculate_sharpness(image)
    
    improvement = {
        'SSIM': ssim_value,
        'Contrast Improvement': contrast - contrast_0,
        'Brightness Improvement': brightness - brightness_0,
        'Sharpness Improvement': sharpness - sharpness_0
    }
    
    return improvement