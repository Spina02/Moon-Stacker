from config import *
import numpy as np
import cv2
from image import to_8bit, to_16bit

from skimage.metrics import structural_similarity as ssim

from brisque import BRISQUE

def calculate_brisque(image):
    brisque = BRISQUE()
    if len(image.shape) < 3:
        return brisque.score(cv2.cvtColor(to_8bit(image), cv2.COLOR_GRAY2RGB))
    return brisque.score(to_8bit(image))

def get_min_brisque(images):
    return min(images, key=calculate_brisque)

def calculate_ssim(image_ref, image):
    ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY) if len(image_ref.shape) == 3 else image_ref
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ssim_value, _ = ssim(to_16bit(ref), to_16bit(img), full=True, data_range=65535)
    return ssim_value

def combined_score(brisque, ssim, alpha = 0.7, beta = 0.3):
    norm_brisque = 1 - normalize(brisque, 0, 100)
    ssim = np.clip(ssim, 0, 1)
    
    # Calculate combined score
    score = alpha * norm_brisque + beta * ssim
    return score

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
    normalized = (value - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, min_value, max_value)
    return normalized

def evaluate_improvement(image_0, image):

    ssim_value = calculate_ssim(image_0, image)

    contrast_0 = calculate_contrast(image_0)
    contrast = calculate_contrast(image)

    brightness_0 = calculate_brightness(image_0)
    brightness = calculate_brightness(image)

    sharpness_0 = calculate_sharpness(image_0)
    sharpness = calculate_sharpness(image)

    improvement = {
        'Contrast Improvement': contrast - contrast_0,
        'Brightness Improvement': brightness - brightness_0,
        'Sharpness Improvement': sharpness - sharpness_0,
        'SSIM': ssim_value,
    }
    
    return improvement