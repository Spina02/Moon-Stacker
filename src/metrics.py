from config import *
import numpy as np
import cv2
from image import to_8bit

from skimage.metrics import structural_similarity as ssim
from skimage.color import deltaE_ciede2000, rgb2lab

def ciede2000(image_0, image):
    lab1 = rgb2lab(cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB))
    lab2 = rgb2lab(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return deltaE_ciede2000(lab1, lab2).mean()

def calculate_tenengrad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = np.sqrt(sobel_x**2 + sobel_y**2).mean()
    return sharpness

def calculate_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
    histogram = histogram / histogram.sum()  # Normalizza
    ent = -np.sum(histogram * np.log2(histogram + 1e-7))  # Calcola l'entropia
    return ent

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

    color_diff = ciede2000(image_0, image)

    tenengrad_0 = calculate_tenengrad(image_0)
    tenengrad = calculate_tenengrad(image)

    entropy_0 = calculate_entropy(image_0)
    entropy = calculate_entropy(image)
    
    improvement = {
        'Contrast Improvement': contrast - contrast_0,
        'Brightness Improvement': brightness - brightness_0,
        'Sharpness Improvement': sharpness - sharpness_0,
        'Sharpness Improvement (Tenengrad)': tenengrad - tenengrad_0,
        'Entropy Improvement': entropy - entropy_0,
        'SSIM': ssim_value,
        'CIEDE2000 Color Difference': color_diff
    }
    
    return improvement