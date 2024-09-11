from config import *
import numpy as np
import cv2

def calculate_snr(image):
    signal = np.mean(image)
    noise = np.std(image)
    return signal / noise

def calculate_contrast(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def calculate_sharpness(image):
# Check if the image is already in grayscale
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var() 

def calculate_sharpness(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_variance(image):
    return np.var(image)

# metric composite
def normalize(value, min_value, max_value):
    # Normalize the value to [0, 10]
    return (value - min_value) / (max_value - min_value) *10

def composite_metric(image):
    type_factor = 1 if image.dtype == np.uint8 else 255
    snr = normalize(calculate_snr(image), 0, 10)
    contrast = normalize(calculate_contrast(image), 0, 255*type_factor/2)
    sharpness = normalize(calculate_sharpness(image), 0, 100*type_factor)
    
    return (2 * snr + 2 * sharpness + contrast) / 5

def image_analysis(image):
    type_factor = 1 if image.dtype == np.uint8 else 255
    print('SNR:', normalize(calculate_snr(image), 0, 10))
    print('Contrast:', normalize(calculate_contrast(image), 0, 255*type_factor/2))
    print('Sharpness:', normalize(calculate_sharpness(image), 0, 100*type_factor))

    print('\nComposite Metric:', composite_metric(image))