from config import *
import numpy as np
import cv2
from skimage.measure import compare_natural_image_quality as niqe
from skimage import img_as_float

def calculate_niqe(image):
    image = img_as_float(image)
    return niqe(image)

def calculate_contrast(image):
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.std()

def calculate_brightness(image):
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.mean(gray)

def calculate_sharpness(image):
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def normalize(value, min_value, max_value):
    # Normalize the value to [0, 10]
    return (value - min_value) / (max_value - min_value) *10

def composite_metric(image):
    type_factor = 1 if image.dtype == np.uint8 else 255
    niqe = normalize(calculate_niqe(image), 0, 100)
    contrast = normalize(calculate_contrast(image), 0, 255*type_factor/2)
    sharpness = normalize(calculate_sharpness(image), 0, 500*type_factor)
    
    return (niqe + sharpness + contrast) / 3

def image_analysis(image):
    type_factor = 1 if image.dtype == np.uint8 else 255
    print('NIQE:', normalize(calculate_niqe(image), 0, 100))
    print('Contrast:', normalize(calculate_contrast(image), 0, 255*type_factor/2))
    print('Sharpness:', normalize(calculate_sharpness(image), 0, 500*type_factor))

    print('\nComposite Metric:', composite_metric(image))