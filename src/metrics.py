from const import DEBUG
import numpy as np
import cv2
from skimage.measure import shannon_entropy

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

def edge_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    return np.mean(sobel)

def calculate_sharpness(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_variance(image):
    return np.var(image)

def calculate_entropy(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(gray)

# metric composite
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def composite_metric(image):
    snr = normalize(calculate_snr(image), 0, 10)
    contrast = normalize(calculate_contrast(image), 0, 255/2)
    sharpness = normalize(calculate_sharpness(image), 0, 100000)
    edge_sharp = normalize(edge_sharpness(image), 0, 1000)
    
    return (3 * snr + 2 * sharpness + 2 * edge_sharp + contrast) / 9

def image_analysis(image):
    print('SNR:', calculate_snr(image))
    print('Contrast:', calculate_contrast(image))
    print('Sharpness:', calculate_sharpness(image))
    print('Edge Sharpness:', edge_sharpness(image))
    print('Variance:', calculate_variance(image))
    print('Entropy:', calculate_entropy(image))

    print('\nComposite Metric:', composite_metric(image))