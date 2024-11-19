import config
from config import *
import numpy as np
import cv2
import pyiqa
import torch
from image import to_8bit, to_16bit
from skimage.metrics import structural_similarity as ssim


def calculate_metric(image, metric):
    if len(image.shape) < 3:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(np.transpose(img, (2, 0, 1)))
    else:
      img_tensor = torch.tensor(np.transpose(image, (2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0)

    if metric not in config.iqa_metrics.keys():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config.iqa_metrics[metric] = pyiqa.create_metric(metric, device = device)

    score_fr = config.iqa_metrics[metric](img_tensor)
    score_fr = score_fr.item() if torch.is_tensor(score_fr) else score_fr
    return score_fr

def calculate_metrics(image, name, metrics):
    scores = {}
    print(f'Calculating metrics for {name}')
    for metric in metrics:
        metric_score = calculate_metric(image, metric)
        scores[metric] = metric_score
        print(f'{metric} score: {metric_score:.4f}')
    return scores

def get_best_image(images, metric = 'liqe'):
    if metric not in config.iqa_metrics.keys():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        config.iqa_metrics[metric] = pyiqa.create_metric(metric, device = device)
    
    if metric != 'liqe':
        best_score = float('inf')
        for image in images:
            score = calculate_metric(image, metric)
            if score < best_score:
                best_score = score
                best_image = image
    else:
        best_score = float('-inf')
        for image in images:
            score = calculate_metric(image, metric)
            if score > best_score:
                best_score = score
                best_image = image

    return best_image, best_score

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

def calculate_sharpness_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) != 2 else image
    # Calcolo del gradiente utilizzando l'operatore Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente lungo l'asse x
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente lungo l'asse y
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Calcolo della magnitudine del gradiente
    mean_magnitude = np.mean(magnitude)  # Media della magnitudine del gradiente
    return mean_magnitude


def normalize(value, min_value, max_value):
    normalized = (value - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, min_value, max_value)
    return normalized