import config
from config import *
import numpy as np
import cv2
import pyiqa
import torch
from image import to_8bit, to_16bit
from skimage.metrics import structural_similarity as ssim

iqa_metrics = {}

def init_metrics(metrics = config.metrics):
    global iqa_metrics
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for metric in metrics:
        iqa_metrics[metric] = pyiqa.create_metric(metric, device = device)

def calculate_metric(image, metric):
    if len(image.shape) < 3:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(np.transpose(img, (2, 0, 1)))
    else:
      img_tensor = torch.tensor(np.transpose(image, (2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0)
    score_fr = iqa_metrics[metric](img_tensor)
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

def calculate_brisque(image):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    iqa_metric = pyiqa.create_metric('brisque_matlab', device = device)

    if len(image.shape) < 3:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(np.transpose(img, (2, 0, 1)))
    else:
        img_tensor = torch.tensor(np.transpose(image, (2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0)
    score_fr = iqa_metric(img_tensor)
    return score_fr.item()

def get_best_image(images, metric = 'liqe'):
    if metric not in iqa_metrics.keys():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        iqa_metrics[metric] = pyiqa.create_metric(metric, device = device)
    
    if metric != 'liqe':
        best_score = float('inf')
        for image in images:
            score = calculate_brisque(image)
            if score < best_score:
                best_score = score
                best_image = image
    else:
        best_score = float('-inf')
        for image in images:
            score = calculate_brisque(image)
            if score > best_score:
                best_score = score
                best_image = image

    return best_image, best_score

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