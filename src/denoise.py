import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from image import *
import cv2
from utils import progress
from skimage import color
from config import DNCNN_MODEL_PATH

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, kernel_size=3, features=64):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            return x - self.dncnn(x)

def model_init(model_path = DNCNN_MODEL_PATH):
    model = DnCNN(channels=1, num_of_layers=17, kernel_size=3, features=64)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    return model

def preprocess_image(image, max_value):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize the image between 0 and 1
    image = image / max_value
    transform = transforms.Toc()
    image = transform(image).unsqueeze(0)
    return image

def postprocess_image(image, max_value):
    image = image.squeeze(0).detach().cpu()
    image = image * max_value
    image = image.numpy().squeeze()
    return image.astype(np.float32)

def denoise_tensor(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)

def perform_denoising(model, image):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if len(image.shape) == 3:
        lab_image = color.rgb2lab(image)
        channels = list(cv2.split(lab_image))
    else: 
        channels = [image]

    model = model.to(device)
    model.eval()

    # Preprocess the image
    l = preprocess_image(channels[0], 100).to(device)

    with torch.no_grad():
        denoised_l = model(l).cpu()

    # Postprocess the l channel
    channels[0] = postprocess_image(denoised_l, 100)

    # Merge the channels
    if len(image.shape) == 3:
        denoised_image = cv2.merge(channels)
        denoised_image = color.lab2rgb(denoised_image)
    else:
        denoised_image = channels[0]

    # Free memory
    torch.cuda.empty_cache()

    return denoised_image

def dncnn_images(model, images):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.cuda.empty_cache()

    denoised_images = []
    for image in images:

        denoised_images.append(perform_denoising(model, image, device))

        # Free memory
        torch.cuda.empty_cache()

        progress(len(denoised_images), len(images), 'images denoised')

    return denoised_images