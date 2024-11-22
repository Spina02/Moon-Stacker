import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import numpy as np
from image import *
import cv2
from utils import progress
from skimage import color

DNCNN_MODEL_PATH = './src/model.pth'

class DnCNN(nn.Module):
    def __init__(self,depth=17,n_channels=64,image_channels=1):
        super().__init__()
        self.dncnn = nn.Sequential(
            nn.Conv2d(image_channels, n_channels, 3, padding=1), nn.ReLU(inplace=True),
            *[layer_block(n_channels) for _ in range(depth-2)],
            nn.Conv2d(n_channels, image_channels, 3, padding=1)
        )
        self._initialize_weights()

    def forward(self, x):
        return x - self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1); init.constant_(m.bias, 0)

def layer_block(n_channels):
    return nn.Sequential(
        nn.Conv2d(n_channels,n_channels,3,padding=1,bias=False),
        nn.BatchNorm2d(n_channels, eps=1e-4, momentum=0.95),
        nn.ReLU(inplace=True)
    )

def model_init(model_path=DNCNN_MODEL_PATH):
    model = torch.load(model_path)
    model.eval()
    return model

def preprocess_image(image, max_value):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize the image between 0 and 1
    image = image / max_value
    #transform = transforms.Toc()
    transform = transforms.ToTensor()
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
    denoised_l = postprocess_image(denoised_l, 100)
    channels[0] = denoised_l

    # Merge the channels
    if len(image.shape) == 3:
        denoised_image = cv2.merge(channels)
        denoised_image = color.lab2rgb(denoised_image)
    else:
        denoised_image = channels[0]

    # Free memory
    torch.cuda.empty_cache()

    return denoised_image