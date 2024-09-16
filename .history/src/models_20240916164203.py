import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from image import *
import cv2
import gc
from utils import progress
from skimage import color

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
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

def model_init(model_path = './models/DnCNN-PyTorch/logs/DnCNN-S-25/net.pth'):
    model = DnCNN(channels=1, num_of_layers=17)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    del state_dict, new_state_dict
    gc.collect()

    return model

def preprocess_image(image, max_value):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.float32)
    else:
        image = image.astype(np.float32)

    # Normalize the image between 0 and 1
    image = image / max_value
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)

    del transform
    gc.collect()
    return image


def postprocess_image(image, max_value):
    image = image.squeeze(0).detach().cpu()
    image = image * max_value
    image = image.numpy().squeeze()
    return image.astype(np.float32)

def denoise_tensor(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)

def perform_denoising(model, image, device):

    if len(image.shape) == 3:
        lab_image = color.rgb2lab(image.astype(np.float32) / 65535.0)
        channels = list(cv2.split(lab_image))
    else: 
        channels = [image]

    model = model.to(device)
    model.eval()

    #if len(image.shape) == 3:
    #    channels = list(cv2.split(image))
    #else:
    #    channels = [image]
    #
    #for i, channel in enumerate(channels):
    for i, channel in enumerate(channels):
        if i > 0: break
        # Preprocess the image
        channel = preprocess_image(channel, 100).to(device)

        with torch.no_grad():
            denoised_channel = model(channel).cpu()

        # Postprocess the image
        #channels[i] = postprocess_image(denoised_channel)
        channels[i] = postprocess_image(denoised_channel, 100)

        # Free memory
        del denoised_channel
        torch.cuda.empty_cache()
        gc.collect()

    # Merge the channels
    #denoised_image = cv2.merge(channels)
    denoised_image = cv2.merge(channels)
    if len(image.shape) == 3:
        denoised_image = color.lab2rgb(denoised_image)
    denoised_image = (denoised_image * 65535).astype(np.uint16)

    # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    return denoised_image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def sobel_images(images):
    sobel_images = []
    for image in images:
        # Convert to grayscale if the image is RGB
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Apply Sobel filter on the grayscale image
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

        # Combine Sobel X and Y
        sobel = np.hypot(sobel_x, sobel_y)

        # Normalize the result to 0-255 and append to the list
        sobel_images.append(normalize(sobel))
    return sobel_images

"""def unsharp_mask(images, model, strength, k = 1.5):
    
    blurred_images = dncnn_images(model, images)

    save_images(blurred_images, './images/blurred', name = 'blurred')

    merged_images = [cv2.addWeighted(image, 1 + strength, blurred_image, -strength, 0) for image, blurred_image in zip(images, blurred_images)]
    
    save_images(merged_images, './images/merged', name = 'merged')
    
    del blurred_images
    gc.collect()
    return merged_images"""

def unsharp_mask(imgs, model, strength):
    blurred_images = dncnn_images(model, images)
    save_images(blurred_images, './images/blurred', name='blurred')

    sobel_masks = sobel_images(images)
    normalized_sobel_masks = [cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint16) for mask in sobel_masks]

    merged_images = []
    for image, blurred_image, sobel_mask in zip(images, blurred_images, normalized_sobel_masks):
        # Ensure sobel_mask has the same number of channels as the image
        if len(image.shape) == 3 and len(sobel_mask.shape) == 2:
            sobel_mask = np.stack([sobel_mask] * 3, axis=-1)
        elif len(image.shape) == 2 and len(sobel_mask.shape) == 3:
            sobel_mask = color.rgb2gray(sobel_mask)
        # Adjust the strength based on the Sobel mask
        sharp_component = cv2.multiply(image, 1 + strength * sobel_mask)
        blurred_component = cv2.multiply(blurred_image, -strength * sobel_mask)
        merged_image = cv2.add(sharp_component, blurred_component)
        merged_images.append(merged_image)

    save_images(merged_images, './images/merged', name='merged')

    del blurred_images
    gc.collect()
    return merged_images

def dncnn_images(model, images):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    denoised_images = []
    for image in images:
        denoised_images.append(perform_denoising(model, image, device))

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

        progress(len(denoised_images), len(images), 'images denoised')

    return denoised_images