import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from image import save_images, read_image, read_images, save_image, to_8bit
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

    lab_image = color.rgb2lab(image.astype(np.float32) / 65535.0)
    channels = list(cv2.split(lab_image))

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
    image_denoised_rgb = color.lab2rgb(denoised_image)
    image_denoised_rgb_16bit = (image_denoised_rgb * 65535).astype(np.uint16)

    # Free memory
    del denoised_image, image_denoised_rgb #, channels
    torch.cuda.empty_cache()
    gc.collect()

    return image_denoised_rgb_16bit

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def unsharp_mask(images, model, strength, k = 1.5):
    blurred_images = dncnn_images(model, images)

    #sharpened_images = [sharpen_image(image) for image in images]
    #sharpened_images = high_boost(images, model)
    #save_images(images, './images/original', name = 'original')

    save_images(blurred_images, './images/blurred', name = 'blurred')

    masks = [image - blurred for image, blurred in zip(images, blurred_images)]
    sharpened_images = [image + k* mask for image, mask in zip(images, masks)]

    save_images(sharpened_images, './images/sharpened', name = 'sharpened')

    merged_images = [cv2.addWeighted(sharpened_image, 1 + strength, blurred_image, -strength, 0) for sharpened_image, blurred_image in zip(sharpened_images, blurred_images)]
    
    save_images(merged_images, './images/merged', name = 'merged')
    
    del sharpened_images, blurred_images
    gc.collect()
    return merged_images

def high_boost(images, model, k = 1.5):
    # onlu l channel
    
    sharpened_images = dncnn_images(model, images)
    masks = [image - sharpened for image, sharpened in zip(images, sharpened_images)]
    sharpened_images = [image + k* mask for image, mask in zip(images, masks)]
    return sharpened_images

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