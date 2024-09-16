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

def preprocess_image(image):
    image_float = image.astype(np.float32) / 65535.0
lab_image = color.rgb2lab(image_float)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.float32)
    else:
        image = image.astype(np.float32)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (255,))
    ])
    image = transform(image).unsqueeze(0)

    del transform
    gc.collect()

    return image

def postprocess_image(image):
    image = image.squeeze(0).detach().cpu()
    image = image * 255
    image = image.numpy().squeeze()
    return image.astype(np.uint16)

def denoise_tensor(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)

def perform_denoising(model, image, device):
    l, a, b = cv2.split(cv2.cvtColor(to_8bit(image), cv2.COLOR_RGB2LAB))
    model = model.to(device)
    model.eval()

    #if len(image.shape) == 3:
    #    channels = list(cv2.split(image))
    #else:
    #    channels = [image]
    #
    #for i, channel in enumerate(channels):
    for channel in [l]:
        # Preprocess the image
        channel = preprocess_image(channel).to(device)

        with torch.no_grad():
            denoised_channel = model(channel).cpu()

        # Postprocess the image
        #channels[i] = postprocess_image(denoised_channel)
        l = postprocess_image(denoised_channel)

        # Free memory
        del channel, denoised_channel
        torch.cuda.empty_cache()
        gc.collect()

    # Merge the channels
    #denoised_image = cv2.merge(channels)
    denoised_image = cv2.merge([l, a, b])
    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_LAB2RGB)

    # Free memory
    del channels
    torch.cuda.empty_cache()
    gc.collect()

    return denoised_image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def unsharp_mask(images, model, strength):
    blurred_images = dncnn_images(model, images)
    sharpened_images = [sharpen_image(image) for image in images]
    #sharpened_images = images
    save_images(blurred_images, './images/blurred', name = 'blurred')
    save_images(sharpened_images, './images/sharpened', name = 'sharpened')
    merged_images = [cv2.addWeighted(sharpened_image, strength, blurred_image, 1 - strength, 0) for sharpened_image, blurred_image in zip(sharpened_images, blurred_images)]
    del sharpened_images, blurred_images
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