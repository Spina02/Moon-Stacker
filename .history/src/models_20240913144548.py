import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from image import save_images, read_image, read_images, save_image
import cv2
import gc
from utils import progress

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

def load_model(model_path):
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
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)

    del transform
    gc.collect()

    return image

def postprocess_image(image):
    image = image.squeeze(0).detach().cpu()
    image = image * 0.5 + 0.5
    image = image.numpy().squeeze()
    return (image * 255).astype(np.uint8)

def denoise_tensor(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)

def model_init(model_path = './models/DnCNN-PyTorch/logs/DnCNN-S-25/net.pth'):
    return load_model(model_path)

def perform_denoising(model, image, device):
    l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))

    model = model.to(device)
    model.eval()

    # Preprocess the image
    l = preprocess_image(l)
    l = l.to(device)

    # Apply denoising to the L channel
    with torch.no_grad():
        denoised_l = model(l).cpu()

    # Postprocess the image
    denoised_l = postprocess_image(denoised_l)

    # Merge the channels
    denoised_image = cv2.merge((denoised_l, a, b))
    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_LAB2RGB)

    # Free memory
    del l, a, b, denoised_l
    torch.cuda.empty_cache()
    gc.collect()

    return denoised_image

def unsharp_mask(images, sigma=1.5, strength=3, model=None):
    if model:
        blurred_images = dncnn_images(model, images)
    else:
        blurred_images = [cv2.GaussianBlur(image, (0, 0), sigma) for image in images]

    sharpened_images = [cv2.addWeighted(image, 1 + strength, blurred_image, -strength, 0) for image, blurred_image in zip(images, blurred_images)]
    
    del blurred_images
    gc.collect()

    return sharpened_images

def dncnn_images(model, images, batch_size=1):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.eval()

    denoised_images = []
    for image in images:
        denoised_image = perform_denoising(model, image, device)
        denoised_images.append(denoised_image)

        # Free memory
        del denoised_image
        torch.cuda.empty_cache()
        gc.collect()

        progress(len(denoised_images), len(images), 'Denoising')

    return denoised_images

if __name__ == "__main__":
    model_path =  './models/DnCNN-PyTorch/logs/DnCNN-S-25/net.pth'
    folder_path =  './images/jpg'
    image_path =  './images/extra/orb_median_5000.png'
    output_path = './images/denoised'

    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    print('Loaded model')

    #image = read_image(image_path)
    images = read_images(folder_path)

    print("Postprocessing image")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    denoised_images = []
    for image in images:
        denoised_images.append(perform_denoising(model, image, device))

    print('Postprocessed image')

    print(f"Saving denoised image to {output_path}")
    save_images(denoised_images, output_path)
    print('Saved image')