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
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if config.DEBUG: print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

#class DnCNN(nn.Module):
#    def __init__(self, channels, num_of_layers=17, kernel_size=3, features=64):
#        super(DnCNN, self).__init__()
#        padding = 1
#        layers = []
#        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
#        layers.append(nn.ReLU(inplace=True))
#        for _ in range(num_of_layers-2):
#            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
#            layers.append(nn.BatchNorm2d(features))
#            layers.append(nn.ReLU(inplace=True))
#        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
#        self.dncnn = nn.Sequential(*layers)
#
#    def forward(self, x):
#        with torch.no_grad():
#            return x - self.dncnn(x)

def model_init(model_path = DNCNN_MODEL_PATH):
    globals()['DnCNN'] = DnCNN
    model = torch.load('./src/model.pth')
    #DnCNN()# channels=1), num_of_layers=17, kernel_size=3, features=64)
    #state_dict = torch.load(model_path, weights_only=True)#map_location=torch.device('cpu'))
    
    #new_state_dict = {}
    #for k, v in state_dict.items():
    #    if k.startswith('module.'):
    #        new_state_dict[k[7:]] = v
    #    else:
    #        new_state_dict[k] = v
    
    #model.load_state_dict(new_state_dict)
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
    denoised_l = reduce_banding_frequency(denoised_l)
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

def reduce_banding_frequency(image):

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Creare una maschera per filtrare le frequenze
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 10  # Raggio del cerchio da mascherare
    cv2.circle(mask, (ccol, crow), r, (0, 0, 0), -1)

    # Applicare la maschera
    fshift = dft_shift * mask

    # Trasformata inversa
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    # Normalizza l'immagine
    img_back = cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)
    return img_back