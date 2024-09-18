import rawpy
import imageio
import os
from utils import *
import config
from config import DEBUG, MAX_IMG
import numpy as np
import cv2

def normalize(image):
    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

def to_8bit(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def to_16bit(image):
    return cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)

def read_folder(folder_path):
    folder = os.listdir(folder_path)
    # Read all image paths in the folder
    return [os.path.join(folder_path, folder[i]) for i in range(min(MAX_IMG, len(folder)))]

def read_image(file_path):
    if file_path.lower().endswith(('.raf', '.dng', '.nef', '.cr2')):
        with rawpy.imread(file_path) as raw:
            # Extract the raw image data
            raw_data = raw.raw_image_visible.astype(np.float32)
            # Normalize the raw image data
            raw_data /= np.max(raw_data)
            # convert to rgb image using the camera white balance
            image = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
    else:
        image = imageio.imread(file_path)
        if image.dtype not in [np.uint8, np.uint16]:
            print(f'\nImage {file_path} has an unsupported dtype: {image.dtype}\n')
            exit(0)
    return image

def read_images(folder_path):
    image_paths = read_folder(folder_path)
    images = []
    for path in image_paths:
        images.append(read_image(path))
        if DEBUG: progress(len(images), len(image_paths), f'images read')
    return images

def save_image(image, folder_path, file_name, out_format=config.output_format.lower()):
    if not file_name.endswith(f'.{out_format}'):
        file_name += f'.{out_format}'

    file_path = os.path.join(folder_path, file_name)

    if image.dtype == np.uint16 and out_format not in ['tiff']:
        if DEBUG > 1: print(f'\nCan not save in 16-bit, converting to 8-bit')
        image = to_8bit(image)

    imageio.imsave(file_path, image)

def save_images(images, folder_path, out_format=config.output_format, name=None, clear=True):
    create_folder(folder_path)
    if clear:
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))
    for i, image in enumerate(images):
        file_name = f'output_{i}' if name is None else f'{name}_{i}'
        save_image(image, folder_path, file_name, out_format)
        if DEBUG: progress(i + 1, len(images), f'images saved')