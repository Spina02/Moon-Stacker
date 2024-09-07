from email.mime import image
import rawpy
import imageio
import os
import numpy as np
from debug import *

from const import *

def read_folder(folder_path):
    image_paths = []
    i = 0
    # Implement the function to read all image paths in the folder
    for f in os.listdir(folder_path):
        i += 1
        if i > MAX_IMG:
            break
        if not (folder_path.endswith('raw') and not f.endswith('.RAF')):
            image_paths.append(os.path.join(folder_path, f))
    return image_paths

def read_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
    return rgb

def read_image(file_path):
    return imageio.imread(file_path)

def read_images(folder_path):
    image_paths = read_folder(folder_path)
    images = []
    i = 1
    print("\n")
    for path in image_paths:
        if path.lower().endswith(('.raf', '.dng', '.nef')):
            image = read_raw_image(path)
        else:
            image = read_image(path)
        # Normalize image data to 8-bit range if necessary
        if image.dtype != 'uint8':
            image = (image / 256).astype('uint8')
        images.append(image)
        if DEBUG: progress(i + 1, len(image_paths), f'images read')
        i += 1

    return images

def save_image(image, file_path, out_format='png'):
    # Convert to 8-bit if the format is not TIFF
    # if image.dtype != 'uint8':
    #     image = (image / 256).astype('uint8')
    if not file_path.endswith(f'.{out_format.lower()}'):
        file_path += f'.{out_format.lower()}'
    imageio.imsave(file_path, image, format=out_format)

def save_images(images, folder_path, out_format='png'):
    for f in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, f))
    for i, image in enumerate(images):
        save_image(image, f'{folder_path}/output_{i}', out_format)
        if DEBUG: progress(i + 1, len(images), f'images saved')