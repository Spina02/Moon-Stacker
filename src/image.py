import rawpy
import imageio
import os
from PIL import Image
from debug import *
from const import *
import numpy as np

def to_8bit(image):
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
    else:
        print(f'\nImage has dtype: {image.dtype}\n')
        image_8bit = image
    return image_8bit

def to_16bit(image):
    if image.dtype == np.uint8:
        image_16bit = (image.astype(np.float32) * 256).astype(np.uint16)
    else:
        print(f'\nImage has dtype: {image.dtype}\n')
        image_16bit = image
    return image_16bit

def read_folder(folder_path):
    image_paths = []
    i = 0
    # Implement the function to read all image paths in the folder
    for f in os.listdir(folder_path):
        i += 1
        if i > MAX_IMG:
            break
        #if not (folder_path.endswith('raw') and not f.endswith(('.raf', '.dng', '.nef', '.cr2'))):
        image_paths.append(os.path.join(folder_path, f))
    return image_paths

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
    print("\n")
    for path in image_paths:
        image = read_image(path)
        images.append(image)
        if DEBUG: progress(len(images), len(image_paths), f'images read')

    return images

def save_image(image, file_path, out_format='jpeg'):
    if not file_path.endswith(f'.{out_format.lower()}'):
        file_path += f'.{out_format.lower()}'

    if image.dtype == np.uint16 and out_format.lower() not in ['tiff']:#, 'jpeg']:
        image = to_8bit(image)

    imageio.imsave(file_path, image)#, format=out_format)

def save_images(images, folder_path, out_format='png', name = None, clear = True):
    print()
    if clear:
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))
    for i, image in enumerate(images):
        save_image(image, f'{folder_path}/' + (f'output' if name is None else name) + f'_{i}', out_format)
        if DEBUG: progress(i + 1, len(images), f'images saved')