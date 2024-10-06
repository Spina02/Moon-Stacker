import rawpy
import imageio
import os
from utils import *
import config
from config import DEBUG, MAX_IMG
import numpy as np
import cv2
import gc

def normalize(image):
    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

def to_float32(image):
    if image.dtype == np.float32:
        return image
    if image.dtype == np.float64:
        return image.astype(np.float32)
    elif image.dtype == np.uint8:
        return (image/255.0).astype(np.float32)
    elif image.dtype == np.uint16:
        return (image/65535.0).astype(np.float32)
    else:
        raise ValueError(f"Invalid image type: {image.dtype}")

def to_8bit(image):
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.float32 or image.dtype == np.float64:
        return (image*255).astype(np.uint8)
    elif image.dtype == np.uint16:
        return (image/255).astype(np.uint8)
    
def to_16bit(image):
    if image.dtype == np.uint16:
        return image
    elif image.dtype == np.float32 or image.dtype == np.float64:
        return (image*65535).astype(np.uint16)
    elif image.dtype == np.uint8:
        return (image*255).astype(np.uint16)

def read_folder(folder_path, max_img):
    folder = os.listdir(folder_path)
    # Read all image paths in the folder
    return [os.path.join(folder_path, folder[i]) for i in range(min(max_img, len(folder)))]

def read_image(file_path):
    if file_path.lower().endswith(('.raf', '.dng', '.nef', '.cr2')):
        with rawpy.imread(file_path) as raw:
            # convert to rgb image using the camera white balance
            return to_float32(raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16))
    else:
        return to_float32(imageio.imread(file_path))
     

def read_images(folder_path, max_img=MAX_IMG):
    image_paths = read_folder(folder_path, max_img)
    images = []
    for path in image_paths:
        images.append(read_image(path))
        if DEBUG: progress(len(images), len(image_paths), f'images read')
        gc.collect()

    del image_paths
    gc.collect()
    
    return images

def save_image(image, name = 'image', folder_path = config.output_folder, out_format=config.output_format.lower(), dtype = None):
    create_folder(folder_path)

    if image is None:
        raise ValueError("image is None")

    if not name.endswith(f'.{out_format}'):
        name += f'.{out_format}'

    file_path = os.path.join(folder_path, name)

    if dtype == None:
        if out_format not in ['tif','tiff']:
            image = to_8bit(image)
            imageio.imsave(file_path, image)
        else:
            image = to_16bit(image)
            imageio.imsave(file_path, image)

    else:
        image = image.astype(dtype)
        imageio.imsave(file_path, image, compression='lzw')

    del image
    gc.collect()

def save_images(images, name=None, folder_path = config.output_folder, out_format=config.output_format, clear=True, dtype = None):
    create_folder(folder_path)
    if clear:
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))
    for i, image in enumerate(images):
        if image is not None:
            # check if name is a list
            if isinstance(name, list):
                file_name = name[i]
            else:
                file_name = f'output_{i}' if name is None else f'{name}_{i}'
            save_image(image, file_name, folder_path, out_format, dtype)
            if DEBUG: progress(i + 1, len(images), f'images saved')

    del images
    gc.collect()