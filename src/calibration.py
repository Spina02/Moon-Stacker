import config
from image import read_images
import numpy as np
import gc
from config import DEBUG
from utils import progress

def calculate_maser_bias(bias):
    if len(bias) < 10:
        if len(bias) == 0: 
            print('No bias frames found')
        else:
            print('Not enough bias frames found')
        return None
    master_bias = np.mean(bias, axis=0).astype(np.float64)
    return master_bias.astype(np.float64)

def calculate_master_dark(dark, master_bias = None):
    if len(dark) < 10: 
        if len(dark) == 0:
            print('No dark frames found')
        else:
            print('Not enough dark frames found')
        return None
    master_dark = np.mean(dark, axis=0)
    if master_bias is not None:
        master_dark -= master_bias
    return master_dark.astype(np.float64)

def calculate_master_flat(flat, master_bias = None, master_dark = None):
    if len(flat) < 10:
        if len(flat) == 0:
            print('No flat frames found')
        else:
            print('Not enough flat frames found')
        return None
    master_flat = np.mean(flat, axis=0)
    if master_bias is not None:
        master_flat -= master_bias
    if master_dark is not None:
        master_flat -= master_dark
    master_flat /= np.mean(master_flat, axis=(0, 1))
    return master_flat.astype(np.float64)

def calculate_masters(max_img = 20):
    print('calculating bias master')
    master_bias = calculate_maser_bias(read_images(config.bias_folder, max_img=max_img))
    gc.collect()
   
    print('calculating dark master')
    master_dark = calculate_master_dark(read_images(config.dark_folder, max_img=max_img), master_bias)
    gc.collect()
    
    print('calculating flat master')
    master_flat = calculate_master_flat(read_images(config.flat_folder, max_img=max_img), master_bias, master_dark)
    gc.collect()

    return master_bias, master_dark, master_flat

def calibrate_images(images, master_bias, master_dark, master_flat):

    calibrated_images = []
    for image in images:
        calibrated_image = image.astype(np.float64)
        if master_bias is not None:
            calibrated_image -= master_bias
        if master_dark is not None:
            calibrated_image -= master_dark
        if master_flat is not None:
            calibrated_image /= master_flat
            
        calibrated_image = np.clip(calibrated_image, 0, np.finfo(np.float64).max)  # Clip to valid range
        calibrated_images.append(calibrated_image.astype(np.float32))  # Convert back to float32

        if DEBUG: progress(len(calibrated_images), len(images), 'images calibrated')

    del master_bias, master_dark, master_flat
    gc.collect()

    print()

    return calibrated_images