import config
from image import read_images
import numpy as np
import gc
from config import DEBUG
from utils import progress

def calculate_maser_bias(bias):
    if len(bias) < 10: 
      print('No bias frames found')
      return None
    master_bias = np.mean(bias, axis=0).astype(np.float64)
    return master_bias.astype(np.float64)

def calculate_master_dark(dark, master_bias = None):
    if len(dark) < 10: 
      print('No dark frames found')
      return None
    master_dark = np.mean(dark, axis=0)
    if master_bias is not None:
        master_dark -= master_bias
    return master_dark.astype(np.float64)

def calculate_master_flat(flat, master_bias = None, master_dark = None):
    if len(flat) < 10:
      print('No flat frames found')
      return None
    master_flat = np.mean(flat, axis=0)
    if master_bias is not None:
        master_flat -= master_bias
    if master_dark is not None:
        master_flat -= master_dark
    master_flat /= np.mean(master_flat, axis=(0, 1))
    return master_flat.astype(np.float64)

def calibrate_images(images):
    # calculate masters
    bias = read_images(config.bias_folder)
    master_bias = calculate_maser_bias(bias)
    del bias
    gc.collect()
   
    dark = read_images(config.dark_folder)
    master_dark = calculate_master_dark(dark, master_bias)

    del dark
    gc.collect()
    
    flat = read_images(config.flat_folder)
    master_flat = calculate_master_flat(flat, master_bias, master_dark)

    del flat
    gc.collect()

    calibrated_images = []
    for image in images:
        calibrated_image = image.copy()
        calibrated_image = calibrated_image.astype(np.float64)
        if master_bias is not None:
            calibrated_image -= master_bias
        if master_dark is not None:
            calibrated_image -= master_dark
        if master_flat is not None:
            calibrated_image /= master_flat
            
        calibrated_image = np.clip(calibrated_image, 0, np.finfo(np.float64).max)  # Clip to valid range
        calibrated_images.append(calibrated_image.astype(np.float32))  # Convert back to float32

        if DEBUG: progress(len(calibrated_images), len(images), 'images calibrated')

    print()

    return calibrated_images