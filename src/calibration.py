import config
from image import read_images, save_image, read_image
import numpy as np
import os
from config import DEBUG
from utils import progress

# Funzione per calcolare il master bias
def calculate_master_bias(bias):
    if len(bias) < config.MIN_CALIBRATION:
        if len(bias) == 0:
            print('No bias frames found')
        else:
            print('Not enough bias frames found')
        return None
    
    print("Calculating master bias...")
    try:
        # Calculate mean iteratively to save memory
        master_bias = np.zeros_like(bias[0], dtype=np.float32)
        for i in range(len(bias)):
            master_bias += bias[i] / len(bias)
        
        if master_bias is None or not np.issubdtype(master_bias.dtype, np.number):
            print("Error: Master bias is not numeric.")
            return None
        print("Master bias calculated successfully.")
    except Exception as e:
        print(f"Exception occurred while calculating master bias: {e}")
        return None

    return master_bias

# Funzione per calcolare il master dark
def calculate_master_dark(dark, master_bias=None):
    if len(dark) < config.MIN_CALIBRATION:
        if len(dark) == 0:
            print('No dark frames found')
        else:
            print('Not enough dark frames found')
        return None
    
    print("Calculating master dark...")
    try:
        # Calculate mean iteratively to save memory
        master_dark = np.zeros_like(dark[0], dtype=np.float32)
        for i in range(len(dark)):
            master_dark += dark[i] / len(dark)
        
        if master_dark is None or not np.issubdtype(master_dark.dtype, np.number):
            print("Error: Master dark is not numeric.")
            return None
        if master_bias is not None:
            master_dark -= master_bias
        print("Master dark calculated successfully.")
    except Exception as e:
        print(f"Exception occurred while calculating master dark: {e}")
        return None

    return master_dark

# Funzione per calcolare il master flat
def calculate_master_flat(flat, master_bias=None, master_dark=None):
    if len(flat) < config.MIN_CALIBRATION:
        if len(flat) == 0:
            print('No flat frames found')
        else:
            print('Not enough flat frames found')
        return None

    print("Calculating master flat...")
    try:
        # Calculate mean iteratively to save memory
        master_flat = np.zeros_like(flat[0], dtype=np.float32)
        for i in range(len(flat)):
            master_flat += flat[i] / len(flat)
        
        if master_flat is None or not np.issubdtype(master_flat.dtype, np.number):
            print("Error: Master flat is not numeric.")
            return None
        if master_bias is not None:
            master_flat -= master_bias
        if master_dark is not None:
            master_flat -= master_dark
        master_flat /= np.mean(master_flat, axis=(0, 1))
        print("Master flat calculated successfully.")
    except Exception as e:
        print(f"Exception occurred while calculating master flat: {e}")
        return None

    return master_flat

def calculate_masters(master_bias = None, master_dark = None, master_flat = None, max_img=config.MAX_CALIBRATION, save=True):
    if master_bias is None:
        print('Calculating bias master...')
        if not os.exist(config.bias_folder):
            print('No bias folder found. Skipping bias calculation.')
            master_bias = None
        bias_images = read_images(config.bias_folder, max_img=max_img)
        if bias_images is None or len(bias_images) < config.MIN_CALIBRATION:
            print('Not enough bias images. Skipping bias calculation.')
            master_bias = None
        else:
            master_bias = calculate_master_bias(bias_images)
            if save and master_bias is not None:
                try:
                    save_image(master_bias, 'bias', config.masters_folder, out_format='tif', dtype = np.float32)
                except Exception as e:
                    print(f"Failed to save master bias: {e}")
        del bias_images

    if master_dark is None:
        print('Calculating dark master...')
        if not os.exist(config.dark_folder):
            print('No dark folder found. Skipping dark calculation.')
            master_dark = None
        dark_images = read_images(config.dark_folder, max_img=max_img)
        if dark_images is None or len(dark_images) < config.MIN_CALIBRATION:
            print('Not enough dark images. Skipping dark calculation.')
            master_dark = None
        else:
            master_dark = calculate_master_dark(dark_images, master_bias)
            if save and master_dark is not None:
                try:
                    save_image(master_dark, 'dark', config.masters_folder, out_format='tif', dtype = np.float32)
                except Exception as e:
                    print(f"Failed to save master dark: {e}")
        del dark_images

    if master_flat is None:
        print('Calculating flat master...')
        if not os.exist(config.flat_folder):
            print('No flat folder found. Skipping flat calculation.')
            master_flat = None
        flat_images = read_images(config.flat_folder, max_img=max_img)
        if flat_images is None or len(flat_images) < config.MIN_CALIBRATION:
            print('Not enough flat images. Skipping flat calculation.')
            master_flat = None
        else:
            master_flat = calculate_master_flat(flat_images, master_bias, master_dark)
            if save and master_flat is not None:
                try:
                    save_image(master_flat, 'flat', config.masters_folder, out_format='tif', dtype = np.float32)
                except Exception as e:
                    print(f"Failed to save master flat: {e}")
        del flat_images

    return master_bias, master_dark, master_flat

def calibrate_images(images, master_bias, master_dark, master_flat):

    calibrated_images = []
    for image in images:
        calibrated_image = image.astype(np.float32)
        if master_bias is not None:
            calibrated_image -= master_bias
        if master_dark is not None:
            calibrated_image -= master_dark
        if master_flat is not None:
            calibrated_image /= master_flat
            
        calibrated_image = np.clip(calibrated_image, 0, np.finfo(np.float32).max)  # Clip to valid range
        calibrated_images.append(calibrated_image)

        if DEBUG: progress(len(calibrated_images), len(images), 'images calibrated')

    print()

    return calibrated_images