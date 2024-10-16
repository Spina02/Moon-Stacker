import config
from image import read_images, save_image
import numpy as np
import os
from config import DEBUG
from utils import progress
from multiprocessing import Pool, cpu_count
from functools import partial

# Funzione per calcolare il master bias
def calculate_master_bias(bias):
    # Calculate mean iteratively to save memory
    master_bias = np.zeros_like(bias[0], dtype=np.float32)
    for i in range(len(bias)):
        master_bias += bias[i] / len(bias)
    
    if master_bias is None or not np.issubdtype(master_bias.dtype, np.number):
        print("Error: Master bias is not numeric.")
        return None
    print("Master bias calculated successfully.")

    return master_bias

# Funzione per calcolare il master dark
def calculate_master_dark(dark, master_bias=None):
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

    return master_dark

# Funzione per calcolare il master flat
def calculate_master_flat(flat, master_bias=None, master_dark=None):
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
    normalization_factor = np.mean(master_flat)
    if normalization_factor != 0:
        master_flat /= normalization_factor
    else:
        print("Error: Normalization factor is zero.")
        return None
    print("Master flat calculated successfully.")

def calculate_masters(master_bias = None, master_dark = None, master_flat = None, max_img=config.MAX_CALIBRATION, save=True):
    if master_bias is None:
        print('Calculating bias master...')
        if not os.path.exists(config.bias_folder):
            print('No bias folder found. Skipping bias calculation.')
        elif len(os.listdir(config.bias_folder)) < config.MIN_CALIBRATION:
            print('Not enough bias images. Skipping bias calculation.')
        else:
            bias_images = read_images(config.bias_folder, max_img=max_img)
            master_bias = calculate_master_bias(bias_images)
            if master_bias is not None:
                save_image(master_bias, 'bias', config.masters_folder, out_format='tif', dtype = np.float32)
            else:
                print("Error while calculating master bias")

    if master_dark is None:
        print('Calculating dark master...')
        if not os.path.exists(config.dark_folder):
            print('No dark folder found. Skipping dark calculation.')
        elif len(os.listdir(config.dark_folder)) < config.MIN_CALIBRATION:
            print('Not enough dark images. Skipping dark calculation.')
        else:
            dark_images = read_images(config.dark_folder, max_img=max_img)
            master_dark = calculate_master_dark(dark_images, master_bias)
            if master_dark is not None:
                save_image(master_dark, 'dark', config.masters_folder, out_format='tif', dtype = np.float32)
            else:
                print("Error while calculating master dark")

    if master_flat is None:
        print('Calculating flat master...')
        if not os.path.exists(config.flat_folder):
            print('No flat folder found. Skipping flat calculation.')
        elif len(os.listdir(config.flat_folder)) < config.MIN_CALIBRATION:
            print('Not enough flat images. Skipping flat calculation.')
        else:
            flat_images = read_images(config.flat_folder, max_img=max_img)
            master_flat = calculate_master_flat(flat_images, master_bias, master_dark)
            if master_flat is not None:
                save_image(master_flat, 'flat', config.masters_folder, out_format='tif', dtype = np.float32)
            else:
                print("Error while calculating master flat")

    return master_bias, master_dark, master_flat

def calibrate_single_image(image, master_bias, master_dark, master_flat):
    calibrated_image = image.astype(np.float32)
    if master_bias is not None:
        calibrated_image -= master_bias
    if master_dark is not None:
        calibrated_image -= master_dark
    if master_flat is not None:
        calibrated_image /= master_flat
    calibrated_image = np.clip(calibrated_image, 0, np.finfo(np.float32).max)  # Clip to valid range
    return calibrated_image

def calibrate_images(images, master_bias, master_dark, master_flat):
    print("Started image calibration")
    num_processes = min(cpu_count(), len(images))
    print(f"Utilizzo di {num_processes} processi per la calibrazione delle immagini.")

    # Creare una funzione parziale con i master predefiniti
    calibrate_partial = partial(calibrate_single_image, master_bias=master_bias, master_dark=master_dark, master_flat=master_flat)

    # Utilizzare un Pool di processi
    with Pool(processes=num_processes) as pool:
        # Mappare le immagini alla funzione di calibrazione
        calibrated_images = pool.map(calibrate_partial, images)

    # Se DEBUG è attivo, mostrare il progresso (non consigliato con multiprocessing per semplicità)
    if DEBUG:
        for idx, _ in enumerate(calibrated_images, 1):
            progress(idx, len(images), 'images calibrated')

    print("Calibrazione delle immagini completata.")
    return calibrated_images