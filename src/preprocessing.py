import cv2
import numpy as np
from config import DEBUG
from utils import progress
from image import to_8bit
from denoise import model_init, dncnn_images, perform_denoising
from image import *
import gc
from calibration import calibrate_images
from align import align_images
from denoise import dncnn_images
from skimage import color
import torch

# ------------------ Filters ------------------

def sobel_images(images, ksize=5, sigma = 0, tale = (9, 9), low_clip = 0.05, high_clip = 0.35):
    sobel_images = []
    for image in images:
        # Split the image into channels
        channels = cv2.split(image)
        sobel_channels = []

        for channel in channels:
            # Apply Sobel filter
            sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=ksize)

            # Combine Sobel X and Y
            sobel = np.hypot(sobel_x, sobel_y)

            # Normalize the Sobel result
            sobel = cv2.normalize(sobel, None, 0, 1, cv2.NORM_MINMAX)

            # Apply Gaussian blur to smooth the mask
            sobel = cv2.GaussianBlur(sobel, tale, sigma)

            sobel = np.clip(sobel, low_clip, high_clip)  # Clip to a maximum value (adjust 0.5 as needed)

            sobel_channels.append(sobel)

        # Merge the channels back
        sobel_image = cv2.merge(sobel_channels)
        sobel_image = normalize(sobel_image)
        sobel_images.append(sobel_image)

        save_images([to_16bit(sobel_image)], './images/merged', name='sobel')

    return sobel_images

def white_balance(image):
    # Convert the image to float32 for precision
    image = image.astype(np.float32)

    # Calculate the average of each channel
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])

    # Calculate the overall average
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Scale each channel to match the overall average
    image[:, :, 0] = image[:, :, 0] * (avg_gray / avg_b)
    image[:, :, 1] = image[:, :, 1] * (avg_gray / avg_g)
    image[:, :, 2] = image[:, :, 2] * (avg_gray / avg_r)

    # Clip the values to the valid range [0, 255] and convert back to uint16
    image = to_16bit(image)

    return image

def enhance_contrast(image, clip_limit=0.25, tile_grid_size=(2, 2)):
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Convert the image to LAB color space using skimage
    lab = color.rgb2lab(image)

    # Ensure the L channel is in the correct format
    l_channel = lab[:, :, 0]

    # Normalize the L channel to the range [0, 255] for OpenCV processing
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply CLAHE on the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_equalized = clahe.apply(l_channel)

    # Normalize the L channel back to the range [0, 100] for LAB color space
    l_channel_equalized = l_channel_equalized.astype(np.float32) / 255 * 100

    # Replace the L channel in the LAB image
    lab[:, :, 0] = l_channel_equalized

    # Convert the image back to RGB color space using skimage
    enhanced_image = color.lab2rgb(lab)

    # Convert the image to 8-bit format for saving
    enhanced_image = to_8bit(enhanced_image)

    return enhanced_image

# ------------------ Unsharp Masking ------------------

def compute_gradient_magnitude(image):
    if image.ndim == 3:
        # Convert to grayscale for gradient computation
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray_image = image

    gradient_x, gradient_y = np.gradient(gray_image)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # make gradient_magnitude 3D if the input image is 3D
    if image.ndim == 3:
        gradient_magnitude = np.repeat(gradient_magnitude[:, :, np.newaxis], 3, axis=2)
    return gradient_magnitude

def new_gmdunsharp(images, model, denoise_strength=0.8, threshold=0.02, blur_kernel_size=3, sharpen_strength=1.0):
    sharpened_images = []

    for idx, image in enumerate(images):
        # Normalize the image
        normalized_image = image.astype(np.float32) / 65535.0

        # Apply denoising with adjustable strength
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        denoised_image = perform_denoising(model, normalized_image, device)

        # Blend the denoised image with the original based on denoise_strength
        denoised_image = denoised_image * denoise_strength + normalized_image * (1 - denoise_strength)

        # Compute gradient magnitude
        gradient_magnitude = compute_gradient_magnitude(normalized_image)

        # Create the denoise mask
        denoise_mask = np.where(gradient_magnitude < threshold, 1.0, 0.0).astype(np.float32)

        # Smooth the denoise mask for softer transitions
        denoise_mask = cv2.GaussianBlur(denoise_mask, (blur_kernel_size, blur_kernel_size), 0.5)

        # Apply the denoise mask to blend denoised and original images
        blended_image = np.clip(denoised_image * denoise_mask + normalized_image * (1 - denoise_mask), 0, 1)

        # Apply unsharp masking to the entire image
        #sharpened_image = unsharp_mask(blended_image, strength=sharpen_strength)
        sharpened_image = blended_image

        if idx == 0: save_image(to_8bit(sharpened_image), './debug_images', f'new_gmdunsharp_{denoise_strength}_{threshold}_{blur_kernel_size}_{sharpen_strength}')

        # Clip and convert back to uint16
        sharpened_image = np.clip(sharpened_image, 0, 1) * 65535
        sharpened_image_16bit = sharpened_image.astype(np.uint16)
        sharpened_images.append(sharpened_image_16bit)

    return sharpened_images

def gradient_mask_denoise_unsharp(images, model, strength=1.0, threshold=0.02, denoise_strength = 0.7):
    sharpened_images = []

    for idx, image in enumerate(images):
        # Normalizza l'immagine
        normalized_image = image.astype(np.float32) / 65535.0

        # Applica il denoising con DnCNN
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        denoised_image = perform_denoising(model, image, device)

        denoised_image = denoised_image * denoise_strength + normalized_image * (1 - denoise_strength)

        # Se l'immagine ha più di due dimensioni, calcola il gradiente per ogni canale
        if normalized_image.ndim == 3:  # Immagine a colori (es. RGB)
            gradient_magnitude = np.zeros_like(normalized_image)
            for i in range(normalized_image.shape[2]):  # Per ogni canale
                gradient_x, gradient_y = np.gradient(normalized_image[:, :, i])
                gradient_magnitude[:, :, i] = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Fai la media dei gradienti sui canali
            gradient_magnitude = np.mean(gradient_magnitude, axis=2)
        else:
            # Immagine in scala di grigi
            gradient_x, gradient_y = np.gradient(normalized_image)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Crea la maschera: dove il gradiente è sotto la soglia, applichiamo il denoising
        denoise_mask = np.where(gradient_magnitude < threshold, 1, 0)

        # Se l'immagine è RGB, espandi la maschera denoise_mask per avere 3 canali
        if normalized_image.ndim == 3:
            denoise_mask = np.repeat(denoise_mask[:, :, np.newaxis], 3, axis=2)

        # Applica la maschera: nelle aree con basso gradiente, usa l'immagine denoised
        # nelle aree con alto gradiente, mantieni i dettagli dell'immagine originale
        #!! modified here
        final_image = (denoised_image * denoise_mask) + (normalized_image * (1 - denoise_mask))

        # Sfuma leggermente la maschera usando GaussianBlur
        #denoise_mask = cv2.GaussianBlur(denoise_mask.astype(np.float32), (3, 3), 0.5)

        # Applica un leggero unsharp mask nelle aree ad alto gradiente
        detail_mask = 1 - denoise_mask  # Maschera inversa per le aree con dettagli

        # Amplifica i dettagli nelle aree selezionate dalla maschera
        amplified_details = (normalized_image - denoised_image) * strength * detail_mask

        # Somma i dettagli amplificati all'immagine finale
        sharpened_image = final_image + amplified_details

        # Clip per mantenere il range valido [0, 65535]
        sharpened_image = np.clip(sharpened_image, 0, 1) * 65535

        #if idx == 0:
        #    save_image(to_8bit(sharpened_image), './debug_images', f'gmdunsharp_{strength}_{threshold}')
        #    save_image(to_8bit(detail_mask), './debug_images', f'detail_mask_{strength}_{threshold}')

        # Converti di nuovo a 16-bit
        sharpened_image_16bit = sharpened_image.astype(np.uint16)
        sharpened_images.append(sharpened_image_16bit)

    return sharpened_images


def multi_scale_unsharp_mask_dncnn(images, model, strengths, thresholds, ks):
    import torch

    sharpened_images = []

    for idx, image in enumerate(images):
        normalized_image = image.astype(np.float32)

        total_details = np.zeros_like(normalized_image)

        for scale_idx, (strength, threshold, k) in enumerate(zip(strengths, thresholds, ks)):
            # Denoising con DNCNN
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            denoised_image = perform_denoising(model, image, device)

            # Calcola i dettagli
            details = normalized_image - denoised_image * 65535

            # Calcola la magnitudine dei dettagli
            detail_magnitude = np.abs(details)

            # Applica una soglia ai dettagli per filtrare il rumore
            detail_threshold = np.mean(detail_magnitude) + np.std(detail_magnitude) * 1.5
            details_filtered = np.where(detail_magnitude >= detail_threshold, details, 0)

            # Salva i dettagli filtrati per il debugging
            #save_debug_image(details_filtered, './debug_images', f'details_filtered_{idx}_scale_{scale_idx}')

            # Normalizza utilizzando la deviazione standard dei dettagli filtrati
            std_dev = np.std(details_filtered)
            if std_dev > 0:
                detail_magnitude = np.abs(details_filtered) / std_dev
            else:
                detail_magnitude = np.zeros_like(details_filtered)

            # Applica una maschera morbida con una funzione sigmoide
            detail_mask = 1 / (1 + np.exp(-k * (detail_magnitude - threshold)))

            # apply gaussian blur to the mask
            detail_mask = cv2.GaussianBlur(detail_mask, (5, 5), 1)

            # Amplifica i dettagli utilizzando la maschera
            amplified_details = details_filtered * strength * detail_mask

            # Limita i dettagli amplificati
            max_amplification = 3000  # Regola questo valore secondo necessità
            amplified_details = np.clip(amplified_details, -max_amplification, max_amplification)

            # Somma i dettagli amplificati
            total_details += amplified_details / len(strengths)

        # Aggiungi i dettagli totali all'immagine originale
        sharpened_image = 0.7*image + 0.3*denoised_image + total_details

        # Clip per mantenere il range valido [0, 65535]
        sharpened_image = np.clip(sharpened_image, 0, 65535)

        # Converti di nuovo a 16-bit
        sharpened_image_16bit = sharpened_image.astype(np.uint16)

        sharpened_images.append(sharpened_image_16bit)

        if DEBUG:
            progress(idx + 1, len(images), 'Images processed with multi-scale unsharp masking')

    return sharpened_images

# ------------------ Cropping ------------------

def crop_to_center(images, margin=10):
    cropped_images = []

    # Process the first image to get the cropping parameters
    first_image = to_8bit(images[0])
    if len(first_image.shape) == 3:
        gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = first_image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary thresholding
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the selected contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the center of the bounding rectangle
    center_x, center_y = x + w // 2, y + h // 2

    # Determine the size of the square
    size = max(w, h) + 2 * margin

    # Calculate the top-left corner of the square
    start_x = max(center_x - size // 2, 0)
    start_y = max(center_y - size // 2, 0)

    # Ensure the square fits within the image boundaries
    end_x = min(start_x + size, first_image.shape[1])
    end_y = min(start_y + size, first_image.shape[0])

    # Crop all images using the same parameters
    for image in images:
        cropped_image = image[start_y:end_y, start_x:end_x]
        cropped_images.append(cropped_image)

        if DEBUG: progress(len(cropped_images), len(images), 'images cropped')

    return cropped_images

# --------------- Preprocessing ----------------

def preprocess_images(images, calibrate=True,
                      align=True, algo='orb', nfeatures=10000, 
                      crop=True, margin=10,
                      unsharp=True, strengths=None, thresholds=None, ks=None,
                      grayscale=True, sharpening_method='multi_scale', gradient_strength=1.0, gradient_threshold=0.02, denoise_strength = 0.7):
    if calibrate:
        imgs = calibrate_images(images)
    else:
        imgs = images.copy()
    
    if align:
        imgs = align_images(imgs, algo=algo, nfeatures=nfeatures)
    
    if crop:
        imgs = crop_to_center(imgs, margin=margin)
    
    if unsharp:
        # Inizializza il modello DNCNN una volta
        model = model_init()
        
        # Imposta parametri di default se non forniti (multi-scale unsharp mask)
        if strengths is None:
            strengths = [0.8, 0.9, 1]
        if thresholds is None:
            thresholds = [0.5, 0.5, 0.5]
        if ks is None:
            ks = [5, 5, 5]

        if sharpening_method == 'multi_scale':
            # Applica multi-scale unsharp mask
            imgs = multi_scale_unsharp_mask_dncnn(imgs, model, strengths, thresholds, ks)
        elif sharpening_method == 'gradient':
            # Applica il denoising selettivo con maschera di gradiente
            imgs = gradient_mask_denoise_unsharp(imgs, model, strength=gradient_strength, threshold=gradient_threshold, denoise_strength = denoise_strength)
            #imgs = new_gmdunsharp(imgs, model, denoise_strength=gradient_strength, threshold=gradient_threshold)
            #imgs = align_images(imgs, algo=algo, nfeatures=nfeatures)
        else:
            raise ValueError(f"Invalid sharpening method: {sharpening_method}")
    
    if grayscale:
        imgs = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image for image in imgs]
    
    return imgs