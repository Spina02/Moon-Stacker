import cv2
import numpy as np
from config import DEBUG
from utils import progress
from image import to_8bit
from denoise import model_init, dncnn_images, perform_denoising
from image import save_images, normalize, to_16bit
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


def save_debug_image(image, path, name):
    """Salva un'immagine per il debugging."""
    save_images([to_16bit(image)], path, name=name, clear=False)

# ------------------ Unsharp Masking ------------------

def gradient_mask_denoise_unsharp(images, model, strength=1.0, threshold=0.02):
    sharpened_images = []

    for idx, image in enumerate(images):
        # Normalizza l'immagine
        normalized_image = image.astype(np.float32) / 65535.0

        # Applica il denoising con DnCNN
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        denoised_image = perform_denoising(model, image, device)

        # Calcola la magnitudine del gradiente per l'immagine originale
        gradient_x, gradient_y = np.gradient(normalized_image)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Crea la maschera: dove il gradiente è sotto la soglia, applichiamo il denoising
        denoise_mask = np.where(gradient_magnitude < threshold, 1, 0)

        # Applica la maschera: nelle aree con basso gradiente, usa l'immagine denoised
        # nelle aree con alto gradiente, mantieni i dettagli dell'immagine originale
        final_image = denoised_image * denoise_mask + normalized_image * (1 - denoise_mask)

        # Applica un leggero unsharp mask nelle aree ad alto gradiente
        detail_mask = 1 - denoise_mask  # Maschera inversa per le aree con dettagli
        detail_mask = cv2.GaussianBlur(detail_mask, (3, 3), 1)  # Sfuma leggermente la maschera

        # Amplifica i dettagli nelle aree selezionate dalla maschera
        amplified_details = (normalized_image - denoised_image) * strength * detail_mask

        # Somma i dettagli amplificati all'immagine finale
        sharpened_image = final_image + amplified_details

        # Clip per mantenere il range valido [0, 65535]
        sharpened_image = np.clip(sharpened_image, 0, 1) * 65535

        # Converti di nuovo a 16-bit
        sharpened_image_16bit = sharpened_image.astype(np.uint16)
        sharpened_images.append(sharpened_image_16bit)

        if DEBUG:
            save_debug_image(sharpened_image_16bit, './debug_images', f'gradient_mask_unsharp_{idx}')
    
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
            save_debug_image(details_filtered, './debug_images', f'details_filtered_{idx}_scale_{scale_idx}')

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

def high_frequency_emphasis_dncnn(images, model, boost=1.5):
    emphasized_images = []

    for idx, image in enumerate(images):
        image = normalize(image).astype(np.float32)
        denoised_image = dncnn_images(model, [image])[0]

        # Estrai le componenti ad alta frequenza
        high_freq = image - denoised_image

        # Enfatizza le componenti ad alta frequenza
        emphasized_image = image + boost * high_freq

        # Clip per mantenere il range valido [0, 1]
        emphasized_image = np.clip(emphasized_image, 0, 1)

        # Converti di nuovo a 16-bit
        emphasized_image = to_16bit(emphasized_image)

        emphasized_images.append(emphasized_image)

        if DEBUG:
            progress(idx + 1, len(images), 'Images processed with high-frequency emphasis')

    return emphasized_images

def unsharp_mask_2(images, model, strength, ksize=3, sigma = 0, tale = (9, 9), low_clip = 0.01, high_clip = 0.5):

    images = unsharp_mask(images, strength)
    sobel_masks = sobel_images(images, ksize=ksize, sigma = sigma, tale = tale, low_clip = low_clip, high_clip = high_clip)
    blurred_images = dncnn_images(model, images)
    merged_images = []

    i = 1
    for image, blurred_image, sobel_mask in zip(images, blurred_images, sobel_masks):
        
        # Convert everything to float32
        sharp_image = image.astype(np.float32)

        if i == 0:
            save_images([to_16bit(sharp_image)], './images/sharpened', name='sharp_pre', clear = False)
            save_images([to_16bit(blurred_image)], './images/blurred', name='blurred_pre', clear = False)

        sobel_mask = normalize(sobel_mask)

        # Apply sharpening with Sobel mask modulation
        sharp_component = sharp_image * (0.5 + normalize(sobel_mask * strength))
        sharp_component = normalize(sharp_component)
        sharp_component = np.clip(sharp_component, 0, 1)  # Clip to valid range

        # Apply blurring with inverted Sobel mask modulation
        blurred_component = blurred_image * normalize(0.5 - sobel_mask * strength)
        blurred_component = normalize(blurred_component)
        blurred_component = np.clip(blurred_component, 0, 1)  # Clip to valid range

        if i == 0:
            save_images([to_16bit(sharp_component)], './images/merged', name='sharp', clear = False)
            save_images([to_16bit(blurred_component)], './images/merged', name='blurred', clear = False)
            i += 1

        # Merge the components
        merged_image = cv2.addWeighted(sharp_component, 0.6, blurred_component, 0.4, 0)
        merged_image = np.clip(merged_image, 0, 1)  # Ensure valid range

        # Convert back to 16-bit
        merged_image = to_16bit(merged_image)
        merged_image = white_balance(merged_image)
        #merged_image = enhance_contrast(merged_image)
        merged_images.append(merged_image)

    del blurred_images, sobel_masks
    gc.collect()

    #save_images(merged_images[:1], './images/merged', name='merged', clear = False)

    return merged_images

def unsharp_mask(images, strength):
    
    #blurred_images = dncnn_images(model, images)
    blurred_images = [cv2.GaussianBlur(image, (3, 3), 3) for image in images]

    #save_images(blurred_images, './images/blurred', name = 'blurred')

    merged_images = [to_16bit(cv2.addWeighted(to_16bit(image), 0.5 + strength, to_16bit(blurred_image), 0.5 -strength, 0)) for image, blurred_image in zip(images, blurred_images)]
    #blurred_images = dncnn_images(model, images)
    blurred_images = [cv2.GaussianBlur(image, (3, 3), 3) for image in images]

    #save_images(blurred_images, './images/blurred', name = 'blurred')

    merged_images = [to_16bit(cv2.addWeighted(to_16bit(image), 0.5 + strength, to_16bit(blurred_image), 0.5 -strength, 0)) for image, blurred_image in zip(images, blurred_images)]
    
    save_images(merged_images, './images/merged', name = 'merged', clear = False)
    
    del blurred_images
    gc.collect()
    return merged_images

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
                      unsharp=True, strengths=None, thresholds=None, ks=None, noise_levels=None,
                      high_freq_emphasis=False, boost=1.5,
                      grayscale=True, sharpening_method='multi_scale', gradient_strength=1.0, gradient_threshold=0.02):
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
            imgs = gradient_mask_denoise_unsharp(imgs, model, strength=gradient_strength, threshold=gradient_threshold)
        else:
            raise ValueError(f"Invalid sharpening method: {sharpening_method}")
    
    if high_freq_emphasis:
        # Applica l'enfasi delle alte frequenze
        model = model_init()  # Assicurati che il modello sia inizializzato
        imgs = high_frequency_emphasis_dncnn(imgs, model, boost=boost)
    
    if grayscale:
        imgs = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image for image in imgs]
    
    return imgs