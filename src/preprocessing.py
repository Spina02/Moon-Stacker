import cv2
import numpy as np
from config import DEBUG
from utils import progress
from image import to_8bit
from denoise import model_init, perform_denoising
from image import *
from calibration import calibrate_images
from align import align_images
from skimage import color

# ------------------ Filters ------------------

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

def force_background_to_black(image, threshold_value=0.03):
    _, corrected_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_TOZERO)
    return corrected_image

# ------------------ Unsharp Masking ------------------

def gradient_mask_denoise_unsharp(images, model, strength=1.0, threshold=0.02, denoise_strength = 0.7):
    sharpened_images = []

    for idx, image in enumerate(images):

        # Apply denoising with DnCNN
        denoised_image = perform_denoising(model, image)

        denoised_image = denoised_image * denoise_strength + image * (1 - denoise_strength)

        # If the image has more than two dimensions, calculate the gradient for each channel
        if image.ndim == 3:  # Color image (e.g., RGB)
            gradient_magnitude = np.zeros_like(image)
            for i in range(image.shape[2]):  # For each channel
                gradient_x, gradient_y = np.gradient(image[:, :, i])
                gradient_magnitude[:, :, i] = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Average the gradients across channels
            gradient_magnitude = np.mean(gradient_magnitude, axis=2)
        else:
            # Grayscale image
            gradient_x, gradient_y = np.gradient(image)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Create the mask: where the gradient is below the threshold, apply denoising
        denoise_mask = np.where(gradient_magnitude < threshold, 1, 0)

        if denoise_mask is None:
            raise ValueError("denoise mask is None")

        # If the image is RGB, expand the denoise_mask to have 3 channels
        if image.ndim == 3:
            denoise_mask = np.repeat(denoise_mask[:, :, np.newaxis], 3, axis=2)

        # Apply the mask: in areas with low gradient, use the denoised image
        # else, retain the details of the original image
        final_image = (denoised_image * denoise_mask) + (image * (1 - denoise_mask))

        # Slightly blur the mask using GaussianBlur
        #denoise_mask = cv2.GaussianBlur(denoise_mask.astype(np.float32), (3, 3), 0.5)

        # Apply a slight unsharp mask in areas with high gradient
        detail_mask = 1 - denoise_mask  # Inverse mask for areas with details

        # Amplify the details in the areas selected by the mask
        amplified_details = (image - denoised_image) * strength * detail_mask
        
        # Add the amplified details to the final image
        sharpened_image = (final_image + amplified_details).astype(np.float32)

        # Clip the image to keep the valid range [0, 1]
        sharpened_image = np.clip(sharpened_image, 0, 1)

        sharpened_images.append(sharpened_image)

    return sharpened_images

def unsharp_mask(images, strength):

    blurred_images = [cv2.GaussianBlur(image, (3, 3), 0.5) for image in images]

    merged_images = [cv2.addWeighted(image, 0.5 + strength, blurred_image, 0.5 -strength, 0) for image, blurred_image in zip(images, blurred_images)]

    return merged_images

# ------------------ Cropping ------------------

def crop_to_center(images, margin=10):
    cropped_images = []

    # Process the first image to get the cropping parameters
    first_image = images[0]
    if len(first_image.shape) == 3:
        gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = first_image

    # Apply Gaussian blur to reduce noise
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)

    # Binary thresholding
    _, thresh = cv2.threshold(gray, 0.1, 1.0, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(to_8bit(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    del images
    gc.collect()

    return cropped_images

# --------------- Preprocessing ----------------

def preprocess_images(images, calibrate=False,
                      align=True, algo='orb', nfeatures=5000, 
                      crop=True, margin=10,
                      unsharp=True, gradient_strength=1.5, gradient_threshold=0.0075, denoise_strength=0.5,
                      grayscale=True):
    imgs = images.copy()
    
    if calibrate:
        imgs = calibrate_images(imgs)

    #imgs = [force_background_to_black(enhance_contrast(image, clip_limit=1, tile_grid_size=(9, 9))) for image in imgs]
    
    if align:
        imgs = align_images(imgs, algo=algo, nfeatures=nfeatures)
    
    if crop:
        imgs = crop_to_center(imgs, margin=margin)
        
    if unsharp:
        imgs = gradient_mask_denoise_unsharp(imgs, model_init(), strength=gradient_strength, threshold=gradient_threshold, denoise_strength = denoise_strength)

    if grayscale and len(imgs[0].shape) == 3:
        imgs = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in imgs]
    
    return imgs