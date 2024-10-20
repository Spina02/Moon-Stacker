import cv2
import numpy as np
from config import DEBUG
from utils import progress
from image import to_8bit
from denoise import perform_denoising#, model_init
#from calibration import calibrate_images
#from align import align_images

def force_background_to_black(image, threshold_value=0.03):
    _, corrected_image = cv2.threshold(image, threshold_value, 1, cv2.THRESH_TOZERO)
    return corrected_image

# ------------------ Unsharp Masking ------------------

def gradient_mask_denoise_unsharp(images, model, strength=1.0, threshold=0.02, denoise_strength = 0.7, blur = False):
    sharpened_images = []

    for idx, image in enumerate(images):

        # Apply denoising with DnCNN
        denoised_image = perform_denoising(model, image)

        denoised_image = denoised_image * denoise_strength + image * (1 - denoise_strength)

        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray_image = image.astype(np.float32)
        
        # Calcola il gradiente
        gradient_x, gradient_y = np.gradient(gray_image)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Create the mask: where the gradient is below the threshold, apply denoising
        denoise_mask = np.where(gradient_magnitude < threshold, 1, 0)

        denoise_mask = cv2.GaussianBlur(denoise_mask.astype(np.float32), (3, 3), 0.5)

        if denoise_mask is None:
            raise ValueError("denoise mask is None")

        # If the image is RGB, expand the denoise_mask to have 3 channels
        if image.ndim == 3:
            denoise_mask = np.repeat(denoise_mask[:, :, np.newaxis], 3, axis=2)

        detail_mask = 1 - denoise_mask  # Inverse mask for areas with details

        # Apply the mask: in areas with low gradient, use the denoised image
        # else, retain the details of the original image
        final_image = (denoised_image * denoise_mask) + (image * detail_mask)

        # Amplify the details in the areas selected by the mask
        amplified_details = (image - denoised_image) * strength * detail_mask
        
        # Add the amplified details to the final image
        sharpened_image = (final_image + amplified_details).astype(np.float32)

        # Clip the image to keep the valid range [0, 1]
        sharpened_image = np.clip(sharpened_image, 0, 1)

        sharpened_images.append(sharpened_image)
        if DEBUG: progress(idx, len(images), "images unsharped")

    return sharpened_images

def unsharp_mask(image, strength):

    blurred_image = cv2.GaussianBlur(image, (3, 3), 0.5)

    merged_image = cv2.addWeighted(image, 0.5 + strength, blurred_image, 0.5 -strength, 0)

    return merged_image

# ------------------ Cropping ------------------

def crop_to_center(images, margin=10):
    cropped_images = []

    # Process the first image to get the cropping parameters
    first_image = images[0]
    if len(first_image.shape) == 3:
        gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = first_image

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
        #cropped_image = force_background_to_black(cropped_image)
        cropped_image = np.clip(cropped_image, 0, 1)
        cropped_images.append(cropped_image)

        if DEBUG: progress(len(cropped_images), len(images), 'images cropped')

    return cropped_images