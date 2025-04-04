import cv2
import numpy as np
from config import DEBUG
from utils import progress
from image import to_8bit, to_float32, save_image
from denoise import perform_denoising

# ------------------ Enhancing ------------------

# Color correction
def shades_of_gray(image, power=6):
    import numpy as np
    norm_values = np.power(np.mean(np.power(image, power), axis=(0, 1)), 1/power)
    mean_norm = np.mean(norm_values)
    scale_factors = mean_norm / norm_values
    balanced_image = image * scale_factors
    balanced_image = np.clip(balanced_image, 0, 1)
    return balanced_image

# Apply a soft threshold to remove background
def soft_threshold(image, threshold = 0.07, alpha=50):
    mask = image < threshold
    smooth_image = image.copy()
    smooth_image[mask] = image[mask] * (1 / (1 + np.exp(-alpha * (image[mask] - threshold))))
    return smooth_image

def enhance_contrast(image, clip_limit, tile_grid_size):
    shape = len(image.shape)
    if shape < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert the image to LAB color space using skimage
    #lab = skimage.color.rgb2lab(image) 
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

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
    enhanced_image =  cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 
    if shape < 3 and len(enhanced_image.shape) == 3:
        enhanced_image = cv2.cvtColor(to_float32(enhanced_image), cv2.COLOR_RGB2GRAY)

    return enhanced_image

# ------------------ Unsharp Masking ------------------

def denoise(img, denoising_method, model = None):
    # Denoising
    if denoising_method == 'dncnn':
        denoised = perform_denoising(model, img)
    elif denoising_method == 'gaussian':
        denoised = cv2.GaussianBlur(img, (3, 3), 1)
    elif denoising_method == 'bilateral':
        denoised = cv2.bilateralFilter(img, 5, 50, 50)
    elif denoising_method == 'median':
        denoised = cv2.medianBlur(img, 3)
    else:
        raise ValueError(f"Unknown denoising method: {denoising_method}")

    save_image(denoised, f'denoised_{denoising_method}', './images/analysis')

    return denoised

def gradient_mask_denoise_unsharp(images, model, strength=1.0, threshold=0.02, denoise_strength = 0.7, blur = False, denoising_method = "dncnn"):
    sharpened_images = []

    for idx, image in enumerate(images):

        # Apply denoising with DnCNN
        denoised_image = denoise(image, denoising_method, model)

        denoised_image = denoised_image * denoise_strength + image * (1 - denoise_strength)

        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray_image = image.astype(np.float32)
        
        # Calcola il gradiente
        gradient_x, gradient_y = np.gradient(gray_image)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        save_image(np.clip(gradient_magnitude + 1, 0, 1), 'gradient', './images/')

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
        cropped_image = np.clip(cropped_image, 0, 1)
        cropped_images.append(cropped_image)

        if DEBUG: progress(len(cropped_images), len(images), 'images cropped')

    return cropped_images