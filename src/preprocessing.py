import cv2
import numpy as np
from config import DEBUG
from utils import progress
from image import to_8bit
from denoise import model_init, dncnn_images
from image import save_images, normalize, to_16bit
import gc
from calibration import calibrate_images
from align import align_images
from denoise import dncnn_images

# ------------------ Filters ------------------

def sobel_images(images):
    sobel_images = []
    for image in images:
        # Split the image into its respective channels
        channels = cv2.split(image)
        sobel_channels = []

        for channel in channels:  # loop su ogni canale (RGB o altri canali)
            # Apply Sobel filter on the channel
            sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=5)

            # Combine Sobel X and Y
            sobel = np.hypot(sobel_x, sobel_y)

            # Apply Gaussian blur to smoothen edges
            sobel = cv2.GaussianBlur(sobel, (9, 9), 0)

            # Normalize the result to 0-1
            sobel = normalize(sobel)

            # Append the sobelized channel to the list
            sobel_channels.append(sobel)

        # Merge the sobelized channels back into an image
        sobel_image = cv2.merge(sobel_channels)

        # Append the sobelized image to the list
        sobel_images.append(sobel_image)

    return sobel_images

def equalize_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:  # Color image

        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127 + 1  # Alpha è il fattore di contrasto
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# ------------------ Unsharp Masking ------------------

def unsharp_mask(images, model, strength):
    blurred_images = dncnn_images(model, images)

    sobel_masks = sobel_images(images)

    merged_images = []

    for image, blurred_image, sobel_mask in zip(images, blurred_images, sobel_masks):
        image = image.astype(np.float32)
        sharp_image = adjust_brightness_contrast(image, brightness=0, contrast=10)

        # Adjust the strength based on the Sobel mask
        sharp_component = cv2.multiply(sharp_image, 1 + strength * sobel_mask)
        sharp_component = normalize(sharp_component)
        blurred_component = cv2.multiply(blurred_image, 1 - strength * sobel_mask)
        blurred_component = normalize(blurred_component)
        # manage too sharpened pixels
        sharp_component = np.clip(sharp_component, 0, 1)
        blurred_component = np.clip(blurred_component, 0, 1)

        # Merge the components
        merged_image = cv2.add(sharp_component, blurred_component)
        merged_image = to_16bit(merged_image)
        merged_images.append(merged_image)

    del blurred_images, sobel_masks, sharp_component, blurred_component
    gc.collect()

    save_images(merged_images, './images/merged', name='merged')

    return merged_images


"""def unsharp_mask(images, model, strength):
    
    blurred_images = dncnn_images(model, images)

    save_images(blurred_images, './images/blurred', name = 'blurred')

    merged_images = [cv2.addWeighted(image, 1 + strength, blurred_image, -strength, 0) for image, blurred_image in zip(images, blurred_images)]
    
    save_images(merged_images, './images/merged', name = 'merged')
    
    del blurred_images
    gc.collect()
    return merged_images"""

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

def preprocess_images(images,           align       = True,
                      algo    = 'orb',  nfeatures   = 10000,
                      crop    = True,   margin      = 10,
                      unsharp = True,   strength    = 0.9,
                      grayscale   = True, calibrate = True):
    
    if calibrate:
        imgs = calibrate_images(images)
    else:
        imgs = images.copy()
    
    #save_images(imgs, "./images/calibrated", name = 'calibrated')

    #gamma_imgs    = [adjust_gamma(image, gamma=0.7) for image in contrast_imgs]

    if align:
        # Align the images
        imgs = align_images(imgs, algo=algo, nfeatures=nfeatures)

    if crop:
        # Crop the images to the center
        imgs = crop_to_center(imgs, margin=margin)

    if unsharp:
        # Apply unsharp mask to the images
        imgs = unsharp_mask(imgs, model_init(), strength)

    if grayscale:
        imgs = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in imgs]

    return imgs