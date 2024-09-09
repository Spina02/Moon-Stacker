import cv2
import numpy as np
from const import *
from debug import progress
import torch
from torch import nn
from models import *

# ------------------ Enhancing -----------------
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def unsharp_mask(image, sigma=1.5, strength=4):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

#def denoise_image(image, algo = 'DnCnn'):
#    if algo == 'DnCnn':
#        model = DnCNN()
#        model.load_state_dict(torch.load(DNCNN_MODEL_PATH))
#        model.eval()
#
#        # Convert the image to a tensor
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        # Normalize the image
#        image = image.astype(np.float32) / 255.0
#        # Add batch and channel dimensions
#        image = np.expand_dims(image, axis=(0, 1)) 
#        image = torch.from_numpy(image).unsqueeze(0)
#
#        # Denoise the image
#        with torch.no_grad():
#            denoised = model(image)
#
#        denoised = denoised.squeeze().cpu().numpy()
#        denoised = (denoised * 255).astype(np.uint8)
#        denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
#        return denoised

# ------------------ Aligning ------------------
def orb(image, nfeatures = 500):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures = nfeatures)
    kp, des = orb.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des, orb

def surf(image, hessian_threshold = 400):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Initialize the SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    # Detect key points and descriptors
    kp, des = surf.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des, surf

def sift(image, nfeatures = 500):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(nfeatures = nfeatures, sigma=1.6)
    # Detect key points and descriptors
    kp, des = sift.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des, sift

def align_image(image, ref_kp, ref_des, ref_image, aligner, matcher):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = aligner.detectAndCompute(image, None)

    if des is None or ref_des is None:
        print("Descriptors are None.")
        return None
    
    # Match the descriptors
    matches = matcher.knnMatch(ref_des, des, k = 1)
    #matches = sorted(matches, key=lambda x: x.distance)

    matches = [m[0] for m in matches if len(m) == 1]

    if len(matches) < 4:
        print(f"Not enaugh matches found: {len(matches)} matches.")
        return None

    # Homography
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt     for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 10.0)

    if H is None or H.shape != (3, 3):
        print(f'Image could not find a valid homography')
        return None

    aligned_img = cv2.warpPerspective(image, H, (ref_image.shape[1], ref_image.shape[0]))

    return aligned_img

def align_images(images, algo='orb', nfeatures=500):
    ref_image = images[0]

    if algo == 'orb':
        ref_kp, ref_des, aligner = orb(ref_image, nfeatures = nfeatures)
        matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True)
    elif algo == 'sift':
        ref_kp, ref_des, aligner = sift(ref_image, nfeatures = nfeatures)
        matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    elif algo == 'surf':
        ref_kp, ref_des, aligner = surf(ref_image)
        matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    aligned_images = []

    for i, image in enumerate(images[1:]):
        aligned_image = align_image(image, ref_kp, ref_des, ref_image, aligner, matcher)
        if aligned_image is not None:
            aligned_images.append(aligned_image)
        progress(i+1, len(images), 'images aligned')

    aligned_images = [images[0]] + aligned_images
    return aligned_images

# ------------------ Cropping ------------------
def crop_to_center(images, margin=10):
    cropped_images = []

    # Process the first image to get the cropping parameters
    first_image = images[0]
    if len(first_image.shape) == 3:
        gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = first_image

    # Binary thresholding
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the moon
    contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the contour
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
def preprocess_images(images, algo='orb', nfeatures=500):
    # Apply Gaussian blur to the images
    blurred_images = [cv2.GaussianBlur(image, (5, 5), 0) for image in images]
    
    # Denoise the images using DnCNN model
    #blurred_images = [denoise_image(image) for image in images]

    # Align the images
    aligned_images = align_images(blurred_images, algo=algo, nfeatures=nfeatures)
    
    # Apply unsharp mask to the images
    sharpened_images = [unsharp_mask(image) for image in aligned_images]
    
    # Crop the images to the center
    cropped_images = crop_to_center(sharpened_images)

    return cropped_images