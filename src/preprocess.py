from weakref import ref
import cv2
import numpy as np
from const import *
from debug import progress
#import torch
#from torch import nn
from image import to_8bit
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

    # find the keypoints and descriptors with ORB using the 8-bit image
    kp, des = aligner.detectAndCompute(to_8bit(image), None)

    if des is None or ref_des is None:
        print('\nDescriptors are None.\n')
        return None
    
    # Match the descriptors
    matches = matcher.knnMatch(ref_des, des, k = 1)

    matches = [m[0] for m in matches if len(m) == 1]

    if len(matches) < 4:
        print(f'\nNot enaugh matches found: {len(matches)} matches\n')
        return None

    # Compute the homography
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt     for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 10.0)

    if H is None or H.shape != (3, 3):
        print(f'\nImage could not find a valid homography\n')
        return None

    # Warp the original image (16 bit)
    aligned_img = cv2.warpPerspective(image, H, (ref_image.shape[1], ref_image.shape[0]))

    return aligned_img

def align_images(images, algo='orb', nfeatures=500):
    ref_image = images[0]

    if algo == 'orb':
        ref_kp, ref_des, aligner = orb(to_8bit(ref_image), nfeatures = nfeatures)
        matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True)
    elif algo == 'sift':
        ref_kp, ref_des, aligner = sift(to_8bit(ref_image), nfeatures = nfeatures)
        matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    elif algo == 'surf':
        ref_kp, ref_des, aligner = surf(to_8bit(ref_image))
        matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    aligned_images = [ref_image]

    for image in images[1:]:
        aligned_image = align_image(image, ref_kp, ref_des, ref_image, aligner, matcher)
        if aligned_image is not None:
            aligned_images.append(aligned_image)
        progress(len(aligned_images), len(images), 'images aligned')

    return aligned_images

# ------------------ Cropping ------------------

def crop_to_center(images, margin=10):
    cropped_images = []

    # Process the first image to get the cropping parameters
    first_image = to_8bit(images[0])
    if len(first_image.shape) == 3:
        gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
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
def preprocess_images(images, algo='orb', nfeatures=500, margin = 10, sharpen = True, align = True, crop = False, sigma = 1.5, strength = 4):
    imgs = images.copy()
    if align:
        # Apply Gaussian blur to the images
        imgs = [cv2.GaussianBlur(image, (5, 5), 0) for image in imgs]
    
        # Denoise the images using DnCNN model
        #blurred_images = [denoise_image(image) for image in images]

        # Align the images
        imgs = align_images(imgs, algo=algo, nfeatures=nfeatures)
    
    if sharpen:
        # Apply unsharp mask to the images
        imgs = [unsharp_mask(image, sigma = sigma, strength = strength) for image in imgs]
    
    if crop:
        # Crop the images to the center
        imgs = crop_to_center(imgs, margin=margin)

    return imgs