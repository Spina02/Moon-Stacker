import cv2
import numpy as np
from const import *
from debug import progress
import time
import matplotlib.pyplot as plt

# ------------------ Enhancing -----------------
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def preprocess_images(images):
    return [sharpen_image(image) for image in images]

# ------------------ Aligning ------------------
def orb(image, nfeatures = 500):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures = nfeatures, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = cv2.ORB_HARRIS_SCORE, patchSize = 31, fastThreshold = 20)
    kp, des = orb.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des

def surf(image, hessian_threshold = 400):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize the SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    # Detect key points and descriptors
    kp, des = surf.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des

def sift(image, nfeatures = 500):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(nfeatures = nfeatures, sigma=1.6)
    # Detect key points and descriptors
    kp, des = sift.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des

def align_image(image, ref_kp, ref_des, ref_image, algo, i, total, nfeatures):
    
    ref_kp_coords = [kp.pt for kp in ref_kp]
    
    if algo == 'orb':
        kp, des = orb(image)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif algo == 'sift':
        kp, des = sift(image)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif algo == 'surf':
        kp, des = surf(image)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Ensure descriptors are not None
    if des is None or ref_des is None:
        progress(i+2, total, 'images aligned')
        print(f'Image {i+2} has no descriptors')
        return None
    
    # Match the descriptors
    matches = matcher.match(ref_des, des)
    matches = sorted(matches, key = lambda x:x.distance)
    ref_pts = np.float32([ref_kp_coords[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0)

    aligned_img = cv2.warpPerspective(image, H, ref_image.shape[:2])

    progress(i+2, total, 'images aligned')
    return aligned_img

def align_images(images, algo='orb', nfeatures=500):
    ref_image = images[0]

    if algo == 'orb':
        ref_kp, ref_des = orb(ref_image, nfeatures)
    elif algo == 'sift':
        ref_kp, ref_des = sift(ref_image, nfeatures)
    elif algo == 'surf':
        ref_kp, ref_des = surf(ref_image)


    start_time = time.time()
    aligned_images = [align_image(image, ref_kp, ref_des, ref_image, algo, i, len(images), nfeatures) for i, image in enumerate(images[1:])]
    end_time = time.time()

    aligned_images = [ref_image] + [img for img in aligned_images if img is not None]
    
    print(f'Single process alignment took {end_time - start_time:.2f} seconds')
    return aligned_images

# ------------------ Cropping ------------------
def crop_to_center(images, margin=10):
    cropped_images = []

    # Process the first image to get the cropping parameters
    first_image = images[0]
    gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)

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

