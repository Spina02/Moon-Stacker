import cv2
from matplotlib import scale
import numpy as np
from const import *
from debug import progress
from image import save_image

# ORB algorithm
def orb(image, n_features = 500):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures = n_features, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = cv2.ORB_HARRIS_SCORE, patchSize = 31, fastThreshold = 20)
    # Detect key points and descriptors
    kp, des = orb.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des

# SURF algorithm
def surf(image, hessian_threshold = 400):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize the SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    # Detect key points and descriptors
    kp, des = surf.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des

def sift(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    # Detect key points and descriptors
    kp, des = sift.detectAndCompute(gray, None)
    # Draw the key points on the image
    return kp, des

#align images using ORB
def align_images(images, algo = 'orb', n_features = 500):
    ref_image = images[0]
    aligned_images = [ref_image]
   
    # Initialize the feature detector and descriptor extractor
    if algo == 'orb':
        ref_kp, ref_des = orb(ref_image)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif algo == 'sift':
        ref_kp, ref_des = sift(ref_image)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif algo == 'surf':
        ref_kp, ref_des = surf(ref_image)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    for image in images[1:]:
        if algo == 'orb':
            kp, des = orb(image)
        elif algo == 'sift':
            kp, des = sift(image)
        elif algo == 'surf':
            kp, des = surf(image)

        # Ensure descriptors are not None
        if des is None or ref_des is None:
            continue
        
        # Match the descriptors
        matches = matcher.match(ref_des, des)

        # Sort the matches based on distance
        matches = sorted(matches, key = lambda x:x.distance)

        # Extract location of good matches
        ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        img_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0)

        # Use the homography matrix to align the images
        height, width = ref_image.shape[:2]
        aligned_img = cv2.warpPerspective(image, H, (width, height))
        aligned_images.append(aligned_img)

        if DEBUG: progress(len(aligned_images), len(images), 'images aligned')

    # Implement image alignment here
    return aligned_images

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
    for image in images[1:]:
        cropped_image = image[start_y:end_y, start_x:end_x]
        cropped_images.append(cropped_image)

        if DEBUG: progress(len(cropped_images), len(images) -1, 'images cropped')

    return cropped_images