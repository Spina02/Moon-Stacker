import cv2
import numpy as np
from image import to_8bit
import config
from config import DEBUG
from utils import progress
import skimage

# --------------- Feature Detection ---------------

def orb(image, nfeatures=500):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(gray, None)

    return kp, des, orb

def surf(image, hessian_threshold=400):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Initialize the SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    # Detect key points and descriptors
    kp, des = surf.detectAndCompute(gray, None)

    return kp, des, surf

def sift(image, nfeatures=500):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(nfeatures=nfeatures, sigma=1.6)
    # Detect key points and descriptors
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des, sift

# ----------------- Preprocessing ------------------
def preprocess(image):
    corrected_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, corrected_image = cv2.threshold(corrected_image, 0.05, 255, cv2.THRESH_TOZERO)
    #corrected_image = cv2.equalizeHist(to_8bit(corrected_image))
    return corrected_image

def enhance_contrast(image, clip_limit=0.5, tile_grid_size=(2, 2)):
    shape = len(image.shape)
    if shape < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Convert the image to LAB color space using skimage
    lab = skimage.color.rgb2lab(image)

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
    enhanced_image = skimage.color.lab2rgb(lab)

    # Convert the image to 8-bit format for saving
    #enhanced_image = to_8bit(enhanced_image)

    if shape < 3:
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)

    return enhanced_image

def force_background_to_black(image, threshold_value=0.03):
    _, corrected_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_TOZERO)
    return corrected_image

# -------------------- Aligning --------------------

def align_image(image, ref_kp, ref_des, ref_image, aligner, matcher):
    # Preprocess the image
    image = enhance_contrast(image, clip_limit=0.8, tile_grid_size=(5, 5))
    image = force_background_to_black(image)
    
    # Find the keypoints and descriptors with the chosen algorithm
    kp, des = aligner.detectAndCompute(cv2.equalizeHist(to_8bit(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))), None)
    #kp, des = aligner.detectAndCompute(to_8bit(preprocessed_image), None)

    if des is None or ref_des is None:
        print('\nDescriptors are None.\n')
        return None
    
    # Match the descriptors
    matches = matcher.knnMatch(ref_des, des, k=2)
    # Apply ratio test to filter good matches
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(matches) < 4:
        print(f'\nNot enough matches found: {len(matches)} matches\n')
        return None

    # Compute the homography
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0, maxIters=1000, confidence=0.99)

    if H is None or H.shape != (3, 3):
        print(f'\nImage could not find a valid homography\n')
        return None

    # Warp the original image (16 bit) using homography
    aligned_img = cv2.warpPerspective(image, H, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_img

def align_images(images, algo='orb', nfeatures=10000):
    # Select reference image
    ref_image = images[0]

    # Choose the feature detection algorithm
    if algo == 'orb':
        ref_kp, ref_des, aligner = orb(to_8bit(ref_image), nfeatures=nfeatures)
        norm = cv2.NORM_HAMMING

    elif algo == 'sift':
        ref_kp, ref_des, aligner = sift(to_8bit(ref_image), nfeatures=nfeatures)
        norm = cv2.NORM_L2
        
    elif algo == 'surf':
        ref_kp, ref_des, aligner = surf(to_8bit(ref_image))
        norm = cv2.NORM_L2

    # Create a matcher object
    matcher = cv2.BFMatcher.create(norm)

    aligned_images = []
    if DEBUG: progress(0, len(images), 'images aligned')

    # Align each image to the reference image
    for idx, image in enumerate(images):
        aligned_image = align_image(image, ref_kp, ref_des, ref_image, aligner, matcher)
        if aligned_image is not None:
            aligned_images.append(aligned_image)
        if DEBUG: progress(idx + 1, len(images), 'images aligned')

    return aligned_images