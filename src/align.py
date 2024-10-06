import cv2
import numpy as np
import gc
from image import to_8bit
import config
from config import DEBUG
from utils import progress

# --------------- Feature Detection ---------------

def orb(image, nfeatures = 500):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures = nfeatures)
    kp, des = orb.detectAndCompute(gray, None)

    del gray
    gc.collect()

    return kp, des, orb

def surf(image, hessian_threshold = 400):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    # Initialize the SURF detector
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    # Detect key points and descriptors
    kp, des = surf.detectAndCompute(gray, None)

    del gray
    gc.collect()

    return kp, des, surf

def sift(image, nfeatures = 500):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(nfeatures = nfeatures, sigma=1.6)
    # Detect key points and descriptors
    kp, des = sift.detectAndCompute(gray, None)
    
    del gray
    gc.collect()

    return kp, des, sift

# ----------------- Preprocessing ------------------
def preprocess(image):
    corrected_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, corrected_image = cv2.threshold(corrected_image, 0.05, 255, cv2.THRESH_TOZERO)
    corrected_image = cv2.equalizeHist(to_8bit(corrected_image))
    return corrected_image

# -------------------- Aligning --------------------

def align_image(image, ref_kp, ref_des, ref_image, aligner, matcher):

    #preprocessed_image = preprocess(image)
    # find the keypoints and descriptors with the chosen algorithm
    kp, des = aligner.detectAndCompute(to_8bit(image), None)

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
    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 15.0)

    del kp, des, matches, ref_pts, img_pts
    gc.collect()

    if H is None or H.shape != (3, 3):
        print(f'\nImage could not find a valid homography\n')
        return None

    # Warp the original image (16 bit) using homography
    aligned_img = cv2.warpPerspective(image, H, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)

    del H, image, preprocessed_image
    gc.collect()

    return aligned_img

def align_images(images, algo='orb', nfeatures=10000):

    ref_image = images[0]

    if algo == 'orb':
        ref_kp, ref_des, aligner = orb(to_8bit(ref_image), nfeatures = nfeatures)
        norm = cv2.NORM_HAMMING

    elif algo == 'sift':
        ref_kp, ref_des, aligner = sift(to_8bit(ref_image), nfeatures = nfeatures)
        norm = cv2.NORM_L2
        
    elif algo == 'surf':
        ref_kp, ref_des, aligner = surf(to_8bit(ref_image))
        norm = cv2.NORM_L2

    matcher = cv2.BFMatcher.create(norm)

    aligned_images = [ref_image]
    if DEBUG: progress(len(aligned_images), len(images), 'images aligned')

    for image in images[1:]:
        aligned_image = align_image(image, ref_kp, ref_des, ref_image, aligner, matcher)
        if aligned_image is not None:
            aligned_images.append(aligned_image)
        if DEBUG: progress(len(aligned_images), len(images), 'images aligned')
        del aligned_image
        gc.collect()

    return aligned_images