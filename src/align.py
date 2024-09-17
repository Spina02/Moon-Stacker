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

# -------------------- Aligning --------------------

def align_image(image, ref_kp, ref_des, ref_image, aligner, matcher):
    # find the keypoints and descriptors with the chosen algorithm
    kp, des = aligner.detectAndCompute(to_8bit(image), None)

    if des is None or ref_des is None:
        print('\nDescriptors are None.\n')
        return None
    
    # Match the descriptors
    if config.IS_MOON:
        matches = matcher.knnMatch(ref_des, des, k=2)
        matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        matches = matcher.knnMatch(ref_des, des, k=1)
        matches = [m[0] for m in matches if len(m) == 1]

    if len(matches) < 4:
        print(f'\nNot enough matches found: {len(matches)} matches\n')
        return None
    else:
        if DEBUG > 1: print(f'\n{len(matches)} matches found\n')

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
    aligned_img_homography = cv2.warpPerspective(image, H, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)

    # Detect keypoints and descriptors again on the homography aligned image
    kp_affine, des_affine = aligner.detectAndCompute(to_8bit(aligned_img_homography), None)

    # Match the descriptors again
    matches_affine = matcher.match(ref_des, des_affine)

    if len(matches_affine) < 4:
        print(f'\nNot enough matches found after homography transformation: {len(matches_affine)} matches\n')
        return aligned_img_homography  # Return the homography aligned image if affine fails

    matches_affine = sorted(matches_affine, key=lambda x: x.distance)
    ref_pts_affine = np.float32([ref_kp[m.queryIdx].pt for m in matches_affine]).reshape(-1, 1, 2)
    img_pts_affine = np.float32([kp_affine[m.trainIdx].pt for m in matches_affine]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(img_pts_affine, ref_pts_affine)

    del kp_affine, des_affine, matches_affine, ref_pts_affine, img_pts_affine
    gc.collect()

    if M is None or M.shape != (2, 3):
        print(f'\nImage could not find a valid affine transformation after homography\n')
        return aligned_img_homography  # Return the homography aligned image if affine fails

    # Warp the original image (16 bit) using affine transformation
    aligned_img = cv2.warpAffine(aligned_img_homography, M, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)

    del H, M, aligned_img_homography
    gc.collect()

    return aligned_img

def align_images(images, algo='orb', nfeatures=500):
    #images = [equalize_histogram(image) for image in images]

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

    if config.IS_MOON:
        matcher = cv2.BFMatcher.create(norm)
    else:
        matcher = cv2.BFMatcher.create(norm, crossCheck=True)

    aligned_images = [ref_image]

    for image in images[1:]:
        aligned_image = align_image(image, ref_kp, ref_des, ref_image, aligner, matcher)
        if aligned_image is not None:
            aligned_images.append(aligned_image)
        if DEBUG: progress(len(aligned_images), len(images), 'images aligned')

    return aligned_images