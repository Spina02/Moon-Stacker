import cv2
import numpy as np
from image import to_8bit, to_float32
from config import DEBUG
from utils import progress
from metrics import get_min_brisque
import skimage

def enhance_contrast(image, clip_limit, tile_grid_size):
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

    if shape < 3 and len(enhanced_image.shape) == 3:
        enhanced_image = cv2.cvtColor(to_float32(enhanced_image), cv2.COLOR_RGB2GRAY)

    return enhanced_image

# ----------------- Preprocessing ------------------
def enhance(image, clip_limit = 0.8, tile_grid_size = (3,3)):
    if len(image.shape) == 3:
        corrected_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        corrected_image = image.copy()
    corrected_image = enhance_contrast(corrected_image, clip_limit = clip_limit, tile_grid_size = tile_grid_size)
    _, corrected_image = cv2.threshold(corrected_image, 0.05, 255, cv2.THRESH_TOZERO)
    return corrected_image

# -------------------- Aligning --------------------

def align_image(image, ref_image, aligner, matcher):
    # Initialize variables for the alignment process
    aligned_image = enhance(image)
    ref_image_pre = enhance(ref_image)

    # Find keypoints and descriptors for both images
    ref_kp, ref_des = aligner.detectAndCompute(to_8bit(ref_image_pre), None)
    kp, des = aligner.detectAndCompute(to_8bit(aligned_image), None)

    if des is None or ref_des is None:
        print(f'\nDescriptors are None.\n')

    # Match the descriptors
    matches = matcher.knnMatch(ref_des, des, k=2)
    # Apply ratio test to filter good matches
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(matches) < 4:
        print(f'\nNot enough matches found: {len(matches)} matches\n')

    # Compute the homography
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 10.0, maxIters=3000, confidence=0.995)

    if H is None or not np.linalg.det(H):
        print(f'\nInvalid homography\n')

    # Warp the aligned image at the current level
    #aligned_image = cv2.warpPerspective(aligned_image, H, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)

    # Warp the original image using the final homography
    aligned_image = cv2.warpPerspective(image, H, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_image


def align_images(images, algo='orb', nfeatures=10000):
    # Select reference image
    print("selecting the reference image")
    ref_image = images[0]#get_min_brisque(images)

    # Choose the feature detection algorithm
    if algo == 'orb':
        aligner = cv2.ORB_create(nfeatures=nfeatures)
        norm = cv2.NORM_HAMMING

    elif algo == 'sift':
        aligner = cv2.SIFT_create(nfeatures=nfeatures, sigma=1.6)
        norm = cv2.NORM_L2
        
    elif algo == 'surf':
        aligner = cv2.xfeatures2d.SURF_create(400)
        norm = cv2.NORM_L2

    # Create a matcher object
    matcher = cv2.BFMatcher.create(norm)

    aligned_images = []
  
    print("starting alignment")
    # Align each image to the reference image using pyramid alignment
    for idx, image in enumerate(images):
        aligned_image = align_image(image, ref_image, aligner, matcher)
        if aligned_image is not None:
            aligned_images.append(aligned_image)
        if DEBUG: progress(idx + 1, len(images), 'images aligned')

    return aligned_images