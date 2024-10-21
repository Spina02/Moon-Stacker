import cv2
import numpy as np
from image import to_8bit, to_float32
from config import DEBUG
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

def align_image(image, ref_image, ref_kp, ref_des, shape, aligner, matcher):
    # Initialize variables for the alignment process
    aligned_image = enhance(image)

    # Find keypoints and descriptors for both images
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
        
    # Warp the original image using the final homography
    aligned_image = cv2.warpPerspective(image, H, shape, flags=cv2.INTER_CUBIC)

    return aligned_image


"""
def align_image(image, ref_image, ref_kp, ref_des, shape, aligner, matcher):
    # Enhance the input image (assuming 'enhance' is a pre-defined function)
    enhanced_image = enhance(image)

    # Find keypoints and descriptors for the enhanced image using the aligner (e.g., ORB)
    kp, des = aligner.detectAndCompute(to_8bit(enhanced_image), None)

    # Check if descriptors are valid
    if des is None or ref_des is None:
        print(f'\nDescriptors are None.\n')
        return None

    # Match the descriptors between the reference and current image
    matches = matcher.knnMatch(ref_des, des, k=2)
    # Apply ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Ensure there are enough good matches
    if len(good_matches) < 4:
        print(f'\nNot enough matches found: {len(good_matches)} matches\n')
        return None

    # Extract location of good matches
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography matrix using RANSAC algorithm
    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 10.0, maxIters=3000, confidence=0.995)

    # Check if homography is valid
    if H is None or not np.linalg.det(H):
        print(f'\nInvalid homography\n')
        return None
        
    # Warp the original image using the initial homography
    aligned_image = cv2.warpPerspective(image, H, shape, flags=cv2.INTER_CUBIC)

    # -------- Subpixel Alignment using ECC Algorithm --------

    # Convert the aligned image and reference image to grayscale
    aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    #ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    # Convert images to float32 and normalize to range [0, 1]
    #aligned_image = to_float32(aligned_gray)
    #ref_image = to_float32(ref_gray)

    # Define the motion model (using translation for subpixel adjustments)
    warp_mode = cv2.MOTION_TRANSLATION  # Options: MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY

    # Initialize the warp matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define criteria for the ECC algorithm (number of iterations and threshold)
    number_of_iterations = 500
    termination_eps = 1e-6
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm to compute the warp matrix for subpixel alignment
    try:
        cc, warp_matrix = cv2.findTransformECC(ref_image, aligned_gray, warp_matrix, warp_mode, criteria)
    except cv2.error as e:
        print('ECC algorithm failed:', e)
        # Return the initial aligned image if ECC fails
        return aligned_image

    # Apply the warp matrix to the aligned image to achieve subpixel alignment
    aligned_image_subpixel = cv2.warpAffine(aligned_image, warp_matrix, (shape[0], shape[1]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned_image_subpixel
"""