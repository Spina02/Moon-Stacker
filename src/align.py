import cv2
import numpy as np
from image import to_8bit
from enhancement import enhance_contrast, soft_threshold

# ---------------- Pre-processing -----------------

def pre_align_enhance(image, clip_limit = 0.8, tile_grid_size = (3,3), thr = 0.05):
    # Convert image to grayscale
    corrected_image = image.copy()
    if len(image.shape) == 3:
        corrected_image = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2GRAY)
    # Enhance contrast using CLACHE
    corrected_image = enhance_contrast(corrected_image, clip_limit, tile_grid_size)
    # Remove background using a threshold
    corrected_image = soft_threshold(corrected_image, thr)
    return corrected_image

# -------------------- Aligning --------------------

def align_image(image, ref_kp, ref_des, detector, matcher):
    # Initialize variables for the alignment process
    aligned_image = pre_align_enhance(image)

    # Find keypoints and descriptors for both images
    kp, des = detector.detectAndCompute(to_8bit(aligned_image), None)

    if des is None or ref_des is None:
        print(f'\nDescriptors are None.\n')

    # Match the descriptors
    matches = matcher.knnMatch(ref_des, des, k=2)
    # Apply ratio test to filter good matches
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(matches) < 4:
        print(f'\nNot enough matches found: {len(matches)} matches\n')

    # Compute the homography
    shape = image.shape[1], image.shape[0]
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 10.0, maxIters=3000, confidence=0.995)

    if H is None or not np.linalg.det(H):
        print(f'\nInvalid homography\n')
        
    # Warp the original image using the final homography
    aligned_image = cv2.warpPerspective(image, H, shape, flags=cv2.LANCZOS4)

    return aligned_image