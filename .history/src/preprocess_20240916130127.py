import cv2
import numpy as np
from config import DEBUG
import config
from utils import progress
from image import to_8bit
from models import model_init, unsharp_mask, dncnn_images
from image import save_images
import gc
from calibration import calibrate_images

# ------------------ Enhancing -----------------
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# ------------------ Aligning ------------------
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

def equalize_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:  # Color image

        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def denoise_images(images, algo):
    if algo == 'gaussian':
        # Apply Gaussian blur to the image
        return [cv2.GaussianBlur(image, (5, 5), 0) for image in images]
    if algo == 'NLMeans':
        # Apply fast non-local means denoising to the image
        return [cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21) for image in images]
    if algo == 'DnCnn':
        return dncnn_images(model_init(), images)

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

# ------------------ Cropping ------------------
def crop_to_center(images, margin=10):
    cropped_images = []

    # Process the first image to get the cropping parameters
    first_image = to_8bit(images[0])
    if len(first_image.shape) == 3:
        gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
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

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127 + 1  # Alpha è il fattore di contrasto
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# --------------- Preprocessing ----------------
def preprocess_images(images,           align       = True,
                      algo    = 'orb',  nfeatures   = 10000,
                      crop    = True,   margin      = 10,
                      unsharp = True,   strength    = 0.9,
                      grayscale   = True, calibrate = True):
    imgs 
    if calibrate:
        imgs = calibrate_images(images)

    #save_images(imgs, "./images/calibrated", name = 'calibrated')


    #contrast_imgs = [adjust_brightness_contrast(image, brightness=0, contrast=0) for image in imgs]
    #gamma_imgs    = [adjust_gamma(image, gamma=0.7) for image in contrast_imgs]

    #imgs = gamma_imgs
    
    #del contrast_imgs, gamma_imgs
    #gc.collect()

    if align:
        # Align the images
        imgs = align_images(imgs, algo=algo, nfeatures=nfeatures)

    if crop:
        # Crop the images to the center
        imgs = crop_to_center(imgs, margin=margin)

    if unsharp:
        # Apply unsharp mask to the images
        imgs = unsharp_mask(imgs, model_init(), strength)

    if grayscale:
        imgs = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in imgs]

    return imgs