from stacking import weighted_average_stack, median_stack, sigma_clipping
from metrics import calculate_metrics, get_best_image
from calibration import calibrate_single_image, calculate_masters
from preprocessing import unsharp_mask, crop_to_center, gradient_mask_denoise_unsharp
from align import enhance_contrast, enhance
import config
from config import DEBUG
from image import save_image, display_image, to_8bit
from align import align_image
import cv2
from utils import progress
from denoise import DnCNN, model_init

def calibrate_images(images, master_bias=None, master_dark=None, master_flat=None):
    master_bias, master_dark, master_flat = calculate_masters(master_bias, master_dark, master_flat)
    print("Started image calibration")
    
    calibrated_images = []

    # Se DEBUG è attivo, mostrare il progresso (non consigliato con multiprocessing per semplicità)
    for idx, image in enumerate(images):
        calibrated_images.append(calibrate_single_image(image, master_bias, master_dark, master_flat))
        if DEBUG:
            progress(idx, len(images), 'images calibrated')

    return calibrated_images

def align_images(images, algo='orb', nfeatures=10000, sigma = 1.6, h_thr = 400, margin = 10):
    if len(images) > 1:
        # Choose the feature detection algorithm
        if algo == 'orb':
            aligner = cv2.ORB_create(nfeatures=nfeatures)
            norm = cv2.NORM_HAMMING

        elif algo == 'sift':
            aligner = cv2.SIFT_create(nfeatures=nfeatures, sigma=sigma)
            norm = cv2.NORM_L2
            
        elif algo == 'surf':
            aligner = cv2.xfeatures2d.SURF_create(h_thr)
            norm = cv2.NORM_L2

        # Create a matcher object
        matcher = cv2.BFMatcher.create(norm)

        if DEBUG: print("selecting the reference image")
        ref_image = images[0]
        aligned_images = [ref_image]
        enhanced_ref = enhance(ref_image)
        ref_kp, ref_des = aligner.detectAndCompute(to_8bit(enhanced_ref), None)
        ref_shape = (ref_image.shape[1], ref_image.shape[0])
    
        if DEBUG: print("starting alignment")
        # Align each image to the reference image using pyramid alignment
        for idx, image in enumerate(images[1:]):
            aligned_image = align_image(image, enhanced_ref, ref_kp, ref_des, ref_shape, aligner, matcher)
            if aligned_image is not None:
                aligned_images.append(aligned_image)
            if DEBUG: progress(idx+1, len(images), 'images aligned')
    else:
        aligned_images = images.copy()
    
    aligned_images = crop_to_center(aligned_images, margin=margin)

    return aligned_images

def dncnn_unsharp_mask(images, gradient_strength=1.5, gradient_threshold=0.005, denoise_strength=0.75):
    model = model_init()
    return gradient_mask_denoise_unsharp(images, model, strength=gradient_strength, threshold=gradient_threshold, denoise_strength = denoise_strength)

def stack_images(images, stacking_alg='weighted average', average_alg='sharpness'):
    if stacking_alg == 'weighted average':
        return weighted_average_stack(images, method=average_alg)
    elif stacking_alg == 'median':
        return median_stack(images)
    elif stacking_alg == 'sigma clipping':
        return sigma_clipping(images)
    else:
        raise ValueError(f"Algoritmo di stacking sconosciuto: {stacking_alg}")

def process_images(images, params = {}, aligned = None, save = True, evaluate = True, denoising_method = 'dncnn'):
    gradient_strength = params.get('gradient_strength', 1.3)
    gradient_threshold = params.get('gradient_threshold', 0.008)
    denoise_strength = params.get('denoise_strength', 1.2)
    stacking_alg = params.get('stacking_alg', 'weighted average')
    average_alg = params.get('average_alg', 'sharpness')
    unsharp_strength = params.get('unsharp_strength', 2.35)
    kernel_size = params.get('kernel_size', (19, 19))
    clip_limit = params.get('clip_limit', 0.8)

    print(f"Processing images with parameters: gradient_strength={gradient_strength}, gradient_threshold={gradient_threshold}, denoise_strength={denoise_strength}, stacking_alg={stacking_alg}, average_alg={average_alg}, unsharp_strength={unsharp_strength}, kernel_size={kernel_size}, clip_limit={clip_limit}, denoising_method={denoising_method}")

    if aligned is None:
        if images is None:
            raise ValueError("Devi fornire almeno 'images' o 'aligned'")
        aligned = align_images(images)
    
    # Denoising
    if denoising_method == 'dncnn':
        denoised = dncnn_unsharp_mask(aligned, gradient_strength=gradient_strength, gradient_threshold=gradient_threshold, denoise_strength=denoise_strength)
    elif denoising_method == 'gaussian':
        denoised = [cv2.GaussianBlur(img, (5, 5), 5) for img in aligned]
    elif denoising_method == 'bilateral':
        denoised = [cv2.bilateralFilter(img, 9, 150, 150) for img in aligned]
    elif denoising_method == 'median':
        denoised = [cv2.medianBlur(img, 5) for img in aligned]
    else:
        raise ValueError(f"Unknown denoising method: {denoising_method}")

    # Stacking
    stacked_image = stack_images(denoised, stacking_alg=stacking_alg, average_alg=average_alg)

    # Enhancing: applica unsharp mask e contrast enhancement
    enhanced_image = unsharp_mask(stacked_image, unsharp_strength)
    final_image = enhance_contrast(enhanced_image, clip_limit=clip_limit, tile_grid_size=kernel_size)

    name = f"{denoising_method}_ush{unsharp_strength}_ker{kernel_size}_clip{clip_limit}_avg{average_alg}"
    
    if save:
        save_image(final_image, name, config.output_folder)
    if config.COLAB:
        display_image(final_image, name)

    if evaluate:
        calculate_metrics(final_image, name, config.metrics)

    return final_image