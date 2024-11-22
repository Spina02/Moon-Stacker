from stacking import weighted_average_stack, median_stack, sigma_clipping
from metrics import calculate_metrics
from calibration import calibrate_single_image, calculate_masters
from enhancement import unsharp_mask, crop_to_center, gradient_mask_denoise_unsharp, shades_of_gray, soft_threshold, enhance_contrast
from align import pre_align_enhance
import config
from image import save_image, display_image, to_8bit, to_16bit
from align import align_image
import cv2
from utils import progress
from denoise import model_init

def calibrate_images(images, master_bias=None, master_dark=None, master_flat=None):
    master_bias, master_dark, master_flat = calculate_masters(master_bias, master_dark, master_flat)
    print("Started image calibration")
    
    calibrated_images = []

    # Se config.DEBUG è attivo, mostrare il progresso (non consigliato con multiprocessing per semplicità)
    for idx, image in enumerate(images):
        calibrated_images.append(calibrate_single_image(image, master_bias, master_dark, master_flat))
        if config.DEBUG:
            progress(idx, len(images), 'images calibrated')

    return calibrated_images

def align_images(images, algo='orb', nfeatures=5000, margin = 10):
    if len(images) > 1:
        # Choose the feature detection algorithm
        if algo == 'orb':
            detector = cv2.ORB_create(nfeatures=nfeatures)
            norm = cv2.NORM_HAMMING

        elif algo == 'sift':
            detector = cv2.SIFT_create(nfeatures=nfeatures)
            norm = cv2.NORM_L2
            
        elif algo == 'surf':
            detector = cv2.xfeatures2d.SURF_create()
            norm = cv2.NORM_L2

        # Create a matcher object
        matcher = cv2.BFMatcher.create(norm)

        if config.DEBUG: print("selecting the reference image")
        ref_image = images[0]
        aligned_images = [ref_image]
        enhanced_ref = pre_align_enhance(ref_image)
        ref_kp, ref_des = detector.detectAndCompute(to_8bit(enhanced_ref), None)
    
        if config.DEBUG: print("starting alignment")
        # Align each image to the reference image using pyramid alignment
        for idx, image in enumerate(images[1:]):
            aligned_image = align_image(image, ref_kp, ref_des, detector, matcher)
            if aligned_image is not None:
                aligned_images.append(aligned_image)
            if config.DEBUG: progress(idx+1, len(images), 'images aligned')
    else:
        aligned_images = images.copy()
    
    aligned_images = crop_to_center(aligned_images, margin=margin)

    return aligned_images

def custom_unsharp_mask(images, gradient_strength=1.5, gradient_threshold=0.005, denoise_strength=0.75, denoising_method="dncnn"):
    model = model_init()
    return gradient_mask_denoise_unsharp(images, model, strength=gradient_strength, threshold=gradient_threshold, denoise_strength = denoise_strength, denoising_method = denoising_method)

def stack_images(images, stacking_alg='weighted average', average_alg='sharpness'):
    if stacking_alg == 'weighted average':
        return weighted_average_stack(images, method=average_alg)
    elif stacking_alg == 'median':
        return median_stack(images)
    elif stacking_alg == 'sigma clipping':
        return sigma_clipping(images)
    else:
        raise ValueError(f"Algoritmo di stacking sconosciuto: {stacking_alg}")

def process_images(images=None, params={}, aligned=None, save=True, evaluate=True, denoising_method='dncnn'):
    gradient_strength = params.get('gradient_strength', 1.3)
    gradient_threshold = params.get('gradient_threshold', 0.008)
    denoise_strength = params.get('denoise_strength', 1.2)
    stacking_alg = params.get('stacking_alg', 'weighted average')
    average_alg = params.get('average_alg', 'sharpness')
    unsharp_strength = params.get('unsharp_strength', 2.35)
    tile_size = params.get('tile_size', (19, 19))
    clip_limit = params.get('clip_limit', 0.8)

    if config.DEBUG: print(f"Processing images with parameters: gradient_strength={gradient_strength}, gradient_threshold={gradient_threshold}, denoise_strength={denoise_strength}, stacking_alg={stacking_alg}, average_alg={average_alg}, unsharp_strength={unsharp_strength}, tile_size={tile_size}, clip_limit={clip_limit}, denoising_method={denoising_method}")

    if aligned is None:
        aligned = align_images(images)

    # Denoising
    denoised = custom_unsharp_mask(aligned, gradient_strength=gradient_strength, gradient_threshold=gradient_threshold, denoise_strength=denoise_strength, denoising_method = denoising_method)
    
    enhanced = []
    for image in denoised:
        no_bg = soft_threshold(image, 0.05, 50)
        unsharped = unsharp_mask(no_bg, unsharp_strength)
        enhanced.append(unsharped)

    # Stacking
    stacked_image = stack_images(enhanced, stacking_alg=stacking_alg, average_alg=average_alg)

    # Enhancing: apply traditional unsharp mask and contrast enhancement
    contrasted = enhance_contrast(stacked_image, clip_limit, tile_size)
    #unsharped = unsharp_mask(contrasted, unsharp_strength)
    final_image = shades_of_gray(unsharped)

    name = f"{denoising_method}_ush{unsharp_strength}_ker{tile_size}_clip{clip_limit}_avg{average_alg}"
    
    if save:
        save_image(final_image, name, config.output_folder)
    if config.COLAB:
        display_image(final_image, name)

    if evaluate: calculate_metrics(final_image, name, config.metrics)

    return final_image