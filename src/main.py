import config
from image import *
from preprocessing import preprocess_images, unsharp_mask, crop_to_center
from stacking import weighted_average_stack, median_stack, sigma_clipping
from metrics import calculate_brisque, calculate_ssim, combined_score, get_min_brisque, calculate_metric
from calibration import calculate_masters, calibrate_images
from align import enhance_contrast
from grid_search import grid_search
import os

best_params = {
    "gradient_strength": 1.8,
    "gradient_threshold": 0.005,
    "denoise_strength": 0.75,
    "stacking_alg": "weighted average",
    "average_alg": "sharpness",
    "strength": 1.25,
    "ker": (11, 11),
    "limit": 0.7
}

metrics = ['niqe', 'piqe', 'liqe']#, 'nima', 'brisque_matlab']
        
def process_images(images, params = best_params, aligned = None, save = True):
    gradient_strength = params.get('gradient_strength', 1.5)
    gradient_threshold = params.get('gradient_threshold', 0.005)
    denoise_strength = params.get('denoise_strength', 0.75)
    stacking_alg = params.get('stacking_alg', 'weighted average')
    average_alg = params.get('average_alg', 'brisque')
    unsharp_strength = params.get('strength', 1.25)
    kernel_size = params.get('ker', (11, 11))
    clip_limit = params.get('limit', 0.75)

    if aligned is None:
        aligned = preprocess_images(images, algo = 'orb', grayscale = False, unsharp = False)
    
    denoised = preprocess_images(aligned, align=False, crop=False, gradient_strength=gradient_strength, gradient_threshold=gradient_threshold, denoise_strength = denoise_strength)

    # Stacking
    if stacking_alg == 'weighted average':
        stacked_image = weighted_average_stack(denoised, method=average_alg)
    elif stacking_alg == 'median':
        stacked_image = median_stack(denoised)
    elif stacking_alg == 'sigma clipping':
        stacked_image = sigma_clipping(denoised)
    else:
        raise ValueError(f"Algoritmo di stacking sconosciuto: {stacking_alg}")

    # Enhancing: apply unsharp mask and contrast enhancement
    enhanced_image = unsharp_mask(stacked_image, unsharp_strength)
    final_image = enhance_contrast(enhanced_image, clip_limit=clip_limit, tile_grid_size=kernel_size)

    ref, brisque_score = get_min_brisque(aligned)
    ssim_score = calculate_ssim(ref, final_image)
    score = combined_score(brisque_score, ssim_score)

    new_name = f"processed_str{gradient_strength}_thr{gradient_threshold}_dstr{denoise_strength}_ush{unsharp_strength}_ker{kernel_size}_clip{clip_limit}_avg{average_alg}"
    
    if save:
        save_image(final_image, new_name, config.output_folder)
    if config.COLAB:
        display_image(final_image, brisque_score, ssim_score, score, new_name)
    else:
        print_score(brisque_score, ssim_score, score, new_name)

    return {
        'BRISQUE Score': brisque_score,
        'SSIM Score': ssim_score,
        'Combined Score': score,
        'Processed Image': final_image
    }
    
def main():
    bias = read_image('./images/masters/bias.tif') if os.path.exists('./images/masters/bias.tif') else None
    dark = read_image('./images/masters/dark.tif') if os.path.exists('./images/masters/dark.tif') else None
    flat = read_image('./images/masters/flat.tif') if os.path.exists('./images/masters/flat.tif') else None
    bias, dark, flat = calculate_masters(bias, dark, flat)
    
    images = read_images(config.input_folder)

    image_0 = crop_to_center([images[0]])[0]
    if config.COLAB:
        display_image(image_0, "image 0")
        for metric in metrics:
            metric_score = calculate_metric(image_0, metric)
            print(f'{metric} score: {metric_score:.4f}')

            # Visualize the image if running on Google Colab

    images = calibrate_images(images, bias, dark, flat)

    calibrated_0 = crop_to_center([images[0]])[0]

    if config.COLAB:
        display_image(calibrated_0, "calibrated")
        for metric in metrics:
            metric_score = calculate_metric(calibrated_0, metric)
            print(f'{metric} score: {metric_score:.4f}')

            # Visualize the image if running on Google Colab

    params, aligned = grid_search(images)
    process_images(None, params = params, aligned = aligned)

if __name__ == '__main__':
    main()