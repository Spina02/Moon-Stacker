import config
from image import *
from preprocessing import preprocess_images, unsharp_mask
from stacking import weighted_average_stack, median_stack, sigma_clipping
from metrics import calculate_brisque, calculate_ssim, combined_score, get_min_brisque
from calibration import calculate_masters, calibrate_images
from align import enhance_contrast
from grid_search import grid_search
import os

best_params = {
    "gradient_strength": 1.75,
    "gradient_threshold": 0.005,
    "denoise_strength": 0.75,
    "stacking_alg": "weighted average",
    "average_alg": "brisque",
    "strength": 1.25,
    "ker": (11, 11),
    "limit": 0.75
}

"""
def image_stacking(images, features_alg = 'orb', calibrate = True, average_alg = 'sharpness', stacking_alg = 'median', n_features = 10000, grayscale = True):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False, crop = True, grayscale = grayscale, unsharp = False, calibrate=False)[0]
    save_images([image_0], 'original')

    preprocessed = preprocess_images(images, algo = 'orb', align = True, crop = True, grayscale = False, unsharp = False, calibrate = False)

    for stacking_alg in ['weighted average', 'median', 'sigma clipping']:
        
        unsharped = preprocess_images(preprocessed, align = False, crop = False, grayscale = grayscale, unsharp = True, calibrate = False)

        name = f'{features_alg}_{stacking_alg}' 
        # Stack the images
        if stacking_alg == 'weighted average':
            name += f'_{average_alg}'
            image = weighted_average_stack(unsharped, method=average_alg)
        elif stacking_alg == 'median':
            image = median_stack(unsharped)
        elif stacking_alg == 'sigma clipping':
            image = sigma_clipping(unsharped)

        # Save the image
        print(f'\nsaving {name}')
        save_image(image, name)

        print(image_0.dtype, image.dtype)
"""
        
def process_images(images, params = best_params, aligned = None, save = True):

    # Estrai i parametri dal dizionario
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
        stacked_image = weighted_average_stack([denoised], method=average_alg)
    elif stacking_alg == 'median':
        stacked_image = median_stack([denoised])
    elif stacking_alg == 'sigma clipping':
        stacked_image = sigma_clipping([denoised])
    else:
        raise ValueError(f"Algoritmo di stacking sconosciuto: {stacking_alg}")

    # Enhancing: applica unsharp mask e enhancement del contrasto
    enhanced_image = unsharp_mask(stacked_image, unsharp_strength)
    contrasted_image = enhance_contrast(enhanced_image, clip_limit=clip_limit, tile_grid_size=kernel_size)

    ref = get_min_brisque(aligned)
    # Calcola le metriche
    brisque_score = calculate_brisque(contrasted_image)
    ssim_score = calculate_ssim(ref, contrasted_image)
    score = combined_score(brisque_score, ssim_score)

    # Salva l'immagine elaborata se richiesto
    if save:
        new_name = f"processed_str{gradient_strength}_thr{gradient_threshold}_dstr{denoise_strength}_ush{unsharp_strength}_ker{kernel_size}_clip{clip_limit}_avg{average_alg}"
        save_image(contrasted_image, new_name, config.output_folder)

    # Ritorna le metriche e l'immagine elaborata
    return {
        'BRISQUE Score': brisque_score,
        'SSIM Score': ssim_score,
        'Combined Score': score,
        'Processed Image': contrasted_image
    }

""" 
def grid_search(images, save = False):
    print()
    image_0 = preprocess_images([images[0]], align=False, crop=True, grayscale=True, unsharp=False, calibrate=False)[0]
    if save: save_image(image_0, name='original')

    best_score = 0
    best_img = ''
    best_params = {}

    features_alg='orb'
    n_features=50000
    print(f'aligning images with {features_alg} and {n_features} features')
    preprocessed = preprocess_images(images, algo=features_alg, nfeatures=n_features, grayscale=False, unsharp=False)
    if save: save_image(preprocessed[0], 'aligned')

    ref = get_min_brisque(preprocessed)

    brisque = calculate_brisque(image_0)
    ssim = calculate_ssim(ref, image_0)
    score = combined_score(brisque, ssim)
    display_image(image_0, brisque, ssim, score, 'original')

    # Lista degli algoritmi di stacking
    stacking_algorithms =   ['weighted average']#, 'sigma clipping', 'median']
    average_algs =          ['brisque', 'sharpness']
    gradient_strengths =    [1.75, 1.8]
    gradient_thresholds =   [0.005, 0.006]
    denoise_strengths =     [0.75, 0.8]
    unsharp_strengths =     [1.2, 1.25]
    kernel_sizes =          [(11, 11), (13, 13)]
    clip_limits =           [0.7, 0.75]

    for gradient_strength in gradient_strengths:
        for gradient_threshold in gradient_thresholds:
            for denoise_strength in denoise_strengths:
                 
                # Preprocess the images with the selected sharpening method
                denoised = preprocess_images(preprocessed, align=False, crop=False, grayscale=True, 
                                                unsharp=True,
                                                calibrate=False, gradient_strength=gradient_strength,
                                                gradient_threshold=gradient_threshold, denoise_strength = denoise_strength)
                if save: save_image(denoised[0], "unsharped")
                for stacking_alg in stacking_algorithms:
                    for average_alg in average_algs: 
                        print(f'\nRunning {features_alg} with gradient strength {gradient_strength}, gradient threshold {gradient_threshold}, denoise strength {denoise_strength} and stacking {stacking_alg} (avg = {average_alg})')
                        name = f'{features_alg}_str_{gradient_strength}_thr_{gradient_threshold}_dstr_{denoise_strength}'

                        # Stack the images using the selected stacking algorithm
                        if stacking_alg == 'weighted average':
                            image = weighted_average_stack(denoised, method=average_alg)
                        elif stacking_alg == 'median':
                            image = median_stack(denoised)
                        elif stacking_alg == 'sigma clipping':
                            image = sigma_clipping(denoised)
                    
                        for strength in unsharp_strengths:
                            for ker in kernel_sizes:
                                for limit in clip_limits:
                                    new_name = name + f'_{strength}_{ker}_{limit}_{average_alg}'
                                    print(f'Enhancing image with unsharp strength {strength}, kernel size {ker}, clip limit {limit}')
                                    unsharped = unsharp_mask([image], strength)[0]
                                    enhanced = enhance_contrast(unsharped, clip_limit=limit, tile_grid_size=ker)
                                    
                                    if save:
                                        print(f'\nSaving {name}')
                                        save_image(enhanced, new_name, 'images/output')
                                    
                                    brisque = calculate_brisque(enhanced)
                                    ssim = calculate_ssim(ref, enhanced)
                                    score = combined_score(brisque, ssim)
                                    print(f'BRISQUE score: {brisque}')
                                    print(f'SSIM score: {ssim}')
                                    print(f'combined_score: {score}')

                                    if config.COLAB:
                                        display_image(enhanced, brisque, ssim, score, new_name)

                                    # Aggiorna il miglior PSNR trovato
                                    if score > best_score:
                                        best_score = score
                                        best_img = new_name
                                        best_params = {"gradient_strength"  : gradient_strength,
                                                       "gradient_threshold" : gradient_threshold,
                                                       "denoise_strength"   : denoise_strength,  
                                                       "stacking_alg"       : stacking_alg,
                                                       "average_alg"        : average_alg,
                                                       "strength"           : strength,                                                     
                                                       "ker"                : ker,
                                                       "limit"              : limit
                                        }
    print(f'Best score: {best_score} at {best_img}')
    return best_params
"""
    
def main():
    bias = read_image('./images/masters/bias.tif') if os.path.exists('./images/masters/bias.tif') else None
    dark = read_image('./images/masters/dark.tif') if os.path.exists('./images/masters/dark.tif') else None
    flat = read_image('./images/masters/flat.tif') if os.path.exists('./images/masters/flat.tif') else None
    bias, dark, flat = calculate_masters(bias, dark, flat)
    
    images = read_images(config.input_folder)
    images = calibrate_images(images, bias, dark, flat)

    params, aligned = grid_search(images)
    process_images(None, params = params, aligned = aligned)

if __name__ == '__main__':
    main()