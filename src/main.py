import config
from image import *
from preprocessing import preprocess_images, unsharp_mask
from stacking import weighted_average_stack, median_stack, sigma_clipping
from metrics import calculate_brisque, calculate_ssim
from calibration import calculate_masters, calibrate_images
from align import enhance_contrast
import os

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
    
def grid_search(images):
    print()
    image_0 = preprocess_images([images[0]], align=False, crop=True, grayscale=True, unsharp=False, calibrate=False)[0]
    save_image(image_0, name='original')

    best_score = 100 # brisque is 0-100 and lower values means better images
    best_img = ''

    features_alg='orb'
    n_features=50000
    print(f'aligning images with {features_alg} and {n_features} features')
    preprocessed = preprocess_images(images, algo=features_alg, nfeatures=n_features, grayscale=False, unsharp=False)
    save_image(preprocessed[0], 'aligned')

    # Lista degli algoritmi di stacking
    stacking_algorithms = ['weighted average']#, 'sigma clipping', 'median']

    average_alg='brisque'#'sharpness'
    for gradient_strength in [1.5, 1.75]:
        for gradient_threshold in [0.0025, 0.005]:
            for denoise_strength in [0.75, 0.9]:
                # Preprocess the images with the selected sharpening method
                unsharped = preprocess_images(preprocessed, align=False, crop=False, grayscale=True, 
                                                unsharp=True,
                                                calibrate=False, gradient_strength=gradient_strength,
                                                gradient_threshold=gradient_threshold, denoise_strength = denoise_strength)
                save_image(unsharped[0], "unsharped")
                for stacking_alg in stacking_algorithms:
                    print(f'\nRunning {features_alg} with gradient strength {gradient_strength}, gradient threshold {gradient_threshold}, denoise strength {denoise_strength} and stacking {stacking_alg}')
                    name = f'{features_alg}_str_{gradient_strength}_thr_{gradient_threshold}_dstr_{denoise_strength}'

                    # Stack the images using the selected stacking algorithm
                    if stacking_alg == 'weighted average':
                        image = weighted_average_stack(unsharped, method=average_alg)
                    elif stacking_alg == 'median':
                        image = median_stack(unsharped)
                    elif stacking_alg == 'sigma clipping':
                        image = sigma_clipping(unsharped)
                    
                    #save_image(image, name, config.output_folder)

                    #unsharped = unsharp_mask([image], 1.3)[0]
                    #enhanced = enhance_contrast(unsharped, clip_limit=0.85, tile_grid_size=(11, 11))

                    #save_image(enhanced, name + '_sharp', config.output_folder)

                    #brisque = calculate_brisque(enhanced)
                    #print(f'BRISQUE score: {brisque}')

                    strengths = [1, 1.25]
                    kernel_sizes = [(7,7), (9,9), (11, 11)]
                    limits = [0.75, 0.8, 0.85]
                
                    #image = read_image('images/output/orb_str_1.75_thr_0.005_dstr_0.75_stack_weighted average.png')
                    for strength in strengths:
                        for ker in kernel_sizes:
                            for limit in limits:
                                new_name = name + f'_{strength}_{ker}_{limit}'
                                print(f'Enhancing image with unsharp strength {strength}, kernel size {ker}, clip limit {limit}')
                                unsharped = unsharp_mask([image], strength)[0]
                                enhanced = enhance_contrast(unsharped, clip_limit=limit, tile_grid_size=ker)
                                
                                print(f'\nSaving {name}')
                                save_image(enhanced, new_name, 'images/output')
                                
                                brisque = calculate_brisque(to_8bit(enhanced))
                                ssim = calculate_ssim(to_8bit(image_0), to_8bit(enhanced))
                                print(f'BRISQUE score: {brisque}')
                                print(f'SSIM score: {ssim}')

                                score = brisque * (1 - ssim)

                                print(f'score [brisque * (1 - ssim)] = {score}')

                                if config.COLAB:
                                  display_image(enhanced, brisque, ssim, score, new_name)

                                # Aggiorna il miglior PSNR trovato
                                if score < best_score:
                                    best_score = score
                                    best_img = new_name 

    print(f'Best score: {best_score} at {best_img}')

def main():
    if os.path.exists('./images/masters/bias.tif'):
      bias = read_image('./images/masters/bias.tif')
    else:
      bias = None
    if os.path.exists('./images/masters/dark.tif'):
      dark = read_image('./images/masters/dark.tif')
    else:
      dark = None
    if os.path.exists('./images/masters/flat.tif'):
        flat = read_image('./images/masters/flat.tif')
    else:
      flat =  None

    bias, dark, flat = calculate_masters(bias, dark, flat)
    
    images = read_images(config.input_folder)
    save_image(images[0], 'original')
    images = calibrate_images(images, bias, dark, flat)
    save_image(images[0], 'calibrated')
    grid_search(images)

    #strengths = [1.25, 1.3, 1.35]
    #kernel_sizes = [(11, 11), (13, 13)]
    #limits = [0.84, 0.85, 0.86]
##
    ##image = read_image('images/output/orb_str_1.75_thr_0.005_dstr_0.75_stack_weighted average.png')
    #for strength in strengths:
    #  for ker in kernel_sizes:
    #    for limit in limits:
    #      print(f'{strength}_{ker}_{limit}')
    #      unsharped = unsharp_mask([image], strength)[0]
    #      enhanced = enhance_contrast(unsharped, clip_limit=limit, tile_grid_size=ker)
    #      save_image(enhanced, f'{strength}_{ker}_{limit}', 'images/boh')
    #      brisque = calculate_brisque(enhanced)
    #      print(f'BRISQUE score: {brisque}')
#


if __name__ == '__main__':
    main()