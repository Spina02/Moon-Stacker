import config
from image import *
from preprocessing import preprocess_images, unsharp_mask
from stacking import *
from metrics import calculate_brisque
from calibration import *

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
        
        #print(f'PSNR: {cv2.PSNR(image_0, image)}')

def grid_search(images):
    print()
    image_0 = preprocess_images([images[0]], align=False, crop=True, grayscale=True, unsharp=False, calibrate=False)[0]
    save_image(image_0, name='original')

    best_brisque = 100 # brisque is 0-100 and lower values means better images
    best_img = ''

    features_alg='orb'
    n_features=10000
    print(f'aligning images with {features_alg} and {n_features} features')
    preprocessed = preprocess_images(images, algo=features_alg, nfeatures=n_features, grayscale=False, unsharp=False)

    # Lista degli algoritmi di stacking
    stacking_algorithms = ['weighted average']#, 'sigma clipping', 'median']

    average_alg='brisque'#'sharpness'
    for gradient_strength in [0.5, 1.0, 1.5]:
        for gradient_threshold in [0.0075]:#[0.005, 0.0075, 0.01, 0.0125]:
            for denoise_strength in [0.5]:#, 0.75, 1]:
                # Preprocess the images with the selected sharpening method
                unsharped = preprocess_images(preprocessed, align=False, crop=False, grayscale=True, 
                                                unsharp=True,
                                                calibrate=False, gradient_strength=gradient_strength,
                                                gradient_threshold=gradient_threshold, denoise_strength = denoise_strength)
                for stacking_alg in stacking_algorithms:
                    print(f'\nRunning {features_alg} with gradient strength {gradient_strength}, gradient threshold {gradient_threshold}, denoise strength {denoise_strength} and stacking {stacking_alg}')
                    name = f'{features_alg}_str_{gradient_strength}_thr_{gradient_threshold}_dstr_{denoise_strength}_stack_{stacking_alg}'

                    # Stack the images using the selected stacking algorithm
                    if stacking_alg == 'weighted average':
                        image = weighted_average_stack(unsharped, method=average_alg)
                    elif stacking_alg == 'median':
                        image = median_stack(unsharped)
                    elif stacking_alg == 'sigma clipping':
                        image = sigma_clipping(unsharped)

                    # Save the image
                    print(f'\nSaving {name}')
                    #save_image(image, name, config.output_folder)

                    image = unsharp_mask([image], 2)[0]

                    save_image(image, name + '_sharp', config.output_folder)

                    brisque = calculate_brisque(image)
                    print(f'BRISQUE score: {brisque}')

                    # Aggiorna il miglior PSNR trovato
                    if brisque > best_brisque:
                        best_brisque = brisque
                        best_img = name

                    del image
                    gc.collect()

    print(f'Best BRISQUE: {best_brisque} at {best_img}')

def main():
    bias = read_image('./images/masters/bias.tif') if os.path.exists('./images/masters/bias.tif') else None
    dark = read_image('./images/masters/dark.tif') if os.path.exists('./images/masters/dark.tif') else None
    flat = read_image('./images/masters/flat.tif') if os.path.exists('./images/masters/flat.tif') else None
    bias, dark, flat = calculate_masters(bias, dark, flat)
    
    images = read_images(config.input_folder)
    save_image(images[0], 'original')
    images = calibrate_images(images, bias, dark, flat)
    save_image(images[0], 'calibrated')
    grid_search(images)

if __name__ == '__main__':
    main()