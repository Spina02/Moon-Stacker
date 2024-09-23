import config
from image import *
from preprocessing import preprocess_images, unsharp_mask
from stacking import *
import cv2
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

def grid_search(images, features_alg='orb', average_alg='sharpness', n_features=10000):
    print()
    image_0 = preprocess_images([images[0]], align=False, crop=True, grayscale=True, unsharp=False, calibrate=False)[0]
    save_image(image_0, name='original')

    best_psnr = -9999
    best_img = ''

    preprocessed = preprocess_images(images, algo=features_alg, nfeatures=n_features, align=True, crop=True, grayscale=False, unsharp=False)

    # Lista degli algoritmi di stacking
    stacking_algorithms = ['weighted average', 'sigma clipping', 'median']

    for gradient_strength in [0.5, 1.0, 1.5]:
        for gradient_threshold in [0.005, 0.0075, 0.01, 0.0125]:
            for denoise_strength in [0.5, 0.75, 1]:
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

                    # Calcola il PSNR rispetto all'immagine originale
                    psnr = cv2.PSNR(to_8bit(image_0), to_8bit(image))
                    print(evaluate_improvement(to_8bit(image_0), to_8bit(image)))
                    print(f'PSNR: {psnr}')

                    # Aggiorna il miglior PSNR trovato
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_img = name

    print(f'Best PSNR: {best_psnr} at {best_img}')

def main():
    bias, dark, flat = calculate_masters()
    images = read_images(config.input_folder)
    save_images(images[:1], 'original')
    images = calibrate_images(images, bias, dark, flat)
    save_image(images[0], 'calibrated')
    grid_search(images)

if __name__ == '__main__':
    main()