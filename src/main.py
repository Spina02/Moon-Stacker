import config
from image import *
from preprocessing import preprocess_images
from stacking import *
#from denoise import model_init, perform_denoising
import cv2

def image_stacking(images, features_alg = 'orb', average_alg = 'composite', stacking_alg = 'median', n_features = 10000, strengths=[1.2, 1.4, 1.6], thresholds=[0.5, 0.5, 0.5], ks=[3, 5, 7]):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False,crop = True, grayscale = True, unsharp = False, calibrate=False)[0]
    save_images([image_0], config.output_folder, name = 'original')

    preprocessed = preprocess_images(images, algo = 'orb', align = True, crop = True, grayscale = False, unsharp = False)

    for stacking_alg in ['weighted average', 'median', 'sigma clipping']:
        unsharped = preprocess_images(preprocessed, align = False, crop = False, grayscale = True, unsharp = True, calibrate = False, strengths=strengths, thresholds=thresholds, ks=ks)

        # Stack the images
        if stacking_alg == 'weighted average':
            image = weighted_average_stack(unsharped, method=average_alg)
        elif stacking_alg == 'median':
            image = median_stack(unsharped)
        elif stacking_alg == 'sigma clipping':
            image = sigma_clipping(unsharped)

        # Save the image
        name = f'/{features_alg}_{stacking_alg}' 
        name += f'_{average_alg}' if stacking_alg == 'weighted average' else ''
        print(f'\nsaving {name}')
        save_images([image], config.output_folder, name = name)
        print(image_0.shape, image.shape)
        psnr = cv2.PSNR(image_0.astype(np.float32), image.astype(np.float32))
        print(f'PSNR: {psnr}')

def grid_search(images, features_alg='orb', average_alg='composite', n_features=10000, method='multi_scale'):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False, crop=True, grayscale=True, unsharp=False, calibrate=False)[0]
    save_images([image_0], config.output_folder, name='original')

    best_psnr = -9999
    best_img = ''

    preprocessed = preprocess_images(images, algo=features_alg, align=True, crop=True, grayscale=False, unsharp=False)

    # Lista di parametri per il sharpening
    strengths_list = [[1.0, 1.2, 1.5]]
    thresholds_list = [[0.5, 0.5, 0.5]]
    ks_list = [[3, 5, 7]]

    # Lista degli algoritmi di stacking
    stacking_algorithms = ['median', 'weighted average', 'sigma clipping']

    for strengths in strengths_list:
        for thresholds in thresholds_list:
            for ks in ks_list:
                for stacking_alg in stacking_algorithms:
                    for gradient_strength in [0.8, 1.5, 2]:
                        for gradient_threshold in [0.001, 0.005, 0.007]:
                            # name
                            if method == 'multi_scale':
                                print(f'\nRunning {features_alg} with strengths {strengths}, thresholds {thresholds}, ks {ks} and stacking {stacking_alg}')
                                name = f'{features_alg}_strengths_{strengths}_ks_{ks}_thresholds_{thresholds}_stacking_{stacking_alg}'
                            elif method == 'gradient':
                                print(f'\nRunning {features_alg} with gradient strength {gradient_strength}, gradient threshold {gradient_threshold} and stacking {stacking_alg}')
                                name = f'{features_alg}_strength_{gradient_strength}_threshold_{gradient_threshold}_stacking_{stacking_alg}'

                            # Preprocess the images with the selected sharpening method
                            unsharped = preprocess_images(preprocessed, align=False, crop=False, grayscale=True, 
                                                          unsharp=True, strengths=strengths, thresholds=thresholds, ks=ks, 
                                                          calibrate=False, sharpening_method=method, gradient_strength=gradient_strength, gradient_threshold=gradient_threshold)

                            # Stack the images using the selected stacking algorithm
                            if stacking_alg == 'weighted average':
                                image = weighted_average_stack(unsharped, method=average_alg)
                            elif stacking_alg == 'median':
                                image = median_stack(unsharped)
                            elif stacking_alg == 'sigma clipping':
                                image = sigma_clipping(unsharped)

                            # Save the image
                            print(f'\nSaving {name}')
                            save_image(image, config.output_folder, name)

                            # Calcola il PSNR rispetto all'immagine originale
                            psnr = cv2.PSNR(image_0.astype(np.float32), image.astype(np.float32))
                            print(f'PSNR: {psnr}')

                            # Aggiorna il miglior PSNR trovato
                            if psnr > best_psnr:
                                best_psnr = psnr
                                best_img = name

    print(f'Best PSNR: {best_psnr} at {best_img}')


def main():
    #grid_search(read_images(config.input_folder), method='multi_scale')
    grid_search(read_images(config.input_folder), method='gradient')

if __name__ == '__main__':
    main()