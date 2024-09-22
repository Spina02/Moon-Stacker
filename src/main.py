import config
from image import *
from preprocessing import preprocess_images
from stacking import *
import cv2

def image_stacking(images, features_alg = 'orb', calibrate = True, average_alg = 'composite', stacking_alg = 'median', n_features = 10000, strengths=[1.2, 1.4, 1.6], thresholds=[0.5, 0.5, 0.5], ks=[3, 5, 7], grayscale = True, method = 'multi-scale'):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False, crop = True, grayscale = grayscale, unsharp = False, calibrate=False)[0]
    save_image(image_0, 'original')

    preprocessed = preprocess_images(images, algo = 'orb', align = True, crop = True, grayscale = False, unsharp = False, calibrate = calibrate)

    for stacking_alg in ['weighted average', 'median', 'sigma clipping']:
        
        unsharped = preprocess_images(preprocessed, align = False, crop = False, grayscale = grayscale, unsharp = True, sharpening_method = method, calibrate = False, strengths=strengths, thresholds=thresholds, ks=ks)

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
        
        print(f'PSNR: {cv2.PSNR(image_0, image)}')

def grid_search(images, features_alg='orb', average_alg='composite', n_features=10000, method='multi_scale'):
    print()
    image_0 = preprocess_images([images[0]], align=False, crop=True, grayscale=True, unsharp=False, calibrate=False)[0]
    save_image(image_0, name='original')

    best_psnr = -9999
    best_img = ''

    preprocessed = preprocess_images(images, algo=features_alg, nfeatures=n_features, align=True, crop=True, grayscale=False, unsharp=False)

    # Lista di parametri per il sharpening
    strengths_list = [[1.0, 1.2, 1.5]]
    thresholds_list = [[0.5, 0.5, 0.5]]
    ks_list = [[3, 5, 7]]

    # Lista degli algoritmi di stacking
    stacking_algorithms = ['weighted average']# , 'median', 'sigma clipping']

    for strengths in strengths_list:
        for thresholds in thresholds_list:
            for ks in ks_list:
                    for gradient_strength in [1.0, 1.1, 1.25]:
                        for gradient_threshold in [0.012, 0.0125]:
                            for denoise_strength in [1]:
                                # Preprocess the images with the selected sharpening method
                                unsharped = preprocess_images(preprocessed, align=False, crop=False, grayscale=True, 
                                                              unsharp=True, strengths=strengths, thresholds=thresholds, ks=ks, 
                                                              calibrate=False, sharpening_method=method, gradient_strength=gradient_strength,
                                                              gradient_threshold=gradient_threshold, denoise_strength = denoise_strength)
                                for stacking_alg in stacking_algorithms:
                                                    # name
                                    if method == 'multi_scale':
                                        print(f'\nRunning {features_alg} with strengths {strengths}, thresholds {thresholds}, ks {ks} and stacking {stacking_alg}')
                                        name = f'{features_alg}_strengths_{strengths}_ks_{ks}_thresholds_{thresholds}_stacking_{stacking_alg}'
                                    elif method == 'gradient':
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
                                    save_image(image, name, config.output_folder)

                                    image = unsharp_mask([image], 2)[0]

                                    save_image(image, name + '_sharp', config.output_folder)

                                    # Calcola il PSNR rispetto all'immagine originale
                                    psnr = cv2.PSNR(image_0.astype(np.float32), image.astype(np.float32))
                                    print(f'PSNR: {psnr}')

                                    # Aggiorna il miglior PSNR trovato
                                    if psnr > best_psnr:
                                        best_psnr = psnr
                                        best_img = name

    print(f'Best PSNR: {best_psnr} at {best_img}')


def unsharp_mask(images, strength):

    #blurred_images = dncnn_images(model, images)
    blurred_images = [cv2.GaussianBlur(image, (3, 3), 1) for image in images]
    #save_images(blurred_images, './images/blurred', name = 'blurred')

    merged_images = [to_16bit(cv2.addWeighted(to_16bit(image), 0.5 + strength, to_16bit(blurred_image), 0.5 -strength, 0)) for image, blurred_image in zip(images, blurred_images)]
    
    #save_images(merged_images, name = 'merged', folder_path = './images/merged', clear = False)

    return merged_images

def main():
    #grid_search(read_images(config.input_folder), method='multi_scale')
    #grid_search(read_images(config.input_folder), method='gradient')
    save_image(image_stacking(read_images(config.input_folder), method='gradient', calibrate = True), 'stacked')

if __name__ == '__main__':
    main()