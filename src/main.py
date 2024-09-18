import config
from image import *
from preprocessing import preprocess_images
from stacking import *
#from denoise import model_init, perform_denoising
import cv2

def image_stacking(images, features_alg = 'orb', average_alg = 'composite', stacking_alg = 'median', n_features = 10000):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False,crop = True, grayscale = True, unsharp = False, calibrate=False)[0]
    save_images([image_0], config.output_folder, name = 'original')

    preprocessed = preprocess_images(images, algo = 'orb', align = True, crop = True, grayscale = False, unsharp = False)

    unsharped = preprocess_images(preprocessed, align = False, crop = False, grayscale = True, unsharp = True, calibrate = False)

    # Stack the images
    if stacking_alg == 'weighted average':
        image = weighted_average_stack(unsharped, method=average_alg)
    elif stacking_alg == 'median':
        image = median_stack(unsharped)
    elif stacking_alg == 'sigma clipping':
        image = sigma_clipping(unsharped)

    # Save the image
    path = config.output_folder + f'/{features_alg}_{stacking_alg}' 
    path += f'_{average_alg}' if stacking_alg == 'weighted average' else ''
    print(f'\nsaving {path}')
    save_image(image, path)
    print(image_0.shape, image.shape)
    psnr = cv2.PSNR(image_0.astype(np.float32), image.astype(np.float32))
    print(f'PSNR: {psnr}')

def grid_search(images, features_alg='orb', average_alg='composite', stacking_alg='median', n_features=10000):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False, crop=True, grayscale=True, unsharp=False, calibrate=False)[0]
    save_images([image_0], config.output_folder, name='original')

    best_psnr = -9999
    best_path = ''

    preprocessed = preprocess_images(images, algo=features_alg, align=True, crop=True, grayscale=False, unsharp=False)

    # Lista di parametri per multi-scale unsharp masking
    strengths_list = [[0.8, 0.9, 1.0], [1.0, 1.2, 1.5], [0.6, 0.8, 1.0]]
    thresholds_list = [[0.4, 0.5, 0.6], [0.5, 0.5, 0.5], [0.3, 0.4, 0.5]]
    ks_list = [[5, 5, 5], [3, 5, 7], [4, 4, 6]]

    for strengths in strengths_list:
        for thresholds in thresholds_list:
            for ks in ks_list:
                print(f"Testing unsharp with strengths: {strengths}, thresholds: {thresholds}, ks: {ks}")

                unsharped = preprocess_images(preprocessed, align=False, crop=False, grayscale=True, 
                                              unsharp=True, strengths=strengths, thresholds=thresholds, ks=ks, calibrate=False)

                # Stack the images
                if stacking_alg == 'weighted average':
                    image = weighted_average_stack(unsharped, method=average_alg)
                elif stacking_alg == 'median':
                    image = median_stack(unsharped)
                elif stacking_alg == 'sigma clipping':
                    image = sigma_clipping(unsharped)

                # Save the image
                path = config.output_folder + f'/{features_alg}_strengths_{strengths}_ks_{ks}_stacking_{stacking_alg}'
                print(f'\nSaving {path}')
                save_image(image, path)

                psnr = cv2.PSNR(image_0.astype(np.float32), image.astype(np.float32))
                print(f'PSNR: {psnr}')

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_path = path
                
    print(f'\nBest PSNR: {best_psnr} at path {best_path}')

def main():
    grid_search(read_images(config.input_folder), stacking_alg='median')

if __name__ == '__main__':
    main()