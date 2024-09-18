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

    best_psnr = -9999
    best_path = ''

    preprocessed = preprocess_images(images, algo = 'orb', align = True, crop = True, grayscale = False, unsharp = False)
    #for stacking_alg in ['median', 'sigma clipping', 'weighted average']:
    
    #for strengths in [0.1, 0.3, 0.5]:
    #    for thresholds in [0.05, 0.1, 0.15]:
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
    #path += f'_{threshold}'
    print(f'\nsaving {path}')
    save_image(image, path)
    print(image_0.shape, image.shape)
    psnr = cv2.PSNR(image_0.astype(np.float32), image.astype(np.float32))
    print(f'PSNR: {psnr}')

def grid_search(images, features_alg = 'orb', average_alg = 'composite', stacking_alg = 'median', n_features = 10000):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False,crop = True, grayscale = True, unsharp = False, calibrate=False)[0]
    save_images([image_0], config.output_folder, name = 'original')

    best_psnr = -9999
    best_path = ''

    preprocessed = preprocess_images(images, algo = 'orb', align = True, crop = True, grayscale = False, unsharp = False)
    #for stacking_alg in ['median', 'sigma clipping', 'weighted average']:
    for strength in [3, 4, 5]:
        for ksize in [3, 5]:
            for sigma in [0, 2]:
                for tale in [(5, 5), (9, 9)]:
                    for low_clip in [0.01, 0.05]:
                        for high_clip in [0.5]:
                            print(f"unsharping with strength: {strength}, ksize: {ksize}, sigma: {sigma}, tale: {tale}, low_clip: {low_clip}, high_clip: {high_clip}")
                            unsharped = preprocess_images(preprocessed, align = False, crop = False, grayscale = True, unsharp = True, strength = strength, ksize = ksize, sigma = sigma, tale = tale, low_clip = low_clip, high_clip = high_clip, calibrate = False)

                            # Stack the images
                            if stacking_alg == 'weighted average':
                                image = weighted_average_stack(unsharped, method=average_alg)
                            elif stacking_alg == 'median':
                                image = median_stack(unsharped)
                            elif stacking_alg == 'sigma clipping':
                                image = sigma_clipping(unsharped)

                            # Save the image
                            path = config.output_folder + f'/{features_alg}_{strength}_{stacking_alg}' 
                            path += f'_{average_alg}' if stacking_alg == 'weighted average' else ''
                            path += f'_{ksize}_{sigma}_{tale}_{low_clip}_{high_clip}'
                            print(f'\nsaving {path}')
                            save_image(image, path)
                            print(image_0.shape, image.shape)
                            psnr = cv2.PSNR(image_0.astype(np.float32), image.astype(np.float32))
                            print(f'PSNR: {psnr}')

                            if psnr > best_psnr:
                                best_psnr = psnr
                                best_path = path
    
    print(f'Best PSNR: {best_psnr} at {best_path}')

def main():
    #images = read_images(config.input_folder)
    #images = preprocess_images(images, align = False, crop = True, grayscale = False, unsharp = False, calibrate = False)
    #save_images(images, config.output_folder, name = '/original')
    #images = [white_balance(image) for image in images]
    #save_images(images, config.output_folder, name = 'white_balanced', clear=False)
    #
    #clips = [0.25, 0.5, 0.75]
    #tiles = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    ## all combinations of clips and tiles
    #for clip in clips:
    #    for tile in tiles:
    #        images = [enhance_contrast(image, clip, tile) for image in images]
    #        save_images(images, config.output_folder, name = f'enhanced_{clip}_{tile}', clear=False)
    image_stacking(read_images(config.input_folder), stacking_alg = 'median')

    #grid_search(read_images(config.input_folder), stacking_alg = 'median')

if __name__ == '__main__':
    main()