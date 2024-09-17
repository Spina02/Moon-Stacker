import config
from image import *
from preprocessing import preprocess_images
from stacking import *
#from denoise import model_init, perform_denoising
import cv2

def image_stacking(images = None, features_alg = 'orb', stacking_alg = 'median', average_alg = None, n_features = 10000, grayscale = True, crop = True, unsharp = True):

    print()
    if not images:
      # Read the images from the input folder
      images = read_images(config.input_folder)

    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False,crop = True, grayscale = grayscale, unsharp = False)[0]
    save_image(image_0, config.output_folder + '/original')
    # Preprocess the images
    preprocessed = preprocess_images(images, nfeatures=n_features, align=True, algo = features_alg, crop = crop, grayscale = grayscale, unsharp = unsharp)

    # Stack the images
    if stacking_alg == 'weighted average':
        image = weighted_average_stack(preprocessed, method=average_alg)
    elif stacking_alg == 'median':
        image = median_stack(preprocessed)
    elif stacking_alg == 'sigma clipping':
        image = sigma_clipping(preprocessed)

    # Save the image
    path = config.output_folder + f'/{features_alg}_{stacking_alg}_{n_features}' 
    path += f'_{average_alg}' if stacking_alg == 'weighted average' else ''
    save_image(image, path)
    psnr = cv2.PSNR(image_0, image)
    print(f'PSNR: {psnr}')
    
    return image

def image_stacking(images, features_alg = 'orb', average_alg = 'composite', stacking_alg = 'median'):
    print()
    image_0 = preprocess_images([images[0]], nfeatures=n_features, align=False,crop = True, grayscale = grayscale, unsharp = False)[0]
    save_image(image_0, config.output_folder + '/original')

    preprocessed = preprocess_images(images, algo = 'orb', align = True, crop = True, grayscale = False, unsharp = False)
    for stacking_alg in ['median', 'sigma clipping', 'weighted average']:
        for strength in [10, 15, 20]:
            print(f"unsharping with strength: {strength}")
            unsharped = preprocess_images(preprocessed, align = False, crop = False, grayscale = True, unsharp = True, strength = strength, calibrate = False)

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
            print(f'\nsaving {path}')
            save_image(image, path)
            psnr = calculate_psnr(image_0, image)
            print(f'PSNR: {psnr}')

def main():
    image_stacking(read_images(config.input_folder), stacking_alg = 'median')

if __name__ == '__main__':
    main()