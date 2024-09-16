import config
from image import *
from preprocess import preprocess_images
from stack import *
import torch
from models import model_init, perform_denoising, unsharp_mask


def image_stacking(images = None, features_alg = 'orb', stacking_alg = 'median', average_alg = None, n_features = 10000, denoise = False, denoise_alg = 'DnCnn', grayscale = True, crop = True, unsharp = True):

    print()
    if not images:
      # Read the images from the input folder
      images = read_images(config.input_folder)

    # Preprocess the images
    preprocessed = preprocess_images(images, nfeatures=n_features, align=features_alg, crop = crop, grayscale = grayscale, unsharp = unsharp)

    # Stack the images
    if stacking_alg == 'weighted average':
        image = weighted_average_stack(preprocessed, method=average_alg)
    elif stacking_alg == 'median':
        image = median_stack(preprocessed)
    elif stacking_alg == 'sigma clipping':
        image = sigma_clipping(preprocessed)

    # denoise the image
    #if denoise:
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model_init()
    #image = perform_denoising(model, image, device)

    # Save the image
    path = config.output_folder + f'/{features_alg}_{stacking_alg}_{n_features}' 
    path += f'_{average_alg}' if stacking_alg == 'weighted average' else ''
    save_image(image, path)
    
    return image

def image_stacking_2(images, features_alg = 'orb', average_alg = 'composite', stacking_alg = 'median'):
    print()

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

def main():
    image_stacking_2(read_images(config.input_folder), stacking_alg = 'median')


if __name__ == '__main__':
    main()