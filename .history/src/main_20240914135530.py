import config
from image import read_images, save_image
from preprocess import preprocess_images
from grid_search import grid_search
from stack import *
import torch
from models import model_init, perform_denoising


def image_stacking(features_alg = 'orb', stacking_alg = 'median', average_alg = None, n_features = 10000, denoise = False, denoise_alg = 'DnCnn', grayscale = True, crop = True, unsharp = True):

    print()
    # Read the images from the input folder
    images = read_images(config.input_folder)

    # Preprocess the images
    preprocessed = preprocess_images(images, nfeatures=n_features, align=features_alg, crop = crop, denoise = denoise, denoise_alg = denoise_alg, grayscale = grayscale, unsharp = unsharp)

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

def main():
    #config_init()

    image_stacking(denoise = False, features_alg = 'orb', grayscale = True)
    image_stacking(denoise = False, features_alg = 'orb', stacking_alg = 'sigma clipping', grayscale = True)
    image_stacking(denoise = False, features_alg = 'surf', stacking_alg = 'sigma clipping', grayscale = True)



if __name__ == '__main__':
    main()