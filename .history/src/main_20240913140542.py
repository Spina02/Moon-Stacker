import config
from image import read_images, save_image, read_image
from preprocess import preprocess_images
from grid_search import visualize_image
from grid_search import grid_search
from stack import *

def image_stacking(features_alg = 'orb', stacking_alg = 'median', avera, n_features = 5000, denoise = False, denoise_alg = 'gaussian', sigma = 1.5, strength = 3, crop = True):

    # Read the images from the input folder
    images = read_images(config.input_folder)

    # Preprocess the images
    preprocessed = preprocess_images(images, nfeatures=n_features, align=features_alg, sigma = sigma, strength = strength, crop = crop, denoise = denoise, denoise_alg = denoise_alg)

    # Stack the images
    if stacking_alg == 'weighted average':
        image = weighted_average_stack(preprocessed, method=average_alg)
    elif stacking_alg == 'median':
        image = median_stack(preprocessed)
    elif stacking_alg == 'sigma clipping':
        image = sigma_clipping(preprocessed)

    if COLAB: visualize_image(image, config.output_folder + f'/{features_alg}_{stacking_alg}_{n_features}' + f'_{average_alg}' if stacking_alg == 'weighted average' else '')

    # Save the image
    save_image(image, config.output_folder + f'/{features_alg}_{stacking_alg}_{n_features}_{average_alg}')
    
    return image

def main():
    #config_init()

    image_stacking()

if __name__ == '__main__':
    import main
    main.main()