from metrics import image_analysis
from stack import *
from preprocess import *
from image import *
import config

# lists of algs
align_algs = ['orb', 'sift', 'surf']
stacking_algs = ['median', 'sigma_clipping', 'weighted_average']
average_algs = ['snr']#, 'composite']
n_features = [5000]#, 5000]
sigmas = [2, 1.5]
strengths = [3]

params = {
    'align_alg':    ['sift',    'surf',             'orb'],
    'stacking_alg': ['median',  'sigma_clipping',   'weighted_average'],
    'average_alg':  ['snr',     'composite'],
    'n_features':   [1000,      2000,               5000],
    'sigma':        [1,         2],
    'strength':     [2,         3]
}

def save_and_analyze_image(image, filepath):
    print()
    save_image(image, filepath)
    if DEBUG: image_analysis(image)
    print('-------------------------------------')

def grid_search(crop = True):
    # Clear the output folder
    clear_folder(config.output_folder)
    clear_folder('./images/preprocessed')

    print("\n starting grid search\n")

    # Read the images from the input folder
    images = read_images(config.input_folder)
    
    original = preprocess_images(images[:1], align = False, sharpen = False, crop = crop, denoise = False)[0]
    print("saving original image")
    save_and_analyze_image(original, f'{config.output_folder}/original')

    for align_alg in align_algs:
        for nfeatures in n_features:
            for sigma in sigmas:
                for strength in strengths:
                    if DEBUG: print(f'Start aligning using {align_alg} ({nfeatures} features)')
                    preprocessed = preprocess_images(images, nfeatures=nfeatures, align=align_alg, sigma = sigma, strength = strength, crop = crop)
                    save_images(preprocessed, './images/preprocessed', name = f'{align_alg}_{nfeatures}_si{sigma}_st{strength}', clear = False)
                    for stacking_alg in stacking_algs:
                        if stacking_alg == 'weighted_average':
                            for average_alg in average_algs:
                                filepath = f'{config.output_folder}/{align_alg}_{stacking_alg}_{nfeatures}_{average_alg}'
                                if DEBUG: print(f'Aligning: {align_alg}, Stacking: {stacking_alg}, Average: {average_alg}')
                                image = weighted_average_stack(preprocessed, method=average_alg)
                                save_and_analyze_image(image, filepath)
                                if COLAB: visualize_image(image, filepath)
                        elif stacking_alg == 'median':
                            filepath = f'{config.output_folder}/{align_alg}_{stacking_alg}_{nfeatures}'
                            if DEBUG: print(f'Aligning: {align_alg}, Stacking: {stacking_alg}')
                            image = median_stack(preprocessed)
                            save_and_analyze_image(image, filepath)
                            if COLAB: visualize_image(image, filepath)
                        elif stacking_alg == 'sigma_clipping':
                            filepath = f'{config.output_folder}/{align_alg}_{stacking_alg}_{nfeatures}'
                            if DEBUG: print(f'Aligning: {align_alg}, Stacking: {stacking_alg}')
                            image = sigma_clipping(preprocessed)
                            save_and_analyze_image(image, filepath)
                            if COLAB: visualize_image(image, filepath)