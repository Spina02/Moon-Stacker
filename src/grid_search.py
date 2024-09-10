from kiwisolver import strength
from sympy import N
from metrics import *
from stack import *
from preprocess import *
from image import *
from const import DEBUG, folder_path
import os

# lists of algs
#align_algs = ['sift', 'surf', 'orb']
align_algs = ['orb']
stacking_algs = ['median', 'sigma_clipping', 'weighted_average']
#stacking_algs = ['weighted_average']
average_algs = ['snr', 'composite']
sharpening_algs = ['unsharp_mask', 'sharpen']
n_features = [100, 300, 500]
sigmas = [1.5]
strengths = [2]

def save_and_analyze_image(image, align_alg, nfeatures, stacking_alg, average_alg=None):
    print()
    #if average_alg:
    filename = f'./images/output/{align_alg}_{stacking_alg}_{nfeatures}'
    filename += f'_{average_alg}' if average_alg else ''
    save_image(image, filename)
    image_analysis(image)
    print('-------------------------------------')

def grid_search():
    # Clear the output folder
    for f in os.listdir('./images/output'):
        os.remove(os.path.join('./images/output', f))

    print("\n starting grid search")

    images = read_images(folder_path)
    
    image_0 = preprocess_images(images[:1], align = False, sharpen = False)[0]

    save_and_analyze_image(image_0, 'original', 0, 'original')


    for align_alg in align_algs:
        for nfeatures in n_features:
            for sigma in sigmas:
                for strength in strengths:
                    print(f'Start aligning using {align_alg} ({nfeatures} features)')
                    preprocessed = preprocess_images(images, nfeatures=nfeatures, align=align_alg, sigma = sigma, strength = strength)
                    save_images(preprocessed, './images/preprocessed', name = f'{align_alg}_{nfeatures}_si{sigma}_st{strength}', clear = False)
                    for stacking_alg in stacking_algs:
                        if stacking_alg == 'weighted_average':
                            for average_alg in average_algs:
                                print(f'Aligning: {align_alg}, Stacking: {stacking_alg}, Average: {average_alg}')
                                image = weighted_average_stack(preprocessed, method=average_alg)
                                save_and_analyze_image(image, align_alg, stacking_alg, nfeatures, average_alg)
                        elif stacking_alg == 'median':
                            print(f'Aligning: {align_alg}, Stacking: {stacking_alg}')
                            image = median_stack(preprocessed)
                            save_and_analyze_image(image, align_alg, nfeatures, stacking_alg)
                        elif stacking_alg == 'sigma_clipping':
                            print(f'Aligning: {align_alg}, Stacking: {stacking_alg}')
                            image = sigma_clipping(preprocessed)
                            save_and_analyze_image(image, align_alg, nfeatures, stacking_alg)