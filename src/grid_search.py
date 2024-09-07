from metrics import *
from stack import *
from preprocess import *
from image import *
from const import DEBUG, folder_path
import stack

# lists of algs
align_algs = ['sift', 'surf', 'orb']
stacking_algs = ['median', 'sigma_clipping', 'weighted_average']
average_algs = ['snr', 'contrast', 'sharpness', 'variance', 'entropy']

def grid_search():
    print("\n starting grid search")
    images = read_images(folder_path)
    for align_alg in align_algs:
        images = align_images(images, algo=align_alg)
        images = crop_to_center(images, margin = 100)
        for stacking_alg in stacking_algs:
            print(f'Aligning: {align_alg}, Stacking: {stacking_alg},', end='')
            if stacking_alg == 'weighted_average':
                for average_alg in average_algs:
                    print(f' Average: {average_alg}')
                    image = weighted_average_stack(images, method=average_alg)
                    save_image(image, f'./images/{align_alg}_{stacking_alg}_{average_alg}', 'png')
                    image_analysis(image)
                    print('-------------------------------------')
            elif stacking_alg == 'median':
                print()
                image = median_stack(images)
                save_image(image, f'./images/{align_alg}_{stacking_alg}', 'png')
                image_analysis(image)
            elif stacking_alg == 'sigma_clipping':
                print()
                image = sigma_clipping(images)
                save_image(image, f'./images/{align_alg}_{stacking_alg}', 'png')
                image_analysis(image)
            print('-------------------------------------')