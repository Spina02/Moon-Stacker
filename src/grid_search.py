from metrics import *
from stack import *
from preprocess import *
from image import *
from const import DEBUG, folder_path
import matplotlib.pyplot as plt
import os

import stack

# lists of algs
#align_algs = ['sift', 'surf', 'orb']
align_algs = ['orb']#, 'surf']
#stacking_algs = ['median', 'sigma_clipping', 'weighted_average']
stacking_algs = ['weighted_average']
#average_algs = ['snr', 'contrast', 'sharpness', 'variance', 'entropy']
average_algs = ['snr']
n_features = [15, 25, 50, 75, 100, 200, 500]

def save_and_analyze_image(image, align_alg, nfeatures, stacking_alg, average_alg=None):
    print()
    #if average_alg:
    filename = f'./images/output/{align_alg}_{stacking_alg}_{nfeatures}'
    filename += f'_{average_alg}' if average_alg else ''
    #else:
    #    filename = f'./images/output/{align_alg}_{stacking_alg}_{nfeatures}'
    save_image(image, filename, 'png')
    image_analysis(image)

    # Visualizza l'immagine su Colab
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title(f'{align_alg} + {stacking_alg}' + (f' + {average_alg}' if average_alg else ''))
    #plt.axis('off')
    #plt.show()
    print('-------------------------------------')

def grid_search():
    # Clear the output folder
    for f in os.listdir('./images/output'):
        os.remove(os.path.join('./images/output', f))

    print("\n starting grid search")

    images = read_images(folder_path)
    
    image_0 = crop_to_center(images[:1])[0]

    save_and_analyze_image(image_0, 'original', 0, '')

    images = preprocess_images(images)

    for align_alg in align_algs:
        for nfeatures in n_features:
            print(f'Start aligning using {align_alg} ({nfeatures} features)')
            images = align_images(images, algo=align_alg, nfeatures=nfeatures)
            save_images(images, './images/aligned', 'png')
            print('Start cropping')
            images = crop_to_center(images, margin = 0)
            save_images(images, './images/cropped', 'png')
            if images is None:
                if DEBUG: print('No images to align')
                continue
            for stacking_alg in stacking_algs:
                if stacking_alg == 'weighted_average':
                    for average_alg in average_algs:
                        print(f'Aligning: {align_alg}, Stacking: {stacking_alg}, Average: {average_alg}')
                        image = weighted_average_stack(images, method=average_alg)
                        save_and_analyze_image(image, align_alg, stacking_alg, nfeatures, average_alg)
                        print('-------------------------------------')
                elif stacking_alg == 'median':
                    print(f'Aligning: {align_alg}, Stacking: {stacking_alg}')
                    image = median_stack(images)
                    save_and_analyze_image(image, align_alg, nfeatures, stacking_alg)
                elif stacking_alg == 'sigma_clipping':
                    print(f'Aligning: {align_alg}, Stacking: {stacking_alg}')
                    image = sigma_clipping(images)
                    save_and_analyze_image(image, align_alg, nfeatures, stacking_alg)
                print('-------------------------------------')