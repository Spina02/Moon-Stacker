import config
from image import read_images, save_image
from preprocess import preprocess_images
from grid_search import grid_search
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
    for strength in [0.5, 1, 1.5, 2, 2.5]:
        print(f"unsharping with strength: {strength}")
        unsharped = preprocess_images(preprocessed, align = False, crop = False, grayscale = True, unsharp = True, strength = strength, calibrate = False)

        # Stack the images
        if stacking_alg == 'weighted average':
            image = weighted_average_stack(unsharped, method=average_alg)
        elif stacking_alg == 'median':
            image = median_stack(unsharped)
        elif stacking_alg == 'sigma clipping':
            image = sigma_clipping(unsharped)

        # denoise the image
        #if denoise:
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model = model_init()
        #image = perform_denoising(model, image, device)

        # Save the image
        path = config.output_folder + f'/{features_alg}__{strength}_{stacking_alg}' 
        path += f'_{average_alg}' if stacking_alg == 'weighted average' else ''
        print(f'\nsaving {path}')
        save_image(image, path)

def canny_images(images, low_threshold=50, high_threshold=150):
    for i, image in enumerate(images):
        # Convert to grayscale if the image is RGB
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Apply Canny edge detection
        edges = cv2.Canny(to_8bit(gray_image, low_threshold, high_threshold)
        
        # Save the resulting image
        save_image(edges, f'./images/canny/canny_{i}')


def sobel_images(images):
    for i, image in enumerate(images):
        # Convert to grayscale if the image is RGB
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Apply Sobel filter on the grayscale image
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

        # Combine Sobel X and Y
        sobel = np.hypot(sobel_x, sobel_y)

        # Normalize the result to 0-255
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
        sobel = sobel.astype(np.uint8)

        save_image(sobel, f'./images/edge/sobel_{i}')

def main():
    #config_init()
    images = read_images(config.input_folder)
    sobel_images(images)
    canny_images(images)

    #image_stacking(images, denoise = False, features_alg = 'orb', stacking_alg = 'median', grayscale = True)
    #image_stacking(images, denoise = False, features_alg = 'orb', stacking_alg = 'sigma clipping', grayscale = True)
    #image_stacking(images, denoise = False, features_alg = 'orb', stacking_alg = 'weighted average', average_alg = 'composite', grayscale = True)

    #image_stacking_2(read_images(config.input_folder), )

if __name__ == '__main__':
    main()