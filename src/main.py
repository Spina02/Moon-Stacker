#from numpy import average, save
#from image import *
#from preprocess import *
#from debug import *
#from const import *
#from stack import *
#from metrics import image_analysis
from grid_search import grid_search


def main():
    # TODO: add a way to select the output format
    # TODO: add a way to select the input folder
    # TODO: add a way to select aligning algorithm
    # TODO: add a way to select stacking algorithm
    """
    # Read and convert RAW images to TIFF out_format in memory
    images = read_images(folder_path)

    # align images
    images = align_images(images, n_features = 1000)

    # crop images
    images = crop_to_center(images, margin = 100)
    
    # save images
    # save_images(cropped_images, output_path, out_format)

    # stack images
    median_image = median_stack(images)
    sigma_clipped_image = sigma_clipping(images)
    weighted_image = weighted_average_stack(images, method='snr')

    # save stacked image
    save_image(median_image, './images/median', out_format)
    
    save_image(sigma_clipped_image, './images/sigma_clipping', out_format)

    save_image(weighted_image, './images/weighted_average', out_format)

    # analysis
    image_analysis(median_image, 'Median')
    image_analysis(sigma_clipped_image, 'Sigma Clipping')
    image_analysis(weighted_image, 'Weighted Average')
    """

    # grid search
    grid_search()
     
if __name__ == '__main__':
    import main
    main.main()