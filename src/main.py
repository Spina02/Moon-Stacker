import config
from image import read_image, read_images,display_image
from preprocessing import crop_to_center
from metrics import calculate_metrics
from calibration import calculate_masters, calibrate_images
from grid_search import grid_search
import os
from process import process_images

metrics = ['niqe', 'piqe', 'liqe']#, 'nima', 'brisque_matlab']
    
def main():
    bias = read_image('images/masters/bias.tif') if os.path.exists('images/masters/bias.tif') else None
    dark = read_image('images/masters/dark.tif') if os.path.exists('images/masters/dark.tif') else None
    flat = read_image('images/masters/flat.tif') if os.path.exists('images/masters/flat.tif') else None
    bias, dark, flat = calculate_masters(bias, dark, flat)
    
    images = read_images(config.input_folder)

    image_0 = crop_to_center([images[0]])[0]
    if config.COLAB:
        display_image(image_0, "image 0")
    
    calculate_metrics(image_0, "image 0", metrics)

    images = calibrate_images(images, bias, dark, flat)

    calibrated_0 = crop_to_center([images[0]])[0]
    if config.COLAB:
        display_image(calibrated_0, "calibrated")

    calculate_metrics(calibrated_0, "calibrated", metrics)

    #grid_search(images)
    process_images(images)

if __name__ == '__main__':
    main()