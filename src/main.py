import config
from image import read_image, read_images,display_image
from preprocessing import crop_to_center
from metrics import calculate_metrics, init_metrics
from calibration import calculate_masters, calibrate_images
from grid_search import grid_search
import os
from process import process_images
    
def main():

    init_metrics(['liqe'])

    bias = read_image('images/masters/bias.tif') if os.path.exists('images/masters/bias.tif') else None
    dark = read_image('images/masters/dark.tif') if os.path.exists('images/masters/dark.tif') else None
    flat = read_image('images/masters/flat.tif') if os.path.exists('images/masters/flat.tif') else None
    bias, dark, flat = calculate_masters(bias, dark, flat)
    
    images = read_images(config.input_folder)

    image_0 = crop_to_center([images[0]])[0]
    if config.COLAB:
        display_image(image_0, "image 0")
    
    calculate_metrics(image_0, "image 0", config.metrics)

    images = calibrate_images(images, bias, dark, flat)

    calibrated_0 = crop_to_center([images[0]])[0]
    if config.COLAB:
        display_image(calibrated_0, "calibrated")

    calculate_metrics(calibrated_0, "calibrated", config.metrics)

    if config.GRID_SEARCH:
        params = grid_search(images, save=True)
        process_images(images, params = params, save=True)
    else:
        process_images(images, save=True)

if __name__ == '__main__':
    main()