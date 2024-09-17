from image import *
from preprocessing import align_images
import cv2
import numpy as np

def equalize_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:  # Color image

        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def snr_compare(image, reference):
    # check if dtype is the same
    if image.dtype != reference.dtype:
        print(f'Image and reference have different dtypes: {image.dtype} and {reference.dtype}')
        exit(0)

    # Ensure the images have the same size
    reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    images = [image, reference]
    aligned_img = align_images(images, algo='orb', nfeatures=10000)
    reference = aligned_img[1]

    save_image(reference, 'images/extra/aligned_reference', 'png')
    #save_image(image, 'images/extra/aligned_image', 'png')

    # Calculate the signal (reference image)
    signal = np.mean(reference ** 2)

    # Calculate the noise (difference between the images)
    noise = np.mean((image - reference) ** 2)

    # Calculate SNR
    snr = 10 * np.log10(signal / noise)
    return snr

def remove_bg(image):
    # find bg color in first pixel
    bg_color = image[0][0]
    # remove bg color
    image[image <= bg_color + 5] = 0
    return image

def main():
    image = read_image('images/output/orb_median_5000.png')
    reference = read_image('images/extra/out.tif')
    #8-bit
    reference = remove_bg(to_8bit(reference))
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    
    # Equalize the histogram of the image
    #image = equalize_histogram(image)
    #reference = equalize_histogram(reference)

    save_image(image, 'images/extra/equalized_image', 'png')
    save_image(reference, 'images/extra/equalized_reference', 'png')

    snr = snr_compare(image, reference)
    print(f'SNR: {snr} dB')

if __name__ == '__main__':
    main()