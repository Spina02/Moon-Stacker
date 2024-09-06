from email.mime import image
import rawpy
import imageio
import os
import numpy as np

MAX_IMG = 10

def read_folder(folder_path):
    image_paths = []
    i = 0
    # Implement the function to read all image paths in the folder
    for f in os.listdir(folder_path):
        i += 1
        if i >= MAX_IMG:
            break
        if not (folder_path.endswith('raw') and not f.endswith('.RAF')):
            image_paths.append(os.path.join(folder_path, f))
    return image_paths

def read_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
    return rgb

def read_image(file_path):
    return imageio.imread(file_path)

def read_and_convert_images(folder_path):
    image_paths = read_folder(folder_path)
    images = []
    i = 0
    print("\n")
    for path in image_paths:
        if path.lower().endswith(('.raf', '.dng', '.nef')):
            image = read_raw_image(path)
        else:
            image = read_image(path)
        # Normalize image data to 16-bit range if necessary
        if image.dtype != 'uint16':
            image = (image / image.max() * 65535).astype('uint16')
        images.append(image)
        i += 1
        print(f'\033[A{i}/{len(image_paths) + 1} images converted')

    return images

def process_images(images):
    # Implement your image processing pipeline here
    # aligned_images = align_images(images)
    # stacked_image = stack_images(aligned_images)
    # denoised_image = denoise_image(stacked_image)
    return images[0]  # Placeholder: return the first image for now

def save_image(file_path, image, format='tiff'):
    # Convert to 8-bit if the format is not TIFF
    if format.lower() not in ['tiff']:
        image = (image / 256).astype('uint8')
    else:
        image = image.astype('uint16')
    if not file_path.endswith(f'.{format.lower()}'):
        file_path += f'.{format.lower()}'
    imageio.imsave(file_path, image, format=format)