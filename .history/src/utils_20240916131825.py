from venv import create
import matplotlib.pyplot as plt
from numpy import uint16
import os

def clear_folder(folder_path):
    for f in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, f))

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def progress(current, total, msg):
    if current == 1: print()
    print(f'\033[A{current}/{total} {msg}')
    if current == total: print()

def visualize_image(image, name='image'):
    # Print the image shape
    print(f'Image shape: {image.shape}')
    # Print the image dtype
    print(f'Image dtype: {image.dtype}')
    # Print the image min and max values
    print(f'Image min: {image.min()}')
    print(f'Image max: {image.max()}')
    # Print the image mean value
    print(f'Image mean: {image.mean()}')
    # Print the image standard deviation
    print(f'Image std: {image.std()}')

    # Normalize the image to [0, 1]
    if image.dtype == uint16:
        normalized_image = (image / 65535.0)  # Normalizza a [0, 1] per 16-bit
    else:
        normalized_image = image / 255.0  # Normalizza a [0, 1] per 8-bit
    plt.imshow(normalized_image)
    plt.axis('off')
    plt.show()