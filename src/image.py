import rawpy
import imageio

def read_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()
    return rgb

def save_image(file_path, image):
    imageio.imsave(file_path, image)

