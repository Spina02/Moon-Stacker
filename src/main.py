from image import *

format = 'png' #'tiff'
MAX_IMG = 10

def main():
    folder_path = './images/jpg'
    tiff_folder = './images/tiff'
    output_path = './images/output'

    # TODO: add a way to select the output format

    # Read and convert RAW images to TIFF format in memory
    images = read_and_convert_images(folder_path)

    #TODO: Process the images
    # processed_image = process_images(images)

    # Save the processed image
    # Salva le immagini elaborate

    print("\n")
    # empty the output folder
    for f in os.listdir(output_path):
        os.remove(os.path.join(output_path, f))
    for i, image in enumerate(images):
        save_image(f'{output_path}/output_{i}', image, format)
        print(f'\033[A{i+1}/{len(images) + 1} images saved')

if __name__ == '__main__':
    import main
    main.main()