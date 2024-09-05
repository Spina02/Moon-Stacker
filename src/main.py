import image

def main():
    file_path = ''
    image_path = ''

    image.save_image(image_path, image.read_raw_image(file_path))

if __name__ == '__main__':
    import main
    main.main()