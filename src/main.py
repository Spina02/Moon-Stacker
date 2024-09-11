import config
from grid_search import grid_search

def main():
    # TODO: add a way to select the output format
    # TODO: add a way to select aligning algorithm
    # TODO: add a way to select stacking algorithm

    # select the input folder from user input
    cmd = input(f"Enter the input folder: (default : {config.input_folder}) ")
    if cmd:
        config.input_folder = cmd
    # select the output folder from user input
    cmd = input(f"Enter the output folder (default : {config.output_folder}): ")
    if cmd:
        config.output_folder = cmd
    # select the output format from user input
    cmd = input(f"Enter the output format (default : {config.output_format}): ")
    if cmd:
        config.output_format = cmd
    config.IS_MOON = input("Are the image of the moon? (Y/n): ").lower() != 'n'
    
    # grid search
    grid_search(crop = config.IS_MOON)
     
if __name__ == '__main__':
    import main
    main.main()