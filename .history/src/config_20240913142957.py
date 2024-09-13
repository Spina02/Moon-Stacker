MAX_IMG = 3
COLAB = False
DEBUG = 1 # 0: no debug, 1: print debug, 2: very verbose debug
IS_MOON = True
#DNCNN_MODEL_PATH = './models/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth'

input_folder = './images/jpg'
output_folder = './images/output'
output_format = 'png'

def config_init():
    global input_folder, output_folder, output_format, IS_MOON
    # select the input folder from user input
    cmd = input(f"Enter the input folder: (default : {input_folder}) ")
    if cmd:
        input_folder = cmd
    # select the output folder from user input
    cmd = input(f"Enter the output folder (default : {output_folder}): ")
    if cmd:
        output_folder = cmd
    # select the output format from user input
    cmd = input(f"Enter the output format (default : {output_format}): ")
    if cmd:
        output_format = cmd
    IS_MOON = input("Are the image of the moon? (Y/n): ").lower() != 'n'