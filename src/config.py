MAX_IMG = 20
MAX_CALIBRATION = 50
MIN_CALIBRATION = 10
COLAB = False
DEBUG = True
DNCNN_MODEL_PATH = './dncnn/logs/DnCNN-S-25/net.pth'

input_folder = './images/lights'
bias_folder = './images/bias'
dark_folder = './images/darks'
flat_folder = './images/flats'
masters_folder = './images/masters'
output_folder = './images/output'
output_format = 'png'

metrics = ['niqe_matlab', 'brisque_matlab', 'liqe_mix']

def config_init():
    global input_folder, output_folder, output_format
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