import torch
import pyiqa

MAX_IMG = 50
MAX_CALIBRATION = 50
MIN_CALIBRATION = 10
COLAB = True
DEBUG = True
GRID_SEARCH = False

input_folder = './images/lights'
bias_folder = './images/bias'
dark_folder = './images/darks'
flat_folder = './images/flats'
masters_folder = './images/masters'
output_folder = './images/output'
output_format = 'png'

iqa_metrics = {}
metrics = ['liqe'] # 'niqe_matlab', 'brisque_matlab'

gs_params = {
    'stacking_algorithms': ['weighted average'],
    'average_algs': ['sharpness'],
    'gradient_strengths': [1, 1.3],
    'gradient_thresholds': [0.009, 0.01],
    'denoise_strengths': [0.9, 1, 1.5],
    'unsharp_strengths': [2.25, 2.5],
    'kernel_sizes': [(17, 17), (19, 19)],
    'clip_limits': [0.7]
}

def init_metrics(metrics = metrics):
    global iqa_metrics
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for metric in metrics:
        iqa_metrics[metric] = pyiqa.create_metric(metric, device = device)

def config_init():
    global COLAB, DEBUG, GRID_SEARCH, input_folder, bias_folder, dark_folder, flat_folder, masters_folder, output_folder, output_format, metrics

    
    cmd = input("Are you running on Google Colab? (y/N) ")
    COLAB = True if cmd.lower() == 'y' else False

    cmd = input("Do you want to enable debug mode? (y/N) ")
    DEBUG = True if cmd.lower() == 'y' else False

    cmd = input("Do you want to enable grid search? (y/N) ")
    if cmd.lower() == 'y':
        GRID_SEARCH = True
        #cmd = input("Do you want to change the default metrics? (y/N) ")
        #if cmd.lower() == 'y':
            #print("Allowed metrics : 'liqe', 'niqe_matlab', 'brisque_matlab'")
            #input_metrics = input(f"Enter the metrics to be calculated separated by a space\n(default: '{' '.join(metrics)}'): ").split() or metrics

        # get parameters for grid search
        check = input("Do you want to change the default grid search parameters? (y/N) ")
        if check.lower() == 'y':
            for key in gs_params:
                gs_params[key] = input(f"Enter the values for {key} separated by a space\n(default: '{' '.join(map(str, gs_params[key]))}'): ").split() or gs_params[key]

    check = input("Do you want to change the default paths? (y/N) ")
    if check.lower() == 'y':
        input_folder = input(f"Enter the path to the folder containing the lights images\n(default: '{input_folder}'): ") or input_folder
        bias_folder = input(f"Enter the path to the folder containing the bias images\n(default: '{bias_folder}'): ") or bias_folder
        dark_folder = input(f"Enter the path to the folder containing the dark images\n(default: '{dark_folder}'): ") or dark_folder
        flat_folder = input(f"Enter the path to the folder containing the flat images\n(default: '{flat_folder}'): ") or flat_folder
        masters_folder = input(f"Enter the path to the folder containing the master images\n(default: '{masters_folder}'): ") or masters_folder
        output_folder = input(f"Enter the path to the folder where the output images will be saved\n(default: '{output_folder}'): ") or output_folder

    output_format = input(f"Enter the output image format\n(default: '{output_format}'): ") or output_format
    
    init_metrics()
