import config
from image import read_image, save_image, read_images_generator, save_images, to_8bit
from preprocessing import crop_to_center
from process import process_images, calibrate_images, align_images
from metrics import calculate_metrics, init_metrics
import matplotlib.pyplot as plt
import numpy as np
import cv2
from calibration import calculate_masters
import os
import gc  # Importa il modulo per la garbage collection
from denoise import DnCNN, model_init, perform_denoising

def get_memory_usage():
    """Helper function to get current memory usage"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Return memory in MB

def analyze_calibration_effect(images_path, calibrated_path):
    print("\n=== Starting Calibration Effect Analysis ===")
    print(f"Current memory usage: {get_memory_usage():.2f} MB")

    # Process images without calibration
    print("Processing uncalibrated images...")
    images_generator = read_images_generator(images_path)
    images_list = []
    for i, img in enumerate(images_generator):
        images_list.append(img)
    
    print(f"Processing {len(images_list)} uncalibrated images...")
    final_image_no_calibration = process_images(images=images_list, save=False, evaluate=False)
    scores_no_calibration = calculate_metrics(final_image_no_calibration, 'no_calibration', config.metrics)
    print("Uncalibrated metrics:", scores_no_calibration)
    
    # Free memory
    print(f"Memory before cleanup: {get_memory_usage():.2f} MB")
    del images_list
    del final_image_no_calibration
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.2f} MB")
    
    # Process calibrated images
    calibrated_generator = read_images_generator(calibrated_path)
    calibrated_list = []
    for img in calibrated_generator:
        calibrated_list.append(img)
    final_image_with_calibration = process_images(images=calibrated_list, save=False, evaluate=False)
    scores_with_calibration = calculate_metrics(final_image_with_calibration, 'with_calibration', config.metrics)
    # Free memory
    del calibrated_list
    del final_image_with_calibration
    gc.collect()

    # Compare metrics
    metrics = list(config.metrics)
    values_no_calibration = [scores_no_calibration[metric] for metric in metrics]
    values_with_calibration = [scores_with_calibration[metric] for metric in metrics]

    # Create plot
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values_no_calibration, width, label='Without Calibration')
    rects2 = ax.bar(x + width/2, values_with_calibration, width, label='With Calibration')

    ax.set_ylabel('Values')
    ax.set_title('Effect of Calibration on Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.tight_layout()
    plt.savefig('./images/analysis/calibration_effect.png')
    plt.show()
    gc.collect()

def analyze_number_of_images():
    print("\n=== Starting Number of Images Analysis ===")
    print(f"Current memory usage: {get_memory_usage():.2f} MB")

    init_metrics()
    import gc
    max_images = 23  # Replace with maximum number of available images
    metrics_values = {metric: [] for metric in config.metrics}
    num_images_list = range(1, max_images + 1)

    for nimg in num_images_list:
        print(f"Processing stack with {nimg} images...")
        images_generator = read_images_generator('images/aligned', nimg)
        images_list = []
        for img in images_generator:
            images_list.append(img)
        params = {}
        final_image = process_images(images=None, aligned=images_list, save=False, evaluate=False)
        scores = calculate_metrics(final_image, f'image_{nimg}', config.metrics)
        
        # Collect all metrics
        for metric in config.metrics:
            metrics_values[metric].append(scores[metric])
            print(f"{metric.upper()} score for {nimg} images: {scores[metric]:.4f}")
        
        # Free memory
        del images_list
        del final_image
        gc.collect()
        print(f"Memory usage after processing {nimg} images: {get_memory_usage():.2f} MB")

    # Save values in separate files
    for metric in config.metrics:
        np.save(f'./images/analysis/{metric}_values.npy', metrics_values[metric])
        print(f"Saved {metric} values.")

    # Create plots for all metrics
    plt.figure(figsize=(10, 6))
    for metric in config.metrics:
        plt.plot(num_images_list, metrics_values[metric], marker='o', label=metric.upper())
        plt.xlabel('Number of Images in Stack')
        plt.ylabel(f'{metric} Value')
        plt.title('Impact of Number of Images on Final Quality')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./images/analysis/number_of_images_effect_{metric}.png')
        plt.close()
        print(f"Saved number of images effect plot for {metric}.")
    gc.collect()

def compare_stacking_algorithms(images_path):
    print("\n=== Starting Stacking Algorithms Comparison ===")
    print(f"Current memory usage: {get_memory_usage():.2f} MB")

    import gc

    print("analyzing calibration effect")

    stacking_algorithms = ['weighted average', 'median', 'sigma clipping']
    weighting_methods = ['sharpness', 'contrast', 'liqe']

    metrics_results = {}

    # Load aligned images using generator
    images_generator = read_images_generator(images_path)
    images_list = []
    for img in images_generator:
        images_list.append(img)

    for alg in stacking_algorithms:
        print(f"\nTesting stacking algorithm: {alg}")
        if (alg == 'weighted average'):
            for method in weighting_methods:
                print(f"Testing weighting method: {method}")
                params = {'stacking_alg': alg, 'average_alg': method}
                final_image = process_images(images=None, aligned=images_list, params=params, save=False, evaluate=False)
                scores = calculate_metrics(final_image, f'{alg}_{method}', config.metrics)
                metrics_results[f'{alg}_{method}'] = scores
                print(f"Scores for {alg}_{method}:", scores)
                # Free memory
                del final_image
                gc.collect()
        else:
            params = {'stacking_alg': alg}
            final_image = process_images(images=None, aligned=images_list, params=params, save=False, evaluate=False)
            scores = calculate_metrics(final_image, alg, config.metrics)
            metrics_results[alg] = scores
            print(f"Scores for {alg}:", scores)
            # Free memory
            del final_image
            gc.collect()

    # Free memory of aligned images
    del images_list
    gc.collect()

    # Create plots for each metric
    metrics_to_plot = list(config.metrics)
    for metric in metrics_to_plot:
        labels = list(metrics_results.keys())
        values = [metrics_results[label][metric] for label in labels]

        plt.figure(figsize=(10, 6))
        x = range(len(labels))
        plt.bar(x, values, align='center')
        plt.xticks(x, labels, rotation=45)
        plt.xlabel('Stacking Algorithms and Weighting Methods')
        plt.ylabel(f'{metric} Value')
        plt.title(f'Stacking Algorithms Comparison - {metric}')
        plt.tight_layout()
        plt.savefig(f'./images/analysis/stacking_comparison_{metric}.png')
        plt.show()
    gc.collect()

def generate_image_histograms(images_path):
    print("\n=== Generating Image Histograms ===")
    print(f"Current memory usage: {get_memory_usage():.2f} MB")
    import gc
    # Load original image (first image)
    original_image = read_image(os.path.join('./images/aligned', os.listdir('./images/aligned')[0]))
    # Load aligned images
    images_generator = read_images_generator(images_path)
    images_list = []
    for img in images_generator:
        images_list.append(img)

    params = {}
    final_image = process_images(images=None, aligned=images_list, params=params, save=False, evaluate=False)

    # Convert images to 8-bit grayscale
    original_gray = cv2.cvtColor(to_8bit(original_image), cv2.COLOR_BGR2GRAY)
    final_gray = cv2.cvtColor(to_8bit(final_image), cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist_original = cv2.calcHist([original_gray], [0], None, [256], [0, 256])
    hist_final = cv2.calcHist([final_gray], [0], None, [256], [0, 256])

    # Create plot
    plt.figure()
    plt.plot(hist_original, label='Original Image')
    plt.plot(hist_final, label='Final Image')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.title('Image Histograms')
    plt.legend()
    plt.savefig('./images/analysis/image_histograms.png')
    plt.show()

    # Free memory
    del images_list
    del final_image
    gc.collect()

def denoise(img, denoising_method):
    # Denoising
    if denoising_method == 'dncnn':
        model = model_init()
        denoised = perform_denoising(model, img)
        denoised = np.clip(denoised * 0.7 + img * (0.3), 0, 1)
    elif denoising_method == 'gaussian':
        denoised = cv2.GaussianBlur(img, (5, 5), 3)
    elif denoising_method == 'bilateral':
        denoised = cv2.bilateralFilter(img, 9, 120, 120)
    elif denoising_method == 'median':
        denoised = cv2.medianBlur(img, 5)
    else:
        raise ValueError(f"Unknown denoising method: {denoising_method}")
        
    return denoised


def compare_denoising_methods(images_path):
    print("\n=== Starting Denoising Methods Comparison ===")
    print(f"Current memory usage: {get_memory_usage():.2f} MB")

    import gc

    denoising_methods = ['gaussian', 'bilateral', 'median', 'dncnn']

    metrics_results_single = {}
    metrics_results = {}

    # Carica le immagini allineate
    images_generator = read_images_generator(images_path)
    images_list = []
    for img in images_generator:
        images_list.append(img)

    for method in denoising_methods:
        print(f"\nTesting denoising method: {method}")
        params = {}
        just_denoised = denoise(images_list[0], method)
        save_image(just_denoised, f"just_denoised_{method}", './images/analysis')
        
        scores = calculate_metrics(just_denoised, method, config.metrics)
        metrics_results_single[method] = scores

        final_image = process_images(images=None, aligned=images_list, params=params, save=False, evaluate=False, denoising_method=method)
        scores = calculate_metrics(final_image, method, config.metrics)
        metrics_results[method] = scores
        print(f"Scores for {method}:", scores)
        save_image(final_image, method, './images/analysis')
        # Libera la memoria
        del final_image
        gc.collect()

    # Libera la memoria delle immagini allineate
    del images_list
    gc.collect()

    # Crea i grafici per ogni metrica per denoise singolo
    metrics_to_plot = list(config.metrics)
    for metric in metrics_to_plot:
        labels = list(metrics_results_single.keys())
        values = [metrics_results_single[label][metric] for label in labels]

        plt.figure(figsize=(8, 6))
        x = range(len(labels))
        plt.bar(x, values, align='center')
        plt.xticks(x, labels)
        plt.xlabel('Metodi di Denoising')
        plt.ylabel(f'Valore {metric}')
        plt.title(f'Confronto Metodi di Denoising - {metric}')
        plt.tight_layout()
        plt.savefig(f'./images/analysis/denoising_comparison_{metric}.png')
        plt.show()
    gc.collect()

    # Crea i grafici per ogni metrica
    metrics_to_plot = list(config.metrics)
    for metric in metrics_to_plot:
        labels = list(metrics_results.keys())
        values = [metrics_results[label][metric] for label in labels]

        plt.figure(figsize=(8, 6))
        x = range(len(labels))
        plt.bar(x, values, align='center')
        plt.xticks(x, labels)
        plt.xlabel('Metodi di Denoising')
        plt.ylabel(f'Valore {metric}')
        plt.title(f'Confronto Metodi di Denoising - {metric}')
        plt.tight_layout()
        plt.savefig(f'./images/analysis/denoising_comparison_{metric}.png')
        plt.show()
    gc.collect()

def main():
    print("\n=== Starting Main Execution ===")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    import gc
    config.config_init(fast=True)

    # Check if aligned images already exist
    if os.path.exists('./images/aligned'):
        print("Using existing aligned images")
        images_path = './images/aligned'
    else:
        print("Need to process images...")
        # Check if calibrated images already exist
        if os.path.exists('./images/calibrated'):
            calibrated_path = './images/calibrated'
        else:
            # Calculate master bias, dark, flat
            bias = read_image('images/masters/bias.tif') if os.path.exists('images/masters/bias.tif') else None
            dark = read_image('images/masters/dark.tif') if os.path.exists('images/masters/dark.tif') else None
            flat = read_image('images/masters/flat.tif') if os.path.exists('images/masters/flat.tif') else None
            print("Calculating master calibration frames...")
            bias, dark, flat = calculate_masters(bias, dark, flat)

            # Read input images
            images_generator = read_images_generator(config.input_folder)
            images_list = []
            for img in images_generator:
                images_list.append(img)

            # Calibrate images
            print("Calibrating images...")
            calibrated = calibrate_images(images_list, bias, dark, flat)
            save_images(calibrated, 'calibrated', './images/calibrated')

            # Free memory
            del images_list
            del calibrated
            gc.collect()

        # Align calibrated images
        print("Aligning images...")
        calibrated_generator = read_images_generator('./images/calibrated')
        calibrated_list = []
        for img in calibrated_generator:
            calibrated_list.append(img)

        aligned = align_images(calibrated_list)
        save_images(aligned, 'aligned', './images/aligned')

        # Free memory
        del calibrated_list
        del aligned
        gc.collect()

        images_path = './images/aligned'

    #print("\n=== Starting Analysis Pipeline ===")
    #print("1. Analyzing calibration effect...")
    #analyze_calibration_effect(config.input_folder, './images/calibrated')
    
    #print("\n2. Analyzing number of images...")
    #analyze_number_of_images()
    
    #print("\n3. Comparing stacking algorithms...")
    #compare_stacking_algorithms(images_path)
    
    #print("\n4. Generating image histograms...")
    #generate_image_histograms(images_path)
    
    print("\n5. Comparing denoising methods...")
    compare_denoising_methods(images_path)
    
    print(f"\nFinal memory usage: {get_memory_usage():.2f} MB")
    gc.collect()

if __name__ == "__main__":
    main()