import itertools
import config
from image import save_image, display_image
from align import enhance_contrast
from preprocessing import unsharp_mask
from metrics import calculate_metrics
from process import stack_images, align_images, dncnn_unsharp_mask

def grid_search(images, save=False):
    print("Starting grid search")

    #best_score = float('-inf')
    #best_img = ''
    #best_params = {}

    scores = {}
    metrics = ['niqe', 'piqe', 'liqe', 'nima', 'brisque_matlab']

    features_alg = 'orb'
    n_features = 10000
    print(f'Aligning images with {features_alg} using {n_features} features')
    
    aligned = align_images(images, algo=features_alg, nfeatures=n_features)

    just_stacked = stack_images(aligned)
    calculate_metrics(just_stacked, 'just_stacked', metrics)

    # Grid search parameters
    stacking_algorithms = ['weighted average']#, "sigma clipping", "median"]
    average_algs = ['brisque']
    gradient_strengths = [1.0, 1.25]
    gradient_thresholds = [0.0075, 0.008]
    denoise_strengths = [0.5, 0.8]
    unsharp_strengths = [1.5]
    kernel_sizes = [(15, 15)]
    clip_limits = [0.65]

    # Iterate over all possible combinations of parameters
    for gradient_strength, gradient_threshold, denoise_strength in itertools.product(
        gradient_strengths, 
        gradient_thresholds, 
        denoise_strengths
    ):
        # Preprocess the images
        denoised = dncnn_unsharp_mask(aligned, gradient_strength=gradient_strength, gradient_threshold=gradient_threshold, denoise_strength = denoise_strength)

        if save:
            save_image(denoised[0], "denoised")

        for stacking_alg in stacking_algorithms:
            for average_alg in average_algs:
                print(f'\nRunning {features_alg} with gradient strength {gradient_strength}, gradient threshold {gradient_threshold}, denoise strength {denoise_strength} and stacking {stacking_alg} (avg = {average_alg})')
                name = f'{features_alg}_str{gradient_strength}_thr{gradient_threshold}_dstr{denoise_strength}'

                stacked_image = stack_images(denoised, stacking_alg=stacking_alg, average_alg=average_alg)

                for strength, ker, limit in itertools.product(
                    unsharp_strengths, 
                    kernel_sizes, 
                    clip_limits
                ):
                    new_name = f"{name}_ush{strength}_ker{ker}_clip{limit}_avg{average_alg}"
                    print(f'Enhancing image with unsharp strength {strength}, kernel size {ker}, clip limit {limit}')
                    
                    # Apply unsharp mask and enhance contrast
                    enhanced_image = unsharp_mask(stacked_image, strength)
                    final_image = enhance_contrast(enhanced_image, clip_limit=limit, tile_grid_size=ker)

                    if save:
                        print(f'Saving {new_name}')
                        save_image(final_image, new_name, 'images/output')

                    # Visualize the image if running on Google Colab
                    if config.COLAB:
                        display_image(final_image, new_name)
                    scores[new_name] = calculate_metrics(final_image, new_name, metrics)
                    

    print('Grid search completed')
    # print scores
    for name, metrics in scores.items():
        print(f'{name}: {metrics}')

    #print(f'Best score: {best_score} at {best_img}')
    #return best_params, aligned
