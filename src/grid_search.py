import itertools
import config
from image import save_image, display_image
from align import enhance_contrast
from enhancement import unsharp_mask
from metrics import calculate_metrics
from process import stack_images, align_images, custom_unsharp_mask

def grid_search(images, aligned = None, save=True):
    print("Starting grid search")

    scores = {}
    params = {}
    if len(config.metrics) == 1:
      if config.metrics[0] == 'liqe':
          best_score = float('-inf')
      else:
          best_score = float('inf')

    features_alg = 'orb'
    n_features = 15000
    
    if aligned is None:
        print(f'Aligning images with {features_alg} using {n_features} features')
        
        aligned = align_images(images, algo=features_alg, nfeatures=n_features)

    just_stacked = stack_images(aligned)
    if save:
        print(f'Saving just stacked image')
        save_image(just_stacked, "just stacked", 'images/output')

    # Visualize the image if running on Google Colab
    if config.COLAB:
        display_image(just_stacked, "just stacked")
    
    calculate_metrics(just_stacked, 'just_stacked', config.metrics)

    # extract parameters
    stacking_algorithms = config.gs_params['stacking_algorithms']
    average_algs = config.gs_params['average_algs']
    gradient_strengths = config.gs_params['gradient_strengths']
    gradient_thresholds = config.gs_params['gradient_thresholds']
    denoise_strengths = config.gs_params['denoise_strengths']
    unsharp_strengths = config.gs_params['unsharp_strengths']
    tile_sizes = config.gs_params['tile_sizes']
    clip_limits = config.gs_params['clip_limits']

    # Iterate over all possible combinations of parameters
    for gradient_strength, gradient_threshold, denoise_strength in itertools.product(
        gradient_strengths, 
        gradient_thresholds, 
        denoise_strengths
    ):
        # Preprocess the images
        denoised = custom_unsharp_mask(aligned, gradient_strength=gradient_strength, gradient_threshold=gradient_threshold, denoise_strength = denoise_strength)

        #if save:
        #    save_image(denoised[0], "denoised")

        for stacking_alg in stacking_algorithms:
            for average_alg in average_algs:
                print(f'\nRunning {features_alg} with gradient strength {gradient_strength}, gradient threshold {gradient_threshold}, denoise strength {denoise_strength} and stacking {stacking_alg} (avg = {average_alg})')
                name = f'{features_alg}_str{gradient_strength}_thr{gradient_threshold}_dstr{denoise_strength}'

                stacked_image = stack_images(denoised, stacking_alg=stacking_alg, average_alg=average_alg)

                for strength, ker, limit in itertools.product(
                    unsharp_strengths, 
                    tile_sizes, 
                    clip_limits
                ):
                    new_name = f"{name}_ush{strength}_ker{ker}_clip{limit}_avg{average_alg}"
                    if config.DEBUG: print(f'\nEnhancing image with unsharp strength {strength}, kernel size {ker}, clip limit {limit}')
                    
                    # Apply unsharp mask and enhance contrast
                    enhanced_image = unsharp_mask(stacked_image, strength)
                    final_image = enhance_contrast(enhanced_image, clip_limit=limit, tile_grid_size=ker)

                    params[new_name] = {
                                'gradient_strength': gradient_strength,
                                'gradient_threshold': gradient_threshold,
                                'denoise_strength': denoise_strength,
                                'stacking_alg': stacking_alg,
                                'average_alg': average_alg,
                                'unsharp_strength': strength,
                                'tile_size': ker,
                                'clip_limit': limit
                            }
                    scores[new_name] = calculate_metrics(final_image, new_name, config.metrics)
                    if len(config.metrics) == 1:
                        if (config.metrics[0] == 'liqe' and scores[new_name][config.metrics[0]] > best_score) or (config.metrics[0] != 'liqe' and scores[new_name][config.metrics[0]] < best_score):
                            best_score = scores[new_name][config.metrics[0]]
                            if save:
                                print(f'Saving {new_name}')
                                save_image(final_image, new_name, 'images/output')

                            # Visualize the image if running on Google Colab
                            if config.COLAB:
                                display_image(final_image, new_name)

    print('Grid search completed')
    # print scores
    for name, score in scores.items():
        print(f'{name}: {score}')

    best = []
    
    for metric in config.metrics:
        if metric == 'liqe':
            best = list(max(scores.items(), key=lambda x: x[1][metric]))
        else:
            best = list(min(scores.items(), key=lambda x: x[1][metric]))
        print(f'Best {metric} score:\n\t {best[0]}\nwith a score of \t{best[1][metric]}')

    if len(config.metrics) == 1:
        return params[best[0]]

