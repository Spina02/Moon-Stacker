import itertools
import config
from image import save_image, display_image, print_score
from align import enhance_contrast
from preprocessing import preprocess_images, unsharp_mask
from stacking import weighted_average_stack, median_stack, sigma_clipping
from metrics import calculate_brisque, calculate_ssim, combined_score, get_min_brisque

def grid_search(images, save=False):
    print("Starting grid search")

    best_score = float('-inf')
    best_img = ''
    best_params = {}

    features_alg = 'orb'
    n_features = 10000
    print(f'Aligning images with {features_alg} using {n_features} features')
    
    # Aligning images with default ORB and 50000 features
    aligned = preprocess_images(
        images, 
        algo=features_alg, 
        nfeatures=n_features, 
        grayscale=False, 
        unsharp=False
    )

    # Get the image with the minimum BRISQUE score
    ref, brisque = get_min_brisque(aligned)
    if save:
        save_image(ref, "best_brisque")
    if config.COLAB:
        display_image(ref, brisque, name = 'best brisque')
    else:
        print_score(brisque, name = 'best brisque')


    # Grid search parameters
    stacking_algorithms = ['weighted average']#, "sigma clipping", "median"]
    average_algs = ['brisque']
    gradient_strengths = [1.8]
    gradient_thresholds = [0.0075]
    denoise_strengths = [0.8]
    unsharp_strengths = [1.3]
    kernel_sizes = [(13, 13)]
    clip_limits = [0.65]

    # Iterate over all possible combinations of parameters
    for gradient_strength, gradient_threshold, denoise_strength in itertools.product(
        gradient_strengths, 
        gradient_thresholds, 
        denoise_strengths
    ):
        # Preprocess the images
        denoised = preprocess_images(
            aligned, 
            align=False, 
            crop=False, 
            grayscale=True, 
            unsharp=True,
            calibrate=False, 
            gradient_strength=gradient_strength,
            gradient_threshold=gradient_threshold, 
            denoise_strength=denoise_strength,
        )
        if save:
            save_image(denoised[0], "denoised")

        for stacking_alg in stacking_algorithms:
            for average_alg in average_algs:
                print(f'\nRunning {features_alg} with gradient strength {gradient_strength}, gradient threshold {gradient_threshold}, denoise strength {denoise_strength} and stacking {stacking_alg} (avg = {average_alg})')
                name = f'{features_alg}_str{gradient_strength}_thr{gradient_threshold}_dstr{denoise_strength}'

                # Stack the images using the selected stacking algorithm
                if stacking_alg == 'weighted average':
                    image = weighted_average_stack(denoised, method=average_alg)
                elif stacking_alg == 'median':
                    image = median_stack(denoised)
                elif stacking_alg == 'sigma clipping':
                    image = sigma_clipping(denoised)

                for strength, ker, limit in itertools.product(
                    unsharp_strengths, 
                    kernel_sizes, 
                    clip_limits
                ):
                    new_name = f"{name}_ush{strength}_ker{ker}_clip{limit}_avg{average_alg}"
                    print(f'Enhancing image with unsharp strength {strength}, kernel size {ker}, clip limit {limit}')
                    
                    # Apply unsharp mask and enhance contrast
                    enhanced_image = unsharp_mask(image, strength)
                    contrasted_image = enhance_contrast(enhanced_image, clip_limit=limit, tile_grid_size=ker)

                    if save:
                        print(f'Saving {new_name}')
                        save_image(contrasted_image, new_name, 'images/output')

                    # Calculate BRISQUE and SSIM scores
                    brisque = calculate_brisque(contrasted_image)
                    ssim = calculate_ssim(ref, contrasted_image)
                    score = combined_score(brisque, ssim)
                    print(f'BRISQUE score: {brisque:.4f}')
                    print(f'SSIM score: {ssim:.4f}')
                    print(f'Combined Score: {score:.4f}')

                    # Visualize the image if running on Google Colab
                    if config.COLAB:
                        display_image(contrasted_image, brisque, ssim, score, new_name)
                    else:
                        print_score(brisque, ssim, score, new_name)

                    # Update the best score found
                    if score > best_score:
                        best_score = score
                        best_img = new_name
                        best_params = {
                            "gradient_strength": gradient_strength,
                            "gradient_threshold": gradient_threshold,
                            "denoise_strength": denoise_strength,  
                            "stacking_alg": stacking_alg,
                            "average_alg": average_alg,
                            "strength": strength,                                                     
                            "ker": ker,
                            "limit": limit
                        }

    print(f'Best score: {best_score} at {best_img}')
    return best_params, aligned
