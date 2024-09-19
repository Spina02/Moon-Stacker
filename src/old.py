
def unsharp_mask_2(images, model, strength, ksize=3, sigma = 0, tale = (9, 9), low_clip = 0.01, high_clip = 0.5):

    images = unsharp_mask(images, strength)
    sobel_masks = sobel_images(images, ksize=ksize, sigma = sigma, tale = tale, low_clip = low_clip, high_clip = high_clip)
    blurred_images = dncnn_images(model, images)
    merged_images = []

    i = 1
    for image, blurred_image, sobel_mask in zip(images, blurred_images, sobel_masks):
        
        # Convert everything to float32
        sharp_image = image.astype(np.float32)

        if i == 0:
            save_images([to_16bit(sharp_image)], './images/sharpened', name='sharp_pre', clear = False)
            save_images([to_16bit(blurred_image)], './images/blurred', name='blurred_pre', clear = False)

        sobel_mask = normalize(sobel_mask)

        # Apply sharpening with Sobel mask modulation
        sharp_component = sharp_image * (0.5 + normalize(sobel_mask * strength))
        sharp_component = normalize(sharp_component)
        sharp_component = np.clip(sharp_component, 0, 1)  # Clip to valid range

        # Apply blurring with inverted Sobel mask modulation
        blurred_component = blurred_image * normalize(0.5 - sobel_mask * strength)
        blurred_component = normalize(blurred_component)
        blurred_component = np.clip(blurred_component, 0, 1)  # Clip to valid range

        if i == 0:
            save_images([to_16bit(sharp_component)], './images/merged', name='sharp', clear = False)
            save_images([to_16bit(blurred_component)], './images/merged', name='blurred', clear = False)
            i += 1

        # Merge the components
        merged_image = cv2.addWeighted(sharp_component, 0.6, blurred_component, 0.4, 0)
        merged_image = np.clip(merged_image, 0, 1)  # Ensure valid range

        # Convert back to 16-bit
        merged_image = to_16bit(merged_image)
        merged_image = white_balance(merged_image)
        #merged_image = enhance_contrast(merged_image)
        merged_images.append(merged_image)

    del blurred_images, sobel_masks
    gc.collect()

    #save_images(merged_images[:1], './images/merged', name='merged', clear = False)

    return merged_images

def unsharp_mask(images, strength):
    
    #blurred_images = dncnn_images(model, images)
    blurred_images = [cv2.GaussianBlur(image, (3, 3), 3) for image in images]

    #save_images(blurred_images, './images/blurred', name = 'blurred')

    merged_images = [to_16bit(cv2.addWeighted(to_16bit(image), 0.5 + strength, to_16bit(blurred_image), 0.5 -strength, 0)) for image, blurred_image in zip(images, blurred_images)]
    #blurred_images = dncnn_images(model, images)
    blurred_images = [cv2.GaussianBlur(image, (3, 3), 3) for image in images]

    #save_images(blurred_images, './images/blurred', name = 'blurred')

    merged_images = [to_16bit(cv2.addWeighted(to_16bit(image), 0.5 + strength, to_16bit(blurred_image), 0.5 -strength, 0)) for image, blurred_image in zip(images, blurred_images)]
    
    save_images(merged_images, './images/merged', name = 'merged', clear = False)
    
    del blurred_images
    gc.collect()
    return merged_images
