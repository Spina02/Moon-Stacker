# TODO List

### Open the input folder

  - [x] RAW images
  - [x] regular images

## Preprocessing

  - [x] Convert RAW and JPEG to 16 bit

### Image alignment

  - [x] Use ORB to detect features in astronomical images
  - [ ] Use SIFT alternatively to ORB
  - [x] Implement image alignment with geometric transformations (rotation, translation, scale) 
  - [x] RANSAC to eliminate outliers 
  - [x] `warpPerspective` from OpenCV to correctly align images

## Stacking (multiple algoritmhs)

  - [x] Implement median stacking to reduce noise
  - [x] Integrate sigma clipping techniques for advanced noise management
  - [ ] Implement Weighted Average Stacking

## Denoising

  - [ ] Implement basic denoising using traditional filters like Gaussian Blur
  - [ ] Integrate a convolutional neural network model for machine learning-based denoising
  - [ ] Add comparison between different denoising techniques (classical vs AI)

## Integration of Machine Learning for alignment

  - [ ] Create a neural network model to improve the alignment of astronomical images
  - [ ] Use a dataset of astronomical images to train the model to recognize and correct geometric transformations between images

## Output and visualization

  - [ ] Save the resulting images in TIFF format
  - [ ] Provide options to visualize intermediate results (aligned images, denoised images, stacked images) and comparisons between before and after

## Performance optimization

  - [ ] Optimize the code to reduce processing times (e.g., multi-threading, GPU with TensorFlow/PyTorch)

## Future extensions

  - [ ] Add functions for image analysis (e.g., quantitative comparisons of noise pre- and post-stacking, pixel distribution graphs)
  - [ ] Integrate a GUI to facilitate image input and visualization of final results