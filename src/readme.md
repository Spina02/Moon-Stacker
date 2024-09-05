# Moon Stacker

This is my thesis project, focused on stacking multiple RAW images of the moon to produce higher quality and more detailed results. The project involves both classical image processing algorithms and modern machine learning techniques to enhance the final output.

## Features (WIP)

- **Image Alignment**: The program aligns the input images using feature detection algorithms (e.g., SIFT, ORB) and robust matching techniques such as RANSAC to ensure proper alignment despite rotation, scaling, or translation differences.
  
- **Denoising with Machine Learning**: Advanced neural networks, powered by PyTorch, are employed to reduce noise in the images while preserving important lunar details, ensuring clean and high-quality results.

- **Classical Stacking Algorithms**: After alignment and denoising, the images are combined using traditional stacking algorithms like median stacking and sigma clipping to enhance image clarity and reduce noise further.

- **RAW to JPEG Conversion**: The program supports RAW image input and processes the images internally in TIFF format, with the final output being saved in JPEG format.

## Future Goals

- Expansion to process other types of astronomical images, such as the Milky Way or deep space objects.
- Integration of a graphical user interface (GUI) to make the program more user-friendly.
- Performance optimization for faster image processing without requiring high-end hardware.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- rawpy
- imageio
- matplotlib (for visualization)