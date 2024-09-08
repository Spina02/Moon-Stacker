from os import cpu_count
import cv2
import numpy as np
from const import *
from debug import progress
from multiprocessing import Pool
import time

# ORB algorithm with CUDA
def orb_cuda(image, nfeatures=500):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Upload the image to the GPU
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(gray)
    # Initialize the ORB detector
    orb = cv2.cuda_ORB.create(nfeatures=nfeatures)
    # Detect key points and descriptors
    kp, des = orb.detectAndCompute(gpu_image, None)
    # Download descriptors from GPU to CPU
    des = des.download()
    # Convert keypoints to CPU format
    kp = [cv2.KeyPoint(x.pt[0], x.pt[1], x.size, x.angle, x.response, x.octave, x.class_id) for x in kp]
    return kp, des

# SURF algorithm with CUDA
def surf_cuda(image, hessian_threshold=400):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Upload the image to the GPU
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(gray)
    # Initialize the SURF detector
    surf = cv2.cuda_SURF.create(hessian_threshold)
    # Detect key points and descriptors
    kp, des = surf.detectAndCompute(gpu_image, None)
    # Download descriptors from GPU to CPU
    des = des.download()
    # Convert keypoints to CPU format
    kp = [cv2.KeyPoint(x.pt[0], x.pt[1], x.size, x.angle, x.response, x.octave, x.class_id) for x in kp]
    return kp, des

# SIFT algorithm with CUDA
def sift_cuda(image, nfeatures=500):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Upload the image to the GPU
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(gray)
    # Initialize the SIFT detector
    sift = cv2.cuda_SIFT.create(nfeatures=nfeatures)
    # Detect key points and descriptors
    kp, des = sift.detectAndCompute(gpu_image, None)
    # Download descriptors from GPU to CPU
    des = des.download()
    # Convert keypoints to CPU format
    kp = [cv2.KeyPoint(x.pt[0], x.pt[1], x.size, x.angle, x.response, x.octave, x.class_id) for x in kp]
    return kp, des

def align_image(image, ref_kp_coords, ref_des, ref_image, algo, i, total, nfeatures=500):
    try:
        if algo == 'orb':
            kp, des = orb_cuda(image, nfeatures)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif algo == 'sift':
            kp, des = sift_cuda(image, nfeatures)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif algo == 'surf':
            kp, des = surf_cuda(image)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Ensure descriptors are not None
        if des is None or ref_des is None:
            return None
        
        # Match the descriptors
        matches = matcher.match(ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        ref_pts = np.float32([ref_kp_coords[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        img_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0)
        height, width = ref_image.shape[:2]
        aligned_img = cv2.warpPerspective(image, H, (width, height))
        progress(i, total, 'images aligned')
        return aligned_img
    
    except Exception as e:
        return None
    

def align_images_multiprocess(images, algo='orb', nfeatures=500):
    ref_image = images[0]

    if algo == 'orb':
        ref_kp, ref_des = orb_cuda(ref_image, nfeatures)
    elif algo == 'sift':
        ref_kp, ref_des = sift_cuda(ref_image, nfeatures)
    elif algo == 'surf':
        ref_kp, ref_des = surf_cuda(ref_image)
        
    ref_kp_coords = [kp.pt for kp in ref_kp]

    n_cpu = cpu_count()
    print(f'Using {n_cpu} cores')

    start_time = time.time()
    aligned_images = [ref_image]

    with Pool(n_cpu) as pool:
        aligned_images += pool.starmap(align_image, [(image, ref_kp_coords, ref_des, ref_image, algo, i, len(images), nfeatures) for i, image in enumerate(images[1:])])

    end_time = time.time()

    print(f'Multiprocessing alignment took {end_time - start_time:.2f} seconds')
    return aligned_images