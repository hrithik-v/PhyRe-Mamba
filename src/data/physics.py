import numpy as np
import cv2
import torch

def get_dark_channel(image, window_size=15):
    """
    Computes the dark channel prior of an image.
    Args:
        image: Numpy array of shape (H, W, 3) or (H, W), values in [0, 1].
        window_size: Size of the sliding window (must be odd).
    Returns:
        dark_channel: Min filter result of shape (H, W).
    """
    # Ensure image is float [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # If RGB, take min across channels first
    if image.ndim == 3:
        min_channel = np.min(image, axis=2)
    else:
        min_channel = image
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """
    Estimates the multichannels atmospheric light A.
    Args:
        image: Original RGB image (H, W, 3).
        dark_channel: Computed dark channel (H, W).
    Returns:
        A: Atmospheric light vector (3,).
    """
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = max(int(num_pixels * 0.001), 1)
    
    flat_dark = dark_channel.reshape(-1)
    flat_img = image.reshape(-1, 3)
    
    # Indices of brightest pixels in dark channel
    indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
    
    # Average of these pixels in original image
    brightest_pixels = flat_img[indices]
    A = np.mean(brightest_pixels, axis=0)
    
    return A

def estimate_transmission(image, A, window_size=15, omega=0.95):
    """
    Estimates the transmission map t(x).
    t(x) = 1 - omega * min_c( min_y( I_c(y) / A_c ) )
    """
    norm_image = image / (A + 1e-6) # Avoid div by zero
    transmission = 1 - omega * get_dark_channel(norm_image, window_size)
    return np.clip(transmission, 0.1, 1.0) # Clip lower bound to avoid division by zero later

def estimate_depth_proxy(transmission):
    """
    Estimates a proxy for depth from transmission.
    According to Beer-Lambert law: t(x) = e^(-beta * d(x))
    So, d(x) is proportional to -ln(t(x)).
    Args:
        transmission: Estimated transmission map (H, W).
    Returns:
        depth: Normalized depth map (H, W) in [0, 1].
    """
    depth = -np.log(transmission + 1e-6)
    
    # Normalize to [0, 1] for network input
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
    
    return depth_norm

def compute_physics_maps(image_np):
    """
    Pipeline to compute Transmission and Depth maps.
    Args:
        image_np: Input image as numpy array (H, W, 3) in [0, 1].
    Returns:
        transmission (H, W), depth (H, W)
    """
    dark = get_dark_channel(image_np)
    A = estimate_atmospheric_light(image_np, dark)
    transmission = estimate_transmission(image_np, A)
    depth = estimate_depth_proxy(transmission)
    return transmission, depth
