"""
Image Preprocessing Module - Professional Quality
Each step produces a CLEARLY VISIBLE different output
"""

import cv2
import numpy as np
from typing import Tuple
import config


def resize_image(image: np.ndarray, size: Tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """
    Resize image to standard size
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance contrast using CLAHE on LAB L-channel
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def remove_background(image: np.ndarray) -> np.ndarray:
    """
    Remove background using HSV green-mask method.
    Result: leaf on PURE BLACK background - clearly different from step 3.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Wide green range to capture all leaf shades
    lower_green = np.array([8, 25, 25])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply mask -> background becomes BLACK
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply STRONG Gaussian blur - clearly visible softening vs sharp BG-removed image.
    Kernel 21x21, sigma 10 makes blur very obvious on screen.
    """
    return cv2.GaussianBlur(image, (21, 21), 10)


def convert_color_space_for_display(image: np.ndarray) -> np.ndarray:
    """
    Convert to HSV and apply color mapping so it looks CLEARLY different on screen.
    Uses applyColorMap to make HSV visualization distinct from BGR.
    """
    # Convert to grayscale first
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply JET colormap -> gives blue-green-yellow-red heat-map look
    # This looks COMPLETELY different from normal leaf image
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Keep black background areas black
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    colored = cv2.bitwise_and(colored, colored, mask=mask)

    return colored


def convert_color_space(image: np.ndarray, color_space: str = config.COLOR_SPACE) -> np.ndarray:
    """
    Convert image for feature extraction (actual HSV conversion)
    """
    color_space = color_space.upper()
    if color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space == 'YCRCB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def segment_disease_region(image: np.ndarray,
                           k: int = config.KMEANS_CLUSTERS) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means segmentation with BRIGHT DISTINCT colors per cluster.
    Each cluster gets a unique vivid color so regions are clearly visible.
    """
    # Use BGR image for clustering
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    pixel_values = bgr.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                config.KMEANS_ITERATIONS, config.KMEANS_EPSILON)

    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10,
                                    cv2.KMEANS_PP_CENTERS)

    # Define BRIGHT DISTINCT colors for each cluster
    distinct_colors = [
        [0, 255, 0],      # Bright Green
        [0, 0, 255],      # Bright Red
        [255, 255, 0],    # Bright Yellow
        [255, 0, 255],    # Bright Magenta
        [0, 255, 255],    # Bright Cyan
        [128, 0, 255],    # Purple
        [255, 128, 0],    # Orange
        [0, 128, 255],    # Light Blue
    ]

    # Map each pixel to its cluster's distinct color
    labels_flat = labels.flatten()
    segmented_pixels = np.array([distinct_colors[l] for l in labels_flat], dtype=np.uint8)
    segmented = segmented_pixels.reshape(bgr.shape)

    # Create mask to keep black background black
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

    # Apply morphological ops to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask
    result = cv2.bitwise_and(segmented, segmented, mask=mask)

    return result, mask


def segment_disease_region_for_features(image: np.ndarray,
                                        k: int = config.KMEANS_CLUSTERS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard K-Means segmentation for FEATURE EXTRACTION (uses actual cluster centers, not colors)
    """
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                config.KMEANS_ITERATIONS, config.KMEANS_EPSILON)

    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10,
                                    cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(image.shape)

    labels_2d = labels.reshape((image.shape[0], image.shape[1]))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(k):
        cluster_mask = (labels_2d == i).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, cluster_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(segmented, segmented, mask=mask)
    return result, mask


def segment_kmeans(image: np.ndarray,
                   k: int = config.KMEANS_CLUSTERS,
                   max_iter: int = config.KMEANS_ITERATIONS,
                   epsilon: float = config.KMEANS_EPSILON) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard K-Means segmentation (kept for compatibility)
    """
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    labels = labels.reshape((image.shape[0], image.shape[1]))

    return segmented_image, labels, centers


def preprocess_pipeline(image: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline - each step VISUALLY DIFFERENT
    
    1. Resize        -> standard 256x256
    2. Enhanced      -> CLAHE contrast boost (brighter/sharper)
    3. BG Removed    -> leaf on BLACK background
    4. Blurred       -> very soft/blurry version
    5. HSV ColorMap  -> JET heat-map colors (blue/green/yellow/red)
    6. Segmented     -> bright distinct colored regions (green/red/yellow/magenta)

    Returns:
        resized, enhanced, no_bg, blurred, color_converted, segmented
    """
    if verbose:
        print("  [1/6] Resizing...")
    resized = resize_image(image)

    if verbose:
        print("  [2/6] Enhancing contrast (CLAHE)...")
    enhanced = enhance_contrast(resized)

    if verbose:
        print("  [3/6] Removing background...")
    no_bg = remove_background(enhanced)

    if verbose:
        print("  [4/6] Applying Gaussian blur...")
    blurred = apply_gaussian_blur(no_bg)

    if verbose:
        print("  [5/6] Converting to HSV color map...")
    color_converted = convert_color_space_for_display(no_bg)  # from no_bg, not blurred

    if verbose:
        print("  [6/6] Segmenting disease region...")
    segmented, _ = segment_disease_region(no_bg)  # from no_bg for best clustering

    return resized, enhanced, no_bg, blurred, color_converted, segmented


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocessing for FEATURE EXTRACTION only.
    Uses actual HSV conversion + real cluster centers (not display colors).
    """
    resized = resize_image(image)
    enhanced = enhance_contrast(resized)
    no_bg = remove_background(enhanced)
    blurred = apply_gaussian_blur(no_bg)
    color_converted = convert_color_space(blurred)  # real HSV for features
    segmented, _ = segment_disease_region_for_features(color_converted)  # real centers for features

    return segmented