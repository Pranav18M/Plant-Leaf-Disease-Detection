"""
Feature Extraction Module for Plant Leaf Disease Detection
Extracts texture features using GLCM and color features
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from typing import List, Tuple
import config


def extract_glcm_features(image: np.ndarray, 
                         distances: List[int] = config.GLCM_DISTANCES,
                         angles: List[int] = config.GLCM_ANGLES,
                         levels: int = config.GLCM_LEVELS,
                         properties: List[str] = config.GLCM_PROPERTIES) -> np.ndarray:
    """
    Extract texture features using Gray Level Co-occurrence Matrix (GLCM)
    
    Args:
        image: Input image (can be color or grayscale)
        distances: List of pixel pair distance offsets
        angles: List of angles in degrees
        levels: Number of gray-levels
        properties: List of GLCM properties to compute
        
    Returns:
        1D array of GLCM features
    """
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to specified gray levels
    gray = cv2.normalize(gray, None, 0, levels - 1, cv2.NORM_MINMAX)
    gray = gray.astype(np.uint8)
    
    # Convert angles to radians
    angles_rad = [angle * np.pi / 180.0 for angle in angles]
    
    # Compute GLCM
    glcm = graycomatrix(gray, 
                        distances=distances, 
                        angles=angles_rad,
                        levels=levels,
                        symmetric=config.GLCM_SYMMETRIC,
                        normed=config.GLCM_NORMED)
    
    # Extract features
    features = []
    for prop in properties:
        feature = graycoprops(glcm, prop)
        # Flatten the feature matrix (distances x angles)
        features.extend(feature.flatten())
    
    return np.array(features)


def extract_color_features(image: np.ndarray) -> np.ndarray:
    """
    Extract basic color features (mean and standard deviation) from each channel
    
    Args:
        image: Input image (must be in color space like LAB, YCrCb, etc.)
        
    Returns:
        1D array of color features
    """
    features = []
    
    # Extract features from each channel
    for channel in range(image.shape[2]):
        channel_data = image[:, :, channel]
        
        # Mean
        mean = np.mean(channel_data)
        features.append(mean)
        
        # Standard deviation
        std = np.std(channel_data)
        features.append(std)
    
    return np.array(features)


def extract_additional_color_features(image: np.ndarray) -> np.ndarray:
    """
    Extract additional color features including median, min, max, and percentiles
    
    Args:
        image: Input image
        
    Returns:
        1D array of additional color features
    """
    features = []
    
    for channel in range(image.shape[2]):
        channel_data = image[:, :, channel]
        
        # Median
        features.append(np.median(channel_data))
        
        # Min and Max
        features.append(np.min(channel_data))
        features.append(np.max(channel_data))
        
        # 25th and 75th percentiles
        features.append(np.percentile(channel_data, 25))
        features.append(np.percentile(channel_data, 75))
    
    return np.array(features)


def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """
    Extract shape features from binary mask of diseased region
    
    Args:
        mask: Binary mask of diseased region
        
    Returns:
        1D array of shape features
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Return default features if no contours found
        return np.array([0, 0, 0, 0, 0])
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Area
    area = cv2.contourArea(largest_contour)
    
    # Perimeter
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Circularity
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    # Aspect ratio
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        if minor_axis > 0:
            aspect_ratio = major_axis / minor_axis
        else:
            aspect_ratio = 0
    else:
        aspect_ratio = 0
    
    # Extent (ratio of contour area to bounding box area)
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect_area = w * h
    if rect_area > 0:
        extent = area / rect_area
    else:
        extent = 0
    
    return np.array([area, perimeter, circularity, aspect_ratio, extent])


def extract_all_features(image: np.ndarray, include_shape: bool = False) -> np.ndarray:
    """
    Extract all features from preprocessed image
    Combines GLCM texture features and color features
    
    Args:
        image: Preprocessed image (after color space conversion and segmentation)
        include_shape: Whether to include shape features (requires binary mask)
        
    Returns:
        Complete feature vector
    """
    # Extract GLCM texture features
    glcm_features = extract_glcm_features(image)
    
    # Extract color features
    color_features = extract_color_features(image)
    
    # Extract additional color features
    additional_color_features = extract_additional_color_features(image)
    
    # Combine all features
    all_features = np.concatenate([glcm_features, color_features, additional_color_features])
    
    # Optionally add shape features (requires manual mask extraction)
    if include_shape:
        # Convert to grayscale and threshold to get mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        shape_features = extract_shape_features(mask)
        all_features = np.concatenate([all_features, shape_features])
    
    return all_features


def get_feature_names() -> List[str]:
    """
    Get names of all extracted features for interpretation
    
    Returns:
        List of feature names
    """
    feature_names = []
    
    # GLCM features
    for prop in config.GLCM_PROPERTIES:
        for dist in config.GLCM_DISTANCES:
            for angle in config.GLCM_ANGLES:
                feature_names.append(f"GLCM_{prop}_d{dist}_a{angle}")
    
    # Color features (mean and std for each channel)
    channels = ['Ch1', 'Ch2', 'Ch3']  # Generic channel names
    for ch in channels:
        feature_names.append(f"Color_{ch}_mean")
        feature_names.append(f"Color_{ch}_std")
    
    # Additional color features
    for ch in channels:
        feature_names.append(f"Color_{ch}_median")
        feature_names.append(f"Color_{ch}_min")
        feature_names.append(f"Color_{ch}_max")
        feature_names.append(f"Color_{ch}_p25")
        feature_names.append(f"Color_{ch}_p75")
    
    return feature_names