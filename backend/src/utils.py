"""
Utility functions for the Plant Leaf Disease Detection System
"""

import os
import numpy as np
from typing import List, Tuple, Dict
import cv2
from tqdm import tqdm


def load_dataset_from_folders(data_dir: str, verbose: bool = True) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load images from folder structure where each subfolder represents a class
    
    Args:
        data_dir: Path to data directory containing class subfolders
        verbose: Print progress information
        
    Returns:
        images: List of loaded images (as numpy arrays)
        labels: List of class labels corresponding to each image
        class_names: List of unique class names
    """
    images = []
    labels = []
    class_names = []
    
    # Get all subdirectories (each represents a disease class)
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(subdirs) == 0:
        raise ValueError(f"No class folders found in {data_dir}")
    
    class_names = sorted(subdirs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Loading dataset from: {data_dir}")
        print(f"Found {len(class_names)} classes: {class_names}")
        print(f"{'='*60}\n")
    
    # Load images from each class folder
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if verbose:
            print(f"Loading class '{class_name}': {len(image_files)} images")
            iterator = tqdm(image_files, desc=f"  Processing {class_name}")
        else:
            iterator = image_files
        
        for img_file in iterator:
            img_path = os.path.join(class_path, img_file)
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
                else:
                    if verbose:
                        print(f"  Warning: Could not read {img_file}")
            except Exception as e:
                if verbose:
                    print(f"  Error loading {img_file}: {str(e)}")
    
    if verbose:
        print(f"\nTotal images loaded: {len(images)}")
        print(f"{'='*60}\n")
    
    return images, labels, class_names


def create_sample_dataset(data_dir: str, num_classes: int = 3, images_per_class: int = 10):
    """
    Create a sample dataset with synthetic leaf images for testing
    This is useful if you don't have a real dataset yet
    
    Args:
        data_dir: Directory to create sample dataset
        num_classes: Number of disease classes
        images_per_class: Number of images per class
    """
    class_names = [f"disease_{i+1}" for i in range(num_classes)]
    
    print(f"Creating sample dataset in {data_dir}")
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(images_per_class):
            # Create synthetic leaf image with random patterns
            img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
            
            # Add some structure to simulate leaf texture
            img = cv2.GaussianBlur(img, (15, 15), 0)
            
            # Add random spots to simulate disease
            num_spots = np.random.randint(3, 10)
            for _ in range(num_spots):
                center = (np.random.randint(50, 250), np.random.randint(50, 250))
                radius = np.random.randint(10, 30)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(img, center, radius, color, -1)
            
            # Save image
            img_path = os.path.join(class_dir, f"{class_name}_{i+1:03d}.jpg")
            cv2.imwrite(img_path, img)
    
    print(f"Created {num_classes} classes with {images_per_class} images each")


def display_class_distribution(labels: List[str], class_names: List[str]):
    """
    Display the distribution of classes in the dataset
    
    Args:
        labels: List of labels
        class_names: List of unique class names
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    
    print(f"\n{'='*60}")
    print("Class Distribution:")
    print(f"{'='*60}")
    for class_name in class_names:
        count = label_counts[class_name]
        percentage = (count / len(labels)) * 100
        print(f"  {class_name:20s}: {count:4d} images ({percentage:5.1f}%)")
    print(f"{'='*60}\n")


def save_results(results: Dict, filename: str):
    """
    Save results to a text file
    
    Args:
        results: Dictionary containing results
        filename: Path to save file
    """
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to {filename}")