"""
Configuration file for Plant Leaf Disease Detection System
"""

import os

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# IMAGE PREPROCESSING
IMAGE_SIZE = (256, 256)
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.0

# COLOR SPACE - HSV better for leaf disease
COLOR_SPACE = 'HSV'  # Changed from LAB

# SEGMENTATION - Better isolation
KMEANS_CLUSTERS = 4  # Increased clusters
KMEANS_ITERATIONS = 100
KMEANS_EPSILON = 0.2

# GLCM - More features
GLCM_DISTANCES = [1, 2, 3, 5]  # Added distance 5
GLCM_ANGLES = [0, 45, 90, 135]
GLCM_LEVELS = 256
GLCM_SYMMETRIC = True
GLCM_NORMED = True
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']  # More properties

COLOR_CHANNELS = 3

# SVM - Optimized
TEST_SIZE = 0.1
RANDOM_STATE = 42
SVM_KERNEL = 'rbf'
SVM_C = 100.0  # Increased
SVM_GAMMA = 'auto'  # Changed
SVM_MAX_ITER = 2000  # Increased

# MODEL FILES
MODEL_FILENAME = 'svm_leaf_disease_model.pkl'
SCALER_FILENAME = 'feature_scaler.pkl'
LABEL_ENCODER_FILENAME = 'label_encoder.pkl'

# VISUALIZATION
FIGURE_SIZE = (12, 8)
DPI = 100
VERBOSE = True
