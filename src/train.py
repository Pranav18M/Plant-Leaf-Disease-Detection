"""
Training Script for Plant Leaf Disease Detection System
Trains an SVM classifier using extracted features
"""

import os
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import load_dataset_from_folders, display_class_distribution, create_sample_dataset
from preprocessing import preprocess_image
from feature_extraction import extract_all_features, get_feature_names


def extract_features_from_images(images, labels, verbose=True):
    """
    Extract features from all images in the dataset
    
    Args:
        images: List of images
        labels: List of labels
        verbose: Print progress
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array
    """
    X = []
    y = []
    
    if verbose:
        print(f"\n{'='*60}")
        print("Extracting Features from Images")
        print(f"{'='*60}\n")
        iterator = tqdm(zip(images, labels), total=len(images), desc="Processing images")
    else:
        iterator = zip(images, labels)
    
    for img, label in iterator:
        try:
            # Preprocess image
            preprocessed = preprocess_image(img)
            
            # Extract features
            features = extract_all_features(preprocessed)
            
            X.append(features)
            y.append(label)
            
        except Exception as e:
            if verbose:
                print(f"\nError processing image with label {label}: {str(e)}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    if verbose:
        print(f"\nFeature extraction complete!")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features per image: {X.shape[1]}")
    
    return X, y


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_feature_importance(model, feature_names, class_names, save_path, top_n=20):
    """
    Plot feature importance based on SVM coefficients (for linear kernel)
    
    Args:
        model: Trained SVM model
        feature_names: List of feature names
        class_names: List of class names
        save_path: Path to save the plot
        top_n: Number of top features to display
    """
    if config.SVM_KERNEL != 'linear':
        print("Feature importance visualization only available for linear kernel")
        return
    
    # Get absolute coefficients
    coef = np.abs(model.coef_).mean(axis=0)
    
    # Get top N features
    top_indices = np.argsort(coef)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_values = coef[top_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance (Absolute Coefficient)', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.DPI)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def train_model(X_train, y_train, X_test, y_test, class_names, label_encoder, verbose=True):
    """
    Train SVM classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        label_encoder: Fitted LabelEncoder
        verbose: Print training progress
        
    Returns:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        accuracy: Model accuracy on test set
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Training SVM Classifier")
        print(f"{'='*60}\n")
    
    # Encode labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    if verbose:
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(class_names)}")
        print(f"\nSVM Parameters:")
        print(f"  Kernel: {config.SVM_KERNEL}")
        print(f"  C: {config.SVM_C}")
        print(f"  Gamma: {config.SVM_GAMMA}")
    
    # Feature scaling
    if verbose:
        print(f"\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    if verbose:
        print(f"Training SVM model...")
    
    svm_model = SVC(
        kernel=config.SVM_KERNEL,
        C=config.SVM_C,
        gamma=config.SVM_GAMMA,
        max_iter=config.SVM_MAX_ITER,
        random_state=config.RANDOM_STATE,
        verbose=verbose
    )
    
    svm_model.fit(X_train_scaled, y_train_encoded)
    
    if verbose:
        print(f"Training complete!")
    
    # Evaluate model
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Model Evaluation")
        print(f"{'='*60}\n")
        print(f"Accuracy: {accuracy * 100:.2f}%\n")
        print("Classification Report:")
        
        # Get unique classes present in both test and predictions
        unique_labels = sorted(set(list(y_test_encoded) + list(y_pred)))
        label_names = [label_encoder.inverse_transform([i])[0] for i in unique_labels]
        
        print(classification_report(y_test_encoded, y_pred, 
                                   labels=unique_labels,
                                   target_names=label_names, 
                                   zero_division=0))
    
    # Confusion matrix
    unique_labels = sorted(set(list(y_test_encoded) + list(y_pred)))
    label_names = [label_encoder.inverse_transform([i])[0] for i in unique_labels]
    cm = confusion_matrix(y_test_encoded, y_pred, labels=unique_labels)
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, label_names, cm_path)
    
    # Feature importance (if linear kernel)
    if config.SVM_KERNEL == 'linear':
        feature_names = get_feature_names()
        fi_path = os.path.join(config.RESULTS_DIR, 'feature_importance.png')
        plot_feature_importance(svm_model, feature_names, class_names, fi_path)
    
    return svm_model, scaler, accuracy


def save_model(model, scaler, label_encoder, class_names):
    """
    Save trained model, scaler, and label encoder
    
    Args:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        class_names: List of class names
    """
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_FILENAME)
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_FILENAME)
    encoder_path = os.path.join(config.MODEL_DIR, config.LABEL_ENCODER_FILENAME)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save label encoder
    joblib.dump(label_encoder, encoder_path)
    print(f"Label encoder saved to {encoder_path}")
    
    # Save class names for reference
    class_names_path = os.path.join(config.MODEL_DIR, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class names saved to {class_names_path}")


def main():
    """
    Main training pipeline
    """
    print(f"\n{'#'*60}")
    print("# Plant Leaf Disease Detection - Training Pipeline")
    print(f"{'#'*60}\n")
    
    # Check if training data exists
    if not os.path.exists(config.TRAIN_DIR):
        print(f"Training directory not found: {config.TRAIN_DIR}")
        print(f"Creating sample dataset for demonstration...")
        create_sample_dataset(config.TRAIN_DIR, num_classes=3, images_per_class=20)
        print(f"\nNote: This is synthetic data. Replace with real leaf disease images.")
    
    # Check if there are any class folders
    subdirs = [d for d in os.listdir(config.TRAIN_DIR) 
               if os.path.isdir(os.path.join(config.TRAIN_DIR, d))]
    
    if len(subdirs) == 0:
        print(f"\nError: No class folders found in {config.TRAIN_DIR}")
        print(f"Please organize your data as: data/train/class_name/images.jpg")
        return
    
    # Load dataset
    images, labels, class_names = load_dataset_from_folders(config.TRAIN_DIR, verbose=config.VERBOSE)
    
    if len(images) == 0:
        print("Error: No images loaded. Check your dataset directory.")
        return
    
    # Display class distribution
    display_class_distribution(labels, class_names)
    
    # Extract features
    X, y = extract_features_from_images(images, labels, verbose=config.VERBOSE)
    
    if len(X) == 0:
        print("Error: No features extracted. Check preprocessing pipeline.")
        return
    
    # Encode all labels first
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Split data
    print(f"\nSplitting data (test size: {config.TEST_SIZE * 100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Train model
    model, scaler, accuracy = train_model(
        X_train, y_train, X_test, y_test, class_names, label_encoder, verbose=config.VERBOSE
    )
    
    # Save model
    print(f"\n{'='*60}")
    print("Saving Model")
    print(f"{'='*60}\n")
    save_model(model, scaler, label_encoder, class_names)
    
    print(f"\n{'#'*60}")
    print(f"# Training Complete! Final Accuracy: {accuracy * 100:.2f}%")
    print(f"{'#'*60}\n")
    
    print("Next steps:")
    print("  1. Review the confusion matrix in the results folder")
    print("  2. Run predict.py to test the model on new images")
    print("  3. Add more training images to improve accuracy\n")


if __name__ == "__main__":
    main()