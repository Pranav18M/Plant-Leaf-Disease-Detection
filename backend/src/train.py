"""
Super Aggressive Augmentation Training
Creates 50+ samples from each original image
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import load_dataset_from_folders, display_class_distribution
from preprocessing import preprocess_image
from feature_extraction import extract_all_features


def super_augment_image(image):
    """
    Create 50+ augmented versions of a single image
    """
    augmented = []
    
    # Original
    augmented.append(image)
    
    # Flips
    augmented.append(cv2.flip(image, 1))  # Horizontal
    augmented.append(cv2.flip(image, 0))  # Vertical
    augmented.append(cv2.flip(cv2.flip(image, 0), 1))  # Both
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotations (every 30 degrees)
    for angle in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)
    
    # Small rotations
    for angle in [5, 10, 15, -5, -10, -15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)
    
    # Brightness variations
    for beta in [30, 20, 10, -10, -20, -30]:
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        augmented.append(adjusted)
    
    # Contrast variations
    for alpha in [0.8, 0.9, 1.1, 1.2]:
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented.append(adjusted)
    
    # Zoom variations
    for scale in [0.8, 0.9, 1.1, 1.2]:
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (scaled_w, scaled_h))
        
        if scale < 1.0:  # Add padding
            top = (h - scaled_h) // 2
            bottom = h - scaled_h - top
            left = (w - scaled_w) // 2
            right = w - scaled_w - left
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                       cv2.BORDER_REFLECT)
            augmented.append(padded)
        else:  # Crop center
            start_h = (scaled_h - h) // 2
            start_w = (scaled_w - w) // 2
            cropped = resized[start_h:start_h+h, start_w:start_w+w]
            augmented.append(cropped)
    
    # Gaussian blur variations
    for ksize in [3, 5, 7]:
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        augmented.append(blurred)
    
    # Add noise
    for _ in range(3):
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        augmented.append(noisy)
    
    print(f"    Generated {len(augmented)} samples from 1 image")
    return augmented


def extract_features_super_augmented(images, labels, verbose=True):
    """Extract features with super augmentation"""
    X = []
    y = []
    
    if verbose:
        print(f"\n{'='*70}")
        print("SUPER AUGMENTATION - Extracting Features")
        print(f"{'='*70}\n")
    
    for idx, (img, label) in enumerate(zip(images, labels), 1):
        try:
            print(f"\n[{idx}/{len(images)}] Processing: {label}")
            
            # Generate 50+ augmented versions
            augmented_images = super_augment_image(img)
            
            # Extract features from each
            for aug_img in tqdm(augmented_images, desc="  Extracting features", leave=False):
                preprocessed = preprocess_image(aug_img)
                features = extract_all_features(preprocessed)
                X.append(features)
                y.append(label)
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    if verbose:
        print(f"\n{'='*70}")
        print("✓ Augmentation Complete!")
        print(f"  Original images: {len(images)}")
        print(f"  Total samples generated: {len(X)}")
        print(f"  Augmentation factor: {len(X) // len(images)}x")
        print(f"  Features per sample: {X.shape[1]}")
        print(f"{'='*70}\n")
    
    return X, y


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Super Augmented Training', fontsize=18, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def train_model(X_train, y_train, X_test, y_test, class_names, label_encoder, verbose=True):
    """Train SVM with optimal parameters for small dataset"""
    if verbose:
        print(f"\n{'='*70}")
        print("Training SVM Classifier")
        print(f"{'='*70}\n")
    
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    if verbose:
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(class_names)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM with RBF kernel - high C for small dataset
    if verbose:
        print(f"\n🔧 SVM Parameters:")
        print(f"   Kernel: rbf")
        print(f"   C: 200 (high regularization for overfitting prevention)")
        print(f"   Gamma: scale")
        print(f"   Max iterations: 3000\n")
        print("Training...")
    
    svm_model = SVC(
        kernel='rbf',
        C=200.0,  # Higher C for better fit
        gamma='scale',  # Auto-scale gamma
        max_iter=3000,
        random_state=42,
        verbose=verbose
    )
    
    svm_model.fit(X_train_scaled, y_train_encoded)
    
    # Evaluate
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    if verbose:
        print(f"\n{'='*70}")
        print("📊 Evaluation Results")
        print(f"{'='*70}\n")
        print(f"✓ Accuracy: {accuracy * 100:.2f}%\n")
        
        unique_labels = sorted(set(list(y_test_encoded) + list(y_pred)))
        label_names = [label_encoder.inverse_transform([i])[0] for i in unique_labels]
        
        print("Classification Report:")
        print(classification_report(y_test_encoded, y_pred, 
                                   labels=unique_labels,
                                   target_names=label_names, 
                                   zero_division=0))
    
    # Confusion matrix
    unique_labels = sorted(set(list(y_test_encoded) + list(y_pred)))
    label_names = [label_encoder.inverse_transform([i])[0] for i in unique_labels]
    cm = confusion_matrix(y_test_encoded, y_pred, labels=unique_labels)
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix_super_aug.png')
    plot_confusion_matrix(cm, label_names, cm_path)
    
    return svm_model, scaler, accuracy


def save_model(model, scaler, label_encoder, class_names):
    """Save model files"""
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_FILENAME)
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_FILENAME)
    encoder_path = os.path.join(config.MODEL_DIR, config.LABEL_ENCODER_FILENAME)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    class_names_path = os.path.join(config.MODEL_DIR, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"✓ Encoder saved: {encoder_path}")
    print(f"✓ Classes saved: {class_names_path}")


def main():
    """Main training pipeline"""
    print(f"\n{'#'*70}")
    print("# SUPER AUGMENTED TRAINING")
    print("# Plant Leaf Disease Detection")
    print(f"{'#'*70}\n")
    
    if not os.path.exists(config.TRAIN_DIR):
        print(f"❌ Error: Training directory not found: {config.TRAIN_DIR}")
        return
    
    # Load dataset
    print("📂 Loading dataset...")
    images, labels, class_names = load_dataset_from_folders(config.TRAIN_DIR, verbose=True)
    
    if len(images) == 0:
        print("❌ Error: No images found!")
        return
    
    print(f"\n📊 Original Dataset:")
    print(f"   Total images: {len(images)}")
    print(f"   Classes: {len(class_names)}")
    
    display_class_distribution(labels, class_names)
    
    # Extract features with SUPER augmentation
    X, y = extract_features_super_augmented(images, labels, verbose=True)
    
    if len(X) == 0:
        print("❌ Error: No features extracted!")
        return
    
    # Check samples per class
    from collections import Counter
    class_counts = Counter(y)
    print("📊 Augmented Dataset Distribution:")
    for cls in class_names:
        count = class_counts[cls]
        print(f"   {cls}: {count} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Split data (85/15 split for small dataset)
    print(f"\n✂️ Splitting data (85% train, 15% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Train
    model, scaler, accuracy = train_model(
        X_train, y_train, X_test, y_test, class_names, label_encoder, verbose=True
    )
    
    # Save
    print(f"\n{'='*70}")
    print("💾 Saving Model")
    print(f"{'='*70}")
    save_model(model, scaler, label_encoder, class_names)
    
    print(f"\n{'#'*70}")
    if accuracy >= 0.80:
        print(f"# ✓ SUCCESS! Final Accuracy: {accuracy * 100:.2f}%")
    elif accuracy >= 0.70:
        print(f"# ⚠ MODERATE! Final Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"# ❌ LOW ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'#'*70}\n")
    
    if accuracy < 0.75:
        print("⚠️  Recommendations to improve accuracy:")
        print("   1. Download PlantVillage dataset for more real images")
        print("   2. Ensure image quality is good")
        print("   3. Check that images are correctly labeled")
        print("   4. Add at least 20-50 images per class\n")
    else:
        print("✓ Model trained successfully!")
        print("  Run 'python predict.py' to test it\n")


if __name__ == "__main__":
    main()