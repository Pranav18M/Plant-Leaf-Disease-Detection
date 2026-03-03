# Plant Leaf Disease Detection System

A complete machine learning system for automated plant leaf disease detection using classical computer vision and machine learning techniques.

## 🎯 Features

- **Classical ML Pipeline** (No Deep Learning/CNN)
- **Image Preprocessing**: Resize, Gaussian blur, color space conversion
- **K-Means Segmentation**: Isolate diseased regions
- **Feature Extraction**:
  - GLCM texture features (contrast, energy, homogeneity, correlation)
  - Color statistics (mean, std, median, min, max, percentiles)
- **SVM Classification**: Support Vector Machine with RBF kernel
- **Interactive Prediction**: GUI-based image selection
- **Comprehensive Visualization**: View all preprocessing steps

## 📁 Project Structure

```
plant_disease_detection/
│
├── data/                          # Dataset directory
│   ├── train/                     # Training images (organized by class)
│   │   ├── disease_1/             # Class 1 images
│   │   ├── disease_2/             # Class 2 images
│   │   └── disease_3/             # Class 3 images
│   └── test/                      # Test images (optional)
│
├── models/                        # Trained models
│   ├── svm_leaf_disease_model.pkl
│   ├── feature_scaler.pkl
│   ├── label_encoder.pkl
│   └── class_names.txt
│
├── results/                       # Output results and visualizations
│   ├── confusion_matrix.png
│   └── prediction_*.png
│
├── src/                           # Source code
│   ├── preprocessing.py           # Image preprocessing functions
│   ├── feature_extraction.py     # GLCM and color feature extraction
│   ├── train.py                  # Training pipeline
│   ├── predict.py                # Prediction script
│   └── utils.py                  # Utility functions
│
├── config.py                      # Configuration parameters
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- VS Code (recommended) or any Python IDE
- pip (Python package manager)

### Step 1: Clone or Extract the Project
```bash
cd plant_disease_detection
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Alternative (if requirements.txt fails):**
```bash
pip install opencv-python scikit-image scikit-learn numpy matplotlib joblib tqdm Pillow
```

### Step 4: Prepare Dataset

Organize your dataset in the following structure:
```
data/train/
├── healthy/
│   ├── healthy_001.jpg
│   ├── healthy_002.jpg
│   └── ...
├── rust/
│   ├── rust_001.jpg
│   ├── rust_002.jpg
│   └── ...
└── blight/
    ├── blight_001.jpg
    ├── blight_002.jpg
    └── ...
```

**Note:** If you don't have a dataset, the system will create sample synthetic data automatically for testing.

## 🎓 Training the Model

### Using Command Line:
```bash
cd src
python train.py
```

### Using VS Code:
1. Open the project folder in VS Code
2. Open `src/train.py`
3. Press `F5` or click "Run" → "Start Debugging"
4. Or right-click in the editor → "Run Python File in Terminal"

### Training Process:
1. Loads images from `data/train/` folders
2. Preprocesses each image (resize, blur, color conversion, segmentation)
3. Extracts features (GLCM + color features)
4. Trains SVM classifier
5. Evaluates on test set
6. Saves model to `models/` directory
7. Generates confusion matrix in `results/`

### Expected Output:
```
============================================================
Loading dataset from: data/train
Found 3 classes: ['disease_1', 'disease_2', 'disease_3']
============================================================

Loading class 'disease_1': 20 images
Loading class 'disease_2': 20 images
Loading class 'disease_3': 20 images

Total images loaded: 60
============================================================

Extracting Features from Images
Processing images: 100%|████████████████| 60/60

Feature extraction complete!
Feature matrix shape: (60, 69)
Number of features per image: 69

Training SVM Classifier
Training samples: 48
Test samples: 12
Number of features: 69
Number of classes: 3

Accuracy: 91.67%

Model saved to models/svm_leaf_disease_model.pkl
```

## 🔮 Making Predictions

### Using Command Line:
```bash
cd src
python predict.py
```

### Using VS Code:
1. Open `src/predict.py`
2. Press `F5` or run the file
3. A file dialog will open
4. Select a leaf image
5. View results and visualization

### Prediction Process:
1. Opens file dialog to select image
2. Loads trained model
3. Preprocesses the image
4. Extracts features
5. Makes prediction
6. Displays comprehensive visualization with:
   - Original image
   - Resized image
   - Blurred image
   - Color space converted image
   - K-Means segmented image
   - Prediction result with confidence

### Example Output:
```
============================================================
PREDICTION RESULTS
============================================================

  Disease Detected: RUST
  Confidence: 87.45%

============================================================

Visualization saved to results/prediction_leaf_001.png
```

## ⚙️ Configuration

Edit `config.py` to customize:

### Image Processing:
```python
IMAGE_SIZE = (256, 256)           # Image dimensions
GAUSSIAN_KERNEL = (5, 5)          # Blur kernel size
COLOR_SPACE = 'LAB'               # 'LAB', 'YCrCb', or 'HSV'
KMEANS_CLUSTERS = 3               # Number of K-Means clusters
```

### Feature Extraction:
```python
GLCM_DISTANCES = [1, 2, 3]        # Pixel distances for GLCM
GLCM_ANGLES = [0, 45, 90, 135]    # Angles for GLCM
GLCM_PROPERTIES = ['contrast', 'energy', 'homogeneity', 'correlation']
```

### SVM Parameters:
```python
SVM_KERNEL = 'rbf'                # 'linear', 'poly', 'rbf', 'sigmoid'
SVM_C = 10.0                      # Regularization parameter
SVM_GAMMA = 'scale'               # Kernel coefficient
```

## 🛠️ VS Code Setup (Recommended)

### Install VS Code Extensions:
1. **Python** (Microsoft)
2. **Pylance** (Microsoft)
3. **Python Indent** (Kevin Rose)

### Configure VS Code:

**Create `.vscode/launch.json`:**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/train.py",
            "console": "integratedTerminal"
        },
        {
            "name": "Predict Disease",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/predict.py",
            "console": "integratedTerminal"
        }
    ]
}
```

**Create `.vscode/settings.json`:**
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "autopep8",
    "editor.formatOnSave": true
}
```

### Running in VS Code:
1. **Open integrated terminal**: `Ctrl + `` (backtick)
2. **Activate virtual environment**
3. **Run training**: `python src/train.py`
4. **Run prediction**: `python src/predict.py`

## 📊 Understanding the Pipeline

### 1. Image Preprocessing
- **Resize**: Standardize all images to 256×256 pixels
- **Gaussian Blur**: Remove noise and smooth the image
- **Color Space Conversion**: Convert BGR → LAB/YCrCb for better feature separation
- **K-Means Segmentation**: Cluster pixels to isolate diseased regions

### 2. Feature Extraction
- **GLCM (Gray Level Co-occurrence Matrix)**:
  - Contrast: Intensity difference between pixel and neighbor
  - Energy: Sum of squared elements (uniformity)
  - Homogeneity: Closeness of distribution to diagonal
  - Correlation: Linear dependency of gray levels
- **Color Features**:
  - Mean and standard deviation per channel
  - Median, min, max, percentiles

### 3. Classification
- **Support Vector Machine (SVM)**:
  - Finds optimal hyperplane to separate classes
  - RBF kernel for non-linear boundaries
  - StandardScaler for feature normalization

## 🐛 Troubleshooting

### Issue: "No module named cv2"
**Solution:**
```bash
pip install opencv-python
```

### Issue: "tk.TclError: no display name"
**Solution:** For headless systems, modify `predict.py` to skip GUI and accept image path as argument.

### Issue: "Model not found"
**Solution:** Run `train.py` first to train and save the model.

### Issue: Low accuracy
**Solutions:**
- Increase training data
- Adjust SVM parameters in `config.py`
- Try different color spaces (LAB, YCrCb, HSV)
- Modify GLCM parameters

### Issue: "Memory Error"
**Solution:** Reduce `IMAGE_SIZE` or process images in batches.

## 📈 Improving Model Performance

1. **More Training Data**: Collect diverse leaf images
2. **Data Augmentation**: Rotate, flip, adjust brightness
3. **Feature Engineering**: Add shape features, edge features
4. **Hyperparameter Tuning**: Use GridSearchCV
5. **Ensemble Methods**: Combine multiple classifiers
6. **Better Preprocessing**: Enhance segmentation quality

## 📚 Dataset Recommendations

### Public Datasets:
- **PlantVillage Dataset**: 54,000+ images, 38 classes
- **PlantDoc Dataset**: 2,598 images across 13 plant species
- **Rice Leaf Disease Dataset**: Multiple rice diseases
- **Cassava Leaf Disease Dataset**: 5 disease classes

### Dataset Requirements:
- Minimum: 50 images per class
- Recommended: 200+ images per class
- High resolution (>300×300 pixels)
- Good lighting and focus
- Diverse backgrounds and angles

## 🔬 Technical Details

### Feature Vector Composition:
- **GLCM Features**: 48 features (4 properties × 3 distances × 4 angles)
- **Basic Color Features**: 6 features (mean, std for 3 channels)
- **Additional Color Features**: 15 features (median, min, max, p25, p75 for 3 channels)
- **Total**: 69 features per image

### Algorithm Complexity:
- **Training**: O(n² × m) where n = samples, m = features
- **Prediction**: O(n_support × m) where n_support = support vectors

## 📝 License

This project is for educational purposes. Modify and use as needed.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional feature extraction methods
- Alternative classifiers (Random Forest, KNN)
- Web interface for predictions
- Mobile app integration
- Real-time video prediction

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review configuration in `config.py`
3. Ensure dataset is properly organized
4. Verify all dependencies are installed

## 🎉 Acknowledgments

- OpenCV for image processing
- scikit-learn for machine learning
- scikit-image for GLCM features
- The open-source community

---

**Happy Disease Detection! 🌿🔬**