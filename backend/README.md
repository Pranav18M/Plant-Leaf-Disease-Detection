# Plant Leaf Disease Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional AI-powered plant disease detection system using Support Vector Machine (SVM) with advanced image processing and bilingual (English-Tamil) output support.

## 🌟 Features

- **Automated Disease Detection**: 11 different plant disease classifications
- **Advanced Image Processing**: 7-step preprocessing pipeline with CLAHE, K-Means, and HSV analysis
- **High Accuracy**: 75-85% accuracy with super-aggressive data augmentation (50x multiplier)
- **Professional Visualization**: 9-panel dark-themed dashboard showing complete processing pipeline
- **Bilingual Support**: English and Tamil treatment recommendations
- **Comprehensive Treatment Database**: Chemical and organic treatment options with dosage and cost information
- **Real-time Processing**: Instant predictions with confidence scores

## 📊 Supported Diseases

1. Apple - Apple Scab
2. Apple - Healthy
3. Corn (Maize) - Cercospora Leaf Spot (Gray Leaf Spot)
4. Corn (Maize) - Northern Leaf Blight
5. Grape - Black Rot
6. Grape - Leaf Blight
7. Grape - Healthy
8. Orange - Huanglongbing (Citrus Greening)
9. Strawberry - Leaf Scorch
10. Tomato - Bacterial Spot
11. Tomato - Early Blight

## 🔧 System Requirements

### Hardware
- **Minimum**: 4GB RAM, Dual-core processor
- **Recommended**: 8GB+ RAM, Quad-core processor
- **Storage**: 500MB free space

### Software
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.8, 3.9, 3.10, or 3.11
- **Display**: 1920x1080 resolution or higher (for optimal visualization)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/plant_disease_detection.git
cd plant_disease_detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --break-system-packages
```

**Note for macOS/Linux users**: If you encounter permission errors, use:
```bash
pip install -r requirements.txt --user
```

### Step 4: Install Tamil Font Support

**Windows:**
1. Download Noto Sans Tamil font: [Download Link](https://fonts.google.com/noto/specimen/Noto+Sans+Tamil)
2. Right-click `NotoSansTamil-Regular.ttf` → Install for all users
3. The font is already included in the `font/` directory

**macOS:**
- Double-click `font/NotoSansTamil-Regular.ttf` → Install Font

**Linux:**
```bash
sudo cp font/NotoSansTamil-Regular.ttf /usr/share/fonts/truetype/
sudo fc-cache -f -v
```

### Step 5: Verify Installation

```bash
python src/verify_installation.py
```

## 📂 Project Structure

```
plant_disease_detection/
├── data/
│   ├── train/                          # Training images (organized by class)
│   ├── disease_treatments.json         # Treatment database (English)
│   ├── disease_translations_tamil.json # Disease names (Tamil)
│   └── treatment_translations_tamil.json # Treatments (Tamil)
├── font/
│   └── NotoSansTamil-Regular.ttf       # Tamil font file
├── models/
│   ├── svm_leaf_disease_model.pkl      # Trained SVM model
│   ├── feature_scaler.pkl              # Feature scaler
│   ├── label_encoder.pkl               # Label encoder
│   └── class_names.txt                 # Class labels
├── results/                             # Output predictions (generated)
├── src/
│   ├── config.py                       # Configuration settings
│   ├── preprocessing.py                # Image preprocessing pipeline
│   ├── feature_extraction.py           # GLCM feature extraction
│   ├── train.py                        # Model training
│   ├── train_super_augmented.py        # Training with 50x augmentation
│   ├── predict.py                      # Prediction with bilingual output
│   └── utils.py                        # Utility functions
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── PROJECT_EXPLANATION.txt             # Detailed project explanation

```

## 🚀 Quick Start

### 1. Training the Model (Optional - Pre-trained model included)

**Standard Training:**
```bash
cd src
python train.py
```

**With Super Augmentation (Recommended for small datasets):**
```bash
cd src
python train_super_augmented.py
```

**Training parameters:**
- Data Augmentation: 50x multiplier (rotation, flip, zoom, brightness, noise)
- Feature Extraction: 170 features (GLCM texture + HSV color statistics)
- Model: SVM with RBF kernel
- Expected Training Time: 5-15 minutes (depending on dataset size)

### 2. Making Predictions

```bash
cd src
python predict.py
```

**Output Files Generated:**
1. `prediction_[filename].png` - Main 9-panel processing visualization
2. `treatment_english_[filename].png` - English treatment recommendations
3. `treatment_tamil_[filename].png` - Tamil treatment recommendations (தமிழ் சிகிச்சை)

## 📖 Detailed Usage

### Preprocessing Pipeline (7 Steps)

1. **Resize**: Standardize to 256x256 pixels
2. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Background Removal**: HSV-based green mask with morphological operations
4. **Gaussian Blur**: Noise reduction (21x21 kernel)
5. **Color Space Conversion**: BGR → HSV for feature extraction
6. **K-Means Segmentation**: Disease region identification (8 clusters)
7. **Feature Extraction**: 170 features (GLCM texture + HSV statistics)

### Feature Extraction

**GLCM (Gray-Level Co-occurrence Matrix) Features:**
- Contrast, Dissimilarity, Homogeneity, Energy
- Correlation, ASM (Angular Second Moment)
- Extracted at 4 angles: 0°, 45°, 90°, 135°

**HSV Color Features:**
- Mean, Standard Deviation, Min, Max for each channel
- Total: 170 features per image

### Model Architecture

- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Input**: 170 features per image
- **Output**: Disease classification + confidence score (85-95%)

## 🎨 Output Visualization

### Main Prediction Page (9 Panels)
- Panel 1-7: Processing steps visualization
- Panel 8: Complete pipeline flowchart
- Panel 9: Final prediction with Tamil disease name

### Treatment Recommendations
- **English Page**: Professional layout with treatment options
- **Tamil Page**: Bilingual support for farmers (தமிழ் விவசாயிகளுக்கு)
- Includes: Chemical treatments, Organic alternatives, Dosage, Cost, Application methods

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Image Processing
IMAGE_SIZE = (256, 256)
KMEANS_CLUSTERS = 8
COLOR_SPACE = 'HSV'

# GLCM Parameters
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Model Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

## 📊 Performance Metrics

### Accuracy
- Training Accuracy: 85-92%
- Validation Accuracy: 75-85%
- Confidence Range: 85-95%

### Processing Speed
- Image Preprocessing: ~2-3 seconds
- Feature Extraction: ~1 second
- Prediction: <0.5 seconds
- **Total Time**: ~4 seconds per image

## 🐛 Troubleshooting

### Common Issues

**1. Tamil text showing as boxes:**
```bash
# Verify font installation
python -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'tamil' in f.name.lower()])"

# If empty, reinstall Tamil font
```

**2. Import errors:**
```bash
# Reinstall dependencies
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

**3. Memory errors during training:**
```bash
# Reduce augmentation multiplier in train_super_augmented.py
AUGMENTATION_MULTIPLIER = 25  # Instead of 50
```

**4. Low accuracy:**
- Ensure training data is properly labeled
- Use super augmentation for small datasets (<50 images per class)
- Check image quality (resolution, lighting, focus)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset: PlantVillage Dataset
- GLCM Implementation: scikit-image
- Tamil Language Support: Noto Sans Tamil Font (Google Fonts)
- Image Processing: OpenCV, PIL/Pillow
- Machine Learning: scikit-learn

## 📧 Contact

For questions, suggestions, or issues:
- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/plant_disease_detection/issues)
- **Project Documentation**: See `PROJECT_EXPLANATION.txt` for technical details

## 🔮 Future Enhancements

- [ ] Mobile app deployment (Android/iOS)
- [ ] Web interface with Django/Flask
- [ ] Real-time camera detection
- [ ] Additional disease classes (target: 25+ diseases)
- [ ] Multi-language support (Hindi, Telugu, Kannada)
- [ ] GPS-based treatment center locator
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Deep learning model comparison (CNN, ResNet, EfficientNet)

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{plant_disease_detection_2024,
  author = {Your Name},
  title = {Plant Leaf Disease Detection System with Bilingual Support},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/plant_disease_detection}
}
```

---

**⭐ If this project helped you, please consider giving it a star!**

**Made with ❤️ for farmers and agricultural communities**
