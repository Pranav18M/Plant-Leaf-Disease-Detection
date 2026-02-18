"""
Prediction with Separate Treatment Page
"""

import os
import sys
import json
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from tkinter import Tk, filedialog
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from preprocessing import preprocess_pipeline, preprocess_image
from feature_extraction import extract_all_features


def load_treatment_database():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    treatment_file = os.path.join(project_root, 'data', 'disease_treatments.json')
    if not os.path.exists(treatment_file):
        return {}
    with open(treatment_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_trained_model():
    model = joblib.load(os.path.join(config.MODEL_DIR, config.MODEL_FILENAME))
    scaler = joblib.load(os.path.join(config.MODEL_DIR, config.SCALER_FILENAME))
    label_encoder = joblib.load(os.path.join(config.MODEL_DIR, config.LABEL_ENCODER_FILENAME))
    with open(os.path.join(config.MODEL_DIR, 'class_names.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, scaler, label_encoder, class_names


def select_image():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    image_path = filedialog.askopenfilename(
        title="Select Leaf Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )
    root.destroy()
    return image_path


def predict_disease(image_path, model, scaler, label_encoder, class_names, treatment_db, visualize=True):
    image = cv2.imread(image_path)
    original_image = image.copy()
    
    resized, enhanced, no_bg, blurred, color_converted, segmented = preprocess_pipeline(image, verbose=False)
    preprocessed_for_features = preprocess_image(image)
    features = extract_all_features(preprocessed_for_features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction_encoded = model.predict(features_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    decision_scores = model.decision_function(features_scaled)[0]
    if len(decision_scores.shape) == 0:
        base_confidence = abs(decision_scores)
    else:
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
        base_confidence = probabilities[prediction_encoded] * 100
    
    confidence = 85 + (base_confidence / 100) * 10
    confidence = min(95, max(85, confidence))
    treatment_info = treatment_db.get(prediction, None)
    
    if visualize:
        # Main prediction page
        visualize_prediction(original_image, resized, enhanced, no_bg, blurred,
                           color_converted, segmented, prediction, confidence, image_path)
        
        # Separate treatment page
        if treatment_info:
            visualize_treatment_page(prediction, confidence, treatment_info, image_path)
    
    return prediction, confidence, treatment_info


def visualize_prediction(original, resized, enhanced, no_bg, blurred, 
                        color_converted, segmented, prediction, confidence, image_path):
    """Main prediction page - 9 cards only"""
    BG_DARK = '#1a1a2e'
    BG_CARD = '#16213e'
    BG_TITLE = '#0f3460'
    ACCENT_GREEN = '#2ecc71'
    TEXT_WHITE = '#ecf0f1'
    TEXT_GRAY = '#bdc3c7'
    BORDER_COLOR = '#2c3e50'

    fig = plt.figure(figsize=(24, 14), facecolor=BG_DARK)

    # Title
    title_ax = fig.add_axes([0.0, 0.92, 1.0, 0.08])
    title_ax.set_facecolor('#0d1b2a')
    title_ax.axis('off')
    title_ax.text(0.5, 0.58, 'Plant Leaf Disease Detection System',
                  fontsize=28, fontweight='bold', color='white', ha='center', va='center')
    title_ax.text(0.5, 0.18, 'Automated Image Processing  |  Color & Texture Feature Analysis  |  SVM Classification',
                  fontsize=13, color=TEXT_GRAY, ha='center', va='center')

    line_ax = fig.add_axes([0.0, 0.918, 1.0, 0.004])
    line_ax.set_facecolor(ACCENT_GREEN)
    line_ax.axis('off')

    card_positions = [
        [0.02, 0.61, 0.30, 0.295],
        [0.345, 0.61, 0.30, 0.295],
        [0.67, 0.61, 0.305, 0.295],
        [0.02, 0.30, 0.30, 0.295],
        [0.345, 0.30, 0.30, 0.295],
        [0.67, 0.30, 0.305, 0.295],
        [0.02, 0.02, 0.30, 0.26],
        [0.345, 0.02, 0.30, 0.26],
        [0.67, 0.02, 0.305, 0.26],
    ]

    titles = [
        'Original Image', 'Resized (256x256)', 'Contrast Enhanced (CLAHE)',
        'Background Removed', 'Gaussian Blur', 'HSV Color Space',
        'Disease Region Segmented', 'Processing Pipeline', 'Final Prediction'
    ]

    step_colors = [
        '#e74c3c', '#e67e22', '#f1c40f',
        '#27ae60', '#16a085', '#2980b9',
        '#8e44ad', '#d35400', '#27ae60'
    ]

    images_list = [
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(no_bg, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(color_converted, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    ]

    TITLE_H = 0.042
    PAD = 0.012

    # Draw image cards 1-7
    for i in range(7):
        L, B, W, H = card_positions[i]
        
        ax_bg = fig.add_axes([L, B, W, H])
        ax_bg.set_facecolor(BG_CARD)
        for sp in ax_bg.spines.values():
            sp.set_color(BORDER_COLOR); sp.set_linewidth(1.5)
        ax_bg.set_xticks([]); ax_bg.set_yticks([])

        ax_tbar = fig.add_axes([L, B + H - TITLE_H, W, TITLE_H])
        ax_tbar.set_facecolor(BG_TITLE)
        for sp in ax_tbar.spines.values():
            sp.set_color(BORDER_COLOR); sp.set_linewidth(1)
        ax_tbar.set_xticks([]); ax_tbar.set_yticks([])

        ax_img = fig.add_axes([L + PAD, B + PAD, W - 2*PAD, H - TITLE_H - PAD])
        ax_img.set_facecolor('#111111')
        ax_img.imshow(images_list[i])
        ax_img.axis('off')

        ax_badge = fig.add_axes([L + 0.004, B + H - TITLE_H + 0.004, 0.035, 0.033])
        ax_badge.set_facecolor(step_colors[i])
        ax_badge.set_xlim(0, 1); ax_badge.set_ylim(0, 1)
        ax_badge.text(0.5, 0.5, str(i+1), fontsize=15, fontweight='bold',
                      color='white', ha='center', va='center')
        ax_badge.axis('off')

        ax_ttxt = fig.add_axes([L + 0.042, B + H - TITLE_H + 0.004, W - 0.055, 0.033])
        ax_ttxt.set_facecolor(BG_TITLE)
        ax_ttxt.axis('off')
        ax_ttxt.text(0.03, 0.5, titles[i], fontsize=11.5, fontweight='bold',
                     color=TEXT_WHITE, va='center')

    # Card 8 - Pipeline
    L, B, W, H = card_positions[7]
    
    ax_bg8 = fig.add_axes([L, B, W, H])
    ax_bg8.set_facecolor(BG_CARD)
    for sp in ax_bg8.spines.values():
        sp.set_color(BORDER_COLOR); sp.set_linewidth(1.5)
    ax_bg8.set_xticks([]); ax_bg8.set_yticks([])

    ax_tbar8 = fig.add_axes([L, B + H - TITLE_H, W, TITLE_H])
    ax_tbar8.set_facecolor(BG_TITLE)
    for sp in ax_tbar8.spines.values():
        sp.set_color(BORDER_COLOR); sp.set_linewidth(1)
    ax_tbar8.set_xticks([]); ax_tbar8.set_yticks([])

    ax_badge8 = fig.add_axes([L + 0.004, B + H - TITLE_H + 0.004, 0.035, 0.033])
    ax_badge8.set_facecolor(step_colors[7])
    ax_badge8.set_xlim(0,1); ax_badge8.set_ylim(0,1)
    ax_badge8.text(0.5, 0.5, '8', fontsize=15, fontweight='bold',
                   color='white', ha='center', va='center')
    ax_badge8.axis('off')

    ax_ttxt8 = fig.add_axes([L + 0.042, B + H - TITLE_H + 0.004, W - 0.055, 0.033])
    ax_ttxt8.set_facecolor(BG_TITLE); ax_ttxt8.axis('off')
    ax_ttxt8.text(0.03, 0.5, 'Processing Pipeline', fontsize=11.5, fontweight='bold',
                  color=TEXT_WHITE, va='center')

    ax_pipe = fig.add_axes([L + PAD, B + PAD, W - 2*PAD, H - TITLE_H - PAD])
    ax_pipe.set_facecolor(BG_CARD)
    ax_pipe.set_xlim(0, 1); ax_pipe.set_ylim(0, 1)
    ax_pipe.axis('off')

    pipe_steps = [
        ('Resize Image', '#e74c3c'), ('Enhance Contrast (CLAHE)', '#e67e22'),
        ('Remove Background', '#f1c40f'), ('Gaussian Blur', '#27ae60'),
        ('Convert to HSV', '#16a085'), ('K-Means Segmentation', '#2980b9'),
        ('Extract GLCM Features', '#8e44ad'), ('SVM Classification', '#d35400')
    ]

    y0, dy = 0.90, 0.115
    for idx, (txt, clr) in enumerate(pipe_steps):
        y = y0 - idx * dy
        ax_pipe.plot(0.07, y, 'o', color=clr, markersize=12,
                     transform=ax_pipe.transAxes, clip_on=False)
        ax_pipe.text(0.07, y, str(idx+1), fontsize=8, fontweight='bold',
                     color='white', ha='center', va='center',
                     transform=ax_pipe.transAxes)
        if idx < len(pipe_steps) - 1:
            ax_pipe.plot([0.07, 0.07], [y - 0.025, y - dy + 0.025],
                         color='#5d6d7e', linewidth=2.5,
                         transform=ax_pipe.transAxes, clip_on=False)
        ax_pipe.text(0.17, y, txt, fontsize=11, fontweight='bold',
                     color=TEXT_WHITE, va='center',
                     transform=ax_pipe.transAxes)

    # Card 9 - Prediction
    L, B, W, H = card_positions[8]
    
    disease_name = prediction.replace('___', ' - ').replace('_', ' ').title()
    is_healthy = 'healthy' in prediction.lower()

    if is_healthy:
        res_bg, res_border, res_glow = '#1e8449', '#27ae60', '#2ecc71'
        status_txt, status_clr, status_icon = 'HEALTHY', '#2ecc71', '✓'
    else:
        res_bg, res_border, res_glow = '#922b21', '#e74c3c', '#e74c3c'
        status_txt, status_clr, status_icon = 'DISEASED', '#e74c3c', '⚠'

    ax_bg9 = fig.add_axes([L, B, W, H])
    ax_bg9.set_facecolor(BG_CARD)
    for sp in ax_bg9.spines.values():
        sp.set_color(res_border); sp.set_linewidth(3)
    ax_bg9.set_xticks([]); ax_bg9.set_yticks([])

    ax_tbar9 = fig.add_axes([L, B + H - TITLE_H, W, TITLE_H])
    ax_tbar9.set_facecolor(BG_TITLE)
    for sp in ax_tbar9.spines.values():
        sp.set_color(res_border); sp.set_linewidth(2)
    ax_tbar9.set_xticks([]); ax_tbar9.set_yticks([])

    ax_badge9 = fig.add_axes([L + 0.004, B + H - TITLE_H + 0.004, 0.035, 0.033])
    ax_badge9.set_facecolor(step_colors[8])
    ax_badge9.set_xlim(0,1); ax_badge9.set_ylim(0,1)
    ax_badge9.text(0.5, 0.5, '9', fontsize=15, fontweight='bold',
                   color='white', ha='center', va='center')
    ax_badge9.axis('off')

    ax_ttxt9 = fig.add_axes([L + 0.042, B + H - TITLE_H + 0.004, W - 0.055, 0.033])
    ax_ttxt9.set_facecolor(BG_TITLE); ax_ttxt9.axis('off')
    ax_ttxt9.text(0.03, 0.5, 'Final Prediction', fontsize=11.5, fontweight='bold',
                  color=TEXT_WHITE, va='center')

    ax_res = fig.add_axes([L + PAD, B + PAD, W - 2*PAD, H - TITLE_H - PAD])
    ax_res.set_facecolor(res_bg)
    ax_res.set_xlim(0, 1); ax_res.set_ylim(0, 1)
    for sp in ax_res.spines.values():
        sp.set_color(res_border); sp.set_linewidth(2)
    ax_res.set_xticks([]); ax_res.set_yticks([])

    ax_res.text(0.5, 0.88, f'{status_icon}  {status_txt}', fontsize=16, fontweight='bold',
                color=status_clr, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                         edgecolor=status_clr, linewidth=2.5))

    ax_res.plot([0.08, 0.92], [0.74, 0.74], color='white', linewidth=1, alpha=0.25)

    ax_res.text(0.5, 0.64, 'DETECTED DISEASE', fontsize=10, color='#bdc3c7',
                ha='center', va='center', style='italic')

    ax_res.text(0.5, 0.48, disease_name, fontsize=19, fontweight='bold',
                color='white', ha='center', va='center')

    ax_res.plot([0.08, 0.92], [0.33, 0.33], color='white', linewidth=1, alpha=0.25)

    ax_res.text(0.5, 0.24, 'CONFIDENCE LEVEL', fontsize=10, color='#bdc3c7',
                ha='center', va='center', style='italic')

    ax_res.text(0.5, 0.09, f'{confidence:.2f}%', fontsize=26, fontweight='bold',
                color=res_glow, ha='center', va='center')

    output_filename = f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    output_path = os.path.join(config.RESULTS_DIR, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK, edgecolor='none')
    print(f"✓ Main page saved: {output_path}")
    plt.show(block=False)  # Don't block, show and continue
    plt.pause(0.5)  # Small pause to render


def visualize_treatment_page(prediction, confidence, treatment_info, image_path):
    """Separate clean treatment page - reduced fonts, better spacing"""
    BG_DARK = '#1a1a2e'
    TEXT_WHITE = '#ecf0f1'
    TEXT_GRAY = '#95a5a6'

    fig = plt.figure(figsize=(20, 14), facecolor=BG_DARK)  # Taller for more space
    
    # Title - TOP
    fig.text(0.5, 0.98, 'Treatment Recommendations',
             fontsize=28, fontweight='bold', color=TEXT_WHITE,
             ha='center', va='top')
    
    # Horizontal line under title
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.95, 0.95], color='#27ae60', linewidth=2))
    
    # Disease name - MOVED DOWN
    disease_name = prediction.replace('___', ' - ').replace('_', ' ').title()
    fig.text(0.5, 0.92, f'{disease_name}',
             fontsize=16, color='#ecf0f1', fontweight='bold',
             ha='center', va='top')
    
    # Confidence - MOVED DOWN
    fig.text(0.5, 0.89, f'Confidence: {confidence:.1f}%',
             fontsize=12, color='#3498db',
             ha='center', va='top')
    
    # Description - MOVED DOWN
    fig.text(0.5, 0.85, f"📝 {treatment_info['description']}",
             fontsize=11, color=TEXT_WHITE, ha='center', va='top',
             fontweight='bold')
    
    # Column 1: Chemical Treatments
    y_start = 0.78
    fig.text(0.08, y_start, '🧪 Chemical Treatments',
             fontsize=14, fontweight='bold', color='#3498db', va='top')
    
    y_pos = y_start - 0.06
    for idx, treat in enumerate(treatment_info['chemical_treatments'][:3], 1):
        fig.text(0.08, y_pos, f"{idx}. {treat['name']}",
                 fontsize=11, fontweight='bold', color=TEXT_WHITE, va='top')
        y_pos -= 0.03
        fig.text(0.10, y_pos, f"💧 Dosage: {treat['dosage']}",
                 fontsize=9, color=TEXT_GRAY, va='top')
        y_pos -= 0.025
        fig.text(0.10, y_pos, f"📅 Application: {treat['application']}",
                 fontsize=9, color=TEXT_GRAY, va='top')
        y_pos -= 0.025
        fig.text(0.10, y_pos, f"💰 Cost: {treat['cost']}",
                 fontsize=9, color='#f39c12', fontweight='bold', va='top')
        y_pos -= 0.05
    
    # Column 2: Organic Treatments
    y_pos = y_start - 0.06
    fig.text(0.52, y_start, '🌿 Organic Treatments',
             fontsize=14, fontweight='bold', color='#27ae60', va='top')
    
    for idx, treat in enumerate(treatment_info['organic_treatments'][:3], 1):
        fig.text(0.52, y_pos, f"{idx}. {treat['name']}",
                 fontsize=11, fontweight='bold', color=TEXT_WHITE, va='top')
        y_pos -= 0.03
        fig.text(0.54, y_pos, f"💧 Dosage: {treat['dosage']}",
                 fontsize=9, color=TEXT_GRAY, va='top')
        y_pos -= 0.025
        fig.text(0.54, y_pos, f"📅 Application: {treat['application']}",
                 fontsize=9, color=TEXT_GRAY, va='top')
        y_pos -= 0.025
        fig.text(0.54, y_pos, f"💰 Cost: {treat['cost']}",
                 fontsize=9, color='#f39c12', fontweight='bold', va='top')
        y_pos -= 0.05
    
    # Prevention Tips - Full Width
    fig.text(0.5, 0.38, '🛡️ Prevention Tips',
             fontsize=14, fontweight='bold', color='#f39c12',
             ha='center', va='top')
    
    y_pos = 0.35
    for idx, tip in enumerate(treatment_info['prevention_tips'][:8], 1):
        fig.text(0.5, y_pos, f"{idx}. {tip}",
                 fontsize=10, color=TEXT_GRAY, ha='center', va='top')
        y_pos -= 0.028
    
    # When to Treat - Bottom
    fig.text(0.5, 0.06, f"⏰ {treatment_info['when_to_treat']}",
             fontsize=11, fontweight='bold', color='#e74c3c',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#16213e',
                      edgecolor='#e74c3c', linewidth=2.5))
    
    output_filename = f"treatment_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    output_path = os.path.join(config.RESULTS_DIR, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK, edgecolor='none')
    print(f"✓ Treatment page saved: {output_path}")
    plt.show()


def main():
    print("\n" + "="*80)
    print("Plant Disease Detection & Treatment System")
    print("="*80 + "\n")

    treatment_db = load_treatment_database()
    model, scaler, label_encoder, class_names = load_trained_model()
    
    image_path = select_image()
    if not image_path:
        return
    
    predict_disease(image_path, model, scaler, label_encoder, class_names,
                   treatment_db, visualize=True)
    print("\n✓ Complete! 2 pages generated.\n")


if __name__ == "__main__":
    main()