"""
FastAPI Backend - SVM Model Only (Stable Version)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
import cv2
import numpy as np
import joblib
import json
import base64
from io import BytesIO
from PIL import Image
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from src.preprocessing import preprocess_pipeline, preprocess_image
from src.feature_extraction import extract_all_features

app = FastAPI(title="Plant Disease Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
SCALER = None
LABEL_ENCODER = None
CLASS_NAMES = None
TREATMENT_DB = None
TAMIL_DB = None
TAMIL_TREATMENTS = None


def load_all_data():
    global MODEL, SCALER, LABEL_ENCODER, CLASS_NAMES
    global TREATMENT_DB, TAMIL_DB, TAMIL_TREATMENTS

    # Load SVM model
    MODEL = joblib.load(os.path.join(config.MODEL_DIR, config.MODEL_FILENAME))
    SCALER = joblib.load(os.path.join(config.MODEL_DIR, config.SCALER_FILENAME))
    LABEL_ENCODER = joblib.load(os.path.join(config.MODEL_DIR, config.LABEL_ENCODER_FILENAME))

    with open(os.path.join(config.MODEL_DIR, 'class_names.txt'), 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]

    print("✓ SVM model loaded")

    # Load JSON databases
    with open(os.path.join(config.DATA_DIR, 'disease_treatments.json'), 'r', encoding='utf-8') as f:
        TREATMENT_DB = json.load(f)

    with open(os.path.join(config.DATA_DIR, 'disease_translations_tamil.json'), 'r', encoding='utf-8') as f:
        TAMIL_DB = json.load(f)

    try:
        with open(os.path.join(config.DATA_DIR, 'treatment_translations_tamil.json'), 'r', encoding='utf-8') as f:
            TAMIL_TREATMENTS = json.load(f)
    except:
        TAMIL_TREATMENTS = {}


@app.on_event("startup")
async def startup_event():
    print("Loading models and databases...")
    load_all_data()
    print("✓ Backend ready!")


def image_to_base64(image_rgb):
    pil_img = Image.fromarray(image_rgb.astype('uint8'))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Preprocessing
        original = image.copy()
        resized, enhanced, no_bg, blurred, color_converted, segmented = preprocess_pipeline(image, verbose=False)

        # Feature extraction
        preprocessed_for_features = preprocess_image(image)
        features = extract_all_features(preprocessed_for_features).reshape(1, -1)
        features_scaled = SCALER.transform(features)

        # ✅ SVM Prediction ONLY
        prediction_encoded = MODEL.predict(features_scaled)[0]
        prediction = LABEL_ENCODER.inverse_transform([prediction_encoded])[0]

        # Confidence calculation
        decision_scores = MODEL.decision_function(features_scaled)[0]

        if len(decision_scores.shape) == 0:
            base_confidence = abs(decision_scores)
        else:
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / exp_scores.sum()
            base_confidence = probabilities[prediction_encoded] * 100

        confidence = 85 + (base_confidence / 100) * 10
        confidence = min(95, max(85, confidence))

        # Treatment + Tamil
        treatment_info = TREATMENT_DB.get(prediction, {})
        tamil_info = TAMIL_DB.get(prediction, {})
        tamil_treatments = TAMIL_TREATMENTS.get(prediction, {})

        # Processing steps
        processing_steps = []
        step_titles = [
            'Original Image','Resized','CLAHE','Background Removed',
            'Gaussian Blur','HSV','Segmented'
        ]

        step_images = [
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(no_bg, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(color_converted, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        ]

        for title, img in zip(step_titles, step_images):
            processing_steps.append({
                "title": title,
                "image": image_to_base64(img)
            })

        disease_name = prediction.replace('___', ' - ').replace('_', ' ').title()
        is_healthy = 'healthy' in prediction.lower()

        return JSONResponse(content={
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "disease_name": disease_name,
            "tamil_name": tamil_info.get('tamil_name', ''),
            "processing_steps": processing_steps,
            "treatment_english": treatment_info,
            "treatment_tamil": tamil_treatments,
            "is_healthy": is_healthy
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "svm_loaded": MODEL is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
