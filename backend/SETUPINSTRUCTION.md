# CONNECTING BACKEND TO FRONTEND

## SETUP STEPS:

### 1. Copy api_server.py to backend folder

```bash
# Copy from outputs to backend
copy api_server.py C:\Users\ELCOT\Documents\Desktop\plant_disease_detection\backend\
```

### 2. Install FastAPI requirements

```bash
cd backend
pip install fastapi uvicorn[standard] python-multipart
```

### 3. Run Backend Server

```bash
# Activate venv first
cd backend
python api_server.py
```

Backend will run on: **http://localhost:8000**

---

### 4. Frontend is already configured

Your `Dashboard.jsx` already has:
```javascript
axios.post('http://localhost:8000/predict', formData)
```

Just run frontend:

```bash
cd frontend
npm start
```

Frontend will run on: **http://localhost:3000**

---

## HOW IT WORKS:

1. **Frontend (React)**: User uploads image → sends to backend
2. **Backend (FastAPI)**: Receives image → runs your existing predict.py logic → returns JSON
3. **Frontend**: Receives JSON → displays 7 processing steps + disease + treatments

---

## FILE STRUCTURE:

```
plant_disease_detection/
├── backend/
│   ├── api_server.py        ← NEW (FastAPI wrapper)
│   ├── src/
│   │   ├── predict.py       ← Your existing logic (unchanged)
│   │   ├── preprocessing.py
│   │   ├── feature_extraction.py
│   │   └── config.py
│   ├── data/
│   │   ├── disease_treatments.json
│   │   ├── disease_translations_tamil.json
│   │   └── treatment_translations_tamil.json
│   ├── models/
│   │   ├── svm_leaf_disease_model.pkl
│   │   ├── feature_scaler.pkl
│   │   └── label_encoder.pkl
│   └── venv/
│
└── frontend/
    ├── src/
    │   ├── pages/
    │   │   ├── Dashboard.jsx
    │   │   ├── Processing.jsx
    │   │   ├── Treatment.jsx
    │   │   ├── TreatmentEnglish.jsx
    │   │   └── TreatmentTamil.jsx
    │   ├── App.js
    │   └── index.js
    └── package.json
```

---

## TESTING:

1. **Test Backend alone:**
   - Go to http://localhost:8000/docs
   - Click "POST /predict"
   - Upload test image
   - See JSON response

2. **Test Full Stack:**
   - Frontend: http://localhost:3000
   - Upload image
   - See processing steps
   - View treatments in English/Tamil

---

## API RESPONSE FORMAT:

```json
{
  "success": true,
  "prediction": "Tomato___Early_blight",
  "confidence": 91.4,
  "disease_name": "Tomato - Early Blight",
  "tamil_name": "தக்காளி - ஆரம்ப கருகல் நோய்",
  "processing_steps": [
    {"title": "Original Image", "image": "base64..."},
    {"title": "Resized (256x256)", "image": "base64..."},
    ...7 steps total
  ],
  "treatment_english": {
    "description": "...",
    "chemical_treatments": [...],
    "organic_treatments": [...]
  },
  "treatment_tamil": {
    "chemical_treatments_tamil": [...],
    "organic_treatments_tamil": [...]
  },
  "is_healthy": false
}
```

---

## DONE! 🎉

Your existing backend logic (predict.py) is **UNCHANGED**.
FastAPI just wraps it to serve the frontend.