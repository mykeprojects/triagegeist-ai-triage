import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer

app = FastAPI(title="TriageGeist API - Production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CARGA DE MODELOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model_demo.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features_demo.pkl")
model = joblib.load(MODEL_PATH)
FEATURES_REDUCED_ORDER = joblib.load(FEATURES_PATH)
embedding_model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

# --- API LOGIC ---

class TriageRequest(BaseModel):
    heart_rate: float
    respiratory_rate: float
    spo2: float
    systolic_bp: float
    diastolic_bp: float
    temperature_c: float
    pain_score: Optional[float] = None
    age: float
    mental_status_triage: str
    arrival_mode: str
    sex: str
    chief_complaint: str

MENTAL_STATUS_MAP = {"alert": 0, "agitated": 1, "confused": 2, "drowsy": 3, "unresponsive": 4}


def detect_zero_based_classes(class_labels) -> bool:
    """Detecta si las clases del modelo están codificadas como 0-4."""
    try:
        class_values = [int(c) for c in class_labels]
    except Exception:
        return False

    return min(class_values) == 0 and max(class_values) <= 4


def to_esi_1_to_5(class_value: int, is_zero_based: bool) -> int:
    """Mapea clase de salida del modelo a ESI 1-5 y limita el rango."""
    mapped = int(class_value) + 1 if is_zero_based else int(class_value)
    return max(1, min(5, mapped))


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: TriageRequest):
    try:
        # 1. Embedding BioClinicalBERT (768-d)
        text_vector = embedding_model.encode([data.chief_complaint])[0]
        if text_vector.shape[0] != 768:
            raise ValueError(f"Embedding dimension mismatch: expected 768, got {text_vector.shape[0]}")

        # 2. Encoding categórico igual a FEATURES_REDUCED del notebook.
        mental_val = MENTAL_STATUS_MAP.get(data.mental_status_triage.lower(), 0)
        arrival_mode = data.arrival_mode.strip().lower()
        sex = data.sex.strip().upper()

        arrival_mode_ambulance = 1.0 if arrival_mode == "ambulance" else 0.0
        arrival_mode_helicopter = 1.0 if arrival_mode == "helicopter" else 0.0
        arrival_mode_walk_in = 1.0 if arrival_mode == "walk-in" else 0.0
        sex_m = 1.0 if sex == "M" else 0.0

        pain_score = np.nan if data.pain_score is None else float(data.pain_score)
        
        # 3. Construir features estructuradas por nombre y ordenar con features_demo.pkl.
        feature_values = {
            "heart_rate": float(data.heart_rate),
            "respiratory_rate": float(data.respiratory_rate),
            "spo2": float(data.spo2),
            "systolic_bp": float(data.systolic_bp),
            "diastolic_bp": float(data.diastolic_bp),
            "temperature_c": float(data.temperature_c),
            "pain_score": pain_score,
            "age": float(data.age),
            "mental_status_triage": float(mental_val),
            "arrival_mode_ambulance": arrival_mode_ambulance,
            "arrival_mode_helicopter": arrival_mode_helicopter,
            "arrival_mode_walk-in": arrival_mode_walk_in,
            "sex_M": sex_m,
        }

        missing_features = [f for f in FEATURES_REDUCED_ORDER if f not in feature_values]
        if missing_features:
            raise ValueError(f"Missing structured features required by features_demo.pkl: {missing_features}")

        numeric_features = [feature_values[f] for f in FEATURES_REDUCED_ORDER]

        if len(numeric_features) != len(FEATURES_REDUCED_ORDER):
            raise ValueError(
                f"Feature count mismatch: expected {len(FEATURES_REDUCED_ORDER)}, got {len(numeric_features)}"
            )

        # 4. Concatenación final (13 structured + 768 embeddings = 781)
        final_input = np.concatenate([np.array(numeric_features), text_vector]).reshape(1, -1)

        # 5. Inferencia
        prediction = int(model.predict(final_input)[0])
        probabilities = model.predict_proba(final_input)[0]
        class_labels = getattr(model, "classes_", np.arange(len(probabilities)))
        is_zero_based = detect_zero_based_classes(class_labels)

        # Ajuste de ESI para exponer siempre 1-5 al frontend.
        esi_result = to_esi_1_to_5(prediction, is_zero_based)

        probabilities_by_esi = {f"ESI_{i}": 0.0 for i in range(1, 6)}
        for class_label, p in zip(class_labels, probabilities):
            mapped_esi = to_esi_1_to_5(int(class_label), is_zero_based)
            probabilities_by_esi[f"ESI_{mapped_esi}"] += round(float(p) * 100, 1)

        return {
            "esi_level": esi_result,
            "confidence": round(float(np.max(probabilities)) * 100, 1),
            "probabilities": probabilities_by_esi
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)