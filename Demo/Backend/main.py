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
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_final.pkl")
model = joblib.load(MODEL_PATH)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- FUNCIONES DE CÁLCULO CLÍNICO ---

def calculate_map(sbp, dbp):
    """Calcula la Presión Arterial Media (MAP)"""
    return (sbp + 2 * dbp) / 3

def calculate_shock_index(hr, sbp):
    """Calcula el Índice de Choque (Shock Index)"""
    if sbp > 0:
        return hr / sbp
    return 0.0

def calculate_news2(hr, rr, sbp, temp, spo2, gcs, mental_status):
    """
    Cálculo simplificado del score NEWS2 (National Early Warning Score 2)
    Nota: Se asume aire ambiente (no oxígeno suplementario) para este cálculo.
    """
    score = 0
    # Respiration Rate
    if rr <= 8 or rr >= 25: score += 3
    elif 21 <= rr <= 24: score += 2
    elif 9 <= rr <= 11: score += 1
    # SpO2
    if spo2 <= 91: score += 3
    elif 92 <= spo2 <= 93: score += 2
    elif 94 <= spo2 <= 95: score += 1
    # Systolic BP
    if sbp <= 90 or sbp >= 220: score += 3
    elif 91 <= sbp <= 100: score += 2
    elif 101 <= sbp <= 110: score += 1
    # Heart Rate
    if hr <= 40 or hr >= 131: score += 3
    elif 111 <= hr <= 130: score += 2
    elif 41 <= hr <= 50 or 91 <= hr <= 110: score += 1
    # Consciousness (GCS/Mental Status)
    if mental_status.lower() != "alert" or gcs < 15: score += 3
    # Temperature
    if temp <= 35.0: score += 3
    elif temp >= 39.1: score += 2
    elif 35.1 <= temp <= 36.0 or 38.1 <= temp <= 39.0: score += 1
    
    return float(score)

# --- API LOGIC ---

class TriageRequest(BaseModel):
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    respiratory_rate: float
    temperature_c: float
    spo2: float
    gcs_total: float
    pain_score: float
    mental_status_triage: str
    num_prior_ed_visits_12m: int
    num_comorbidities: int
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
        # 1. Cálculos derivados
        map_val = calculate_map(data.systolic_bp, data.diastolic_bp)
        si_val = calculate_shock_index(data.heart_rate, data.systolic_bp)
        news2_val = calculate_news2(
            data.heart_rate, data.respiratory_rate, data.systolic_bp, 
            data.temperature_c, data.spo2, data.gcs_total, data.mental_status_triage
        )
        
        # 2. Embedding del texto (384-d)
        text_vector = embedding_model.encode([data.chief_complaint])[0]

        # 3. Construcción del vector siguiendo el ORDEN EXACTO del entrenamiento:
        # 1: mental_status_triage, 2: num_prior_ed_visits_12m, 3: num_comorbidities, 
        # 4: systolic_bp, 5: diastolic_bp, 6: mean_arterial_pressure, 
        # 7: heart_rate, 8: respiratory_rate, 9: temperature_c, 10: spo2, 
        # 11: gcs_total, 12: pain_score, 13: shock_index, 14: news2_score
        
        mental_val = MENTAL_STATUS_MAP.get(data.mental_status_triage.lower(), 0)
        
        numeric_features = [
            float(mental_val),
            float(data.num_prior_ed_visits_12m),
            float(data.num_comorbidities),
            float(data.systolic_bp),
            float(data.diastolic_bp),
            float(map_val),
            float(data.heart_rate),
            float(data.respiratory_rate),
            float(data.temperature_c),
            float(data.spo2),
            float(data.gcs_total),
            float(data.pain_score),
            float(si_val),
            float(news2_val)
        ]

        # 4. Concatenación final (14 features numéricas + 384 embeddings = 398 total)
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