"""
FastAPI REST API Microservice for Document Classification
PT Mobil Sejuta Umat / RVL-CDIP Document AI
Tugas 05 - Kelompok 05
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ocr_preprocessor import clean_noisy_ocr_text, TextCleanerTransformer

app = FastAPI(
    title="Document AI API - PT Mobil Sejuta Umat",
    description="Layanan REST API Otomasi Verifikasi & Klasifikasi Dokumen Finansial/Operasional",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Pipeline & Metadata
MODEL_PATH = os.path.join(script_dir, "model_final_terbaik.joblib")
METADATA_PATH = os.path.join(script_dir, "model_metadata.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Artefak model tidak ditemukan di {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

CLASS_NAMES = ['form', 'invoice', 'letter', 'memo']
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
        if "target_classes" in meta_data:
            CLASS_NAMES = meta_data["target_classes"]

# Pydantic Schemas
class DocumentInput(BaseModel):
    Jumlah_Kata: int = Field(
        ...,
        ge=0,
        le=50000,
        description="Jumlah kata total dari dokumen pindaian OCR",
        example=185
    )
    Rasio_Angka_Teks: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Rasio karakter angka terhadap total karakter teks (0.0 s.d. 1.0)",
        example=0.38
    )
    Ada_Kata_Kunci_Finansial: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indikator biner adanya kata kunci finansial (1 = Ada, 0 = Tidak)",
        example=1
    )
    Teks_OCR: str = Field(
        ...,
        min_length=1,
        description="Teks mentah hasil ekstraksi OCR dokumen",
        example="INVOICE NO: 8849 TOTAL AMOUNT DUE: $48,500.00 COMPANY: AKEBONO BRAKE"
    )

    @field_validator('Teks_OCR')
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Teks_OCR tidak boleh kosong atau hanya berisi spasi.")
        return v

class PredictionResponse(BaseModel):
    status: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    is_ambiguous_warning_zone: bool
    recommended_action: str
    latency_ms: float

# Routes
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Document AI Inference API PT Mobil Sejuta Umat is active.",
        "documentation": "/docs",
        "endpoints": ["/predict", "/health", "/metadata"]
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": [str(c).upper() for c in CLASS_NAMES],
        "timestamp": time.time()
    }

@app.get("/metadata", tags=["Metadata"])
def get_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Metadata file not found."}

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict_document(payload: DocumentInput):
    t_start = time.perf_counter()
    
    # Construct DataFrame
    input_df = pd.DataFrame([{
        'Jumlah_Kata': payload.Jumlah_Kata,
        'Rasio_Angka_Teks': payload.Rasio_Angka_Teks,
        'Ada_Kata_Kunci_Finansial': payload.Ada_Kata_Kunci_Finansial,
        'Teks_OCR': payload.Teks_OCR
    }])
    
    try:
        probabilities = pipeline.predict_proba(input_df)[0]
        prediction_idx = int(np.argmax(probabilities))
        classes = [str(c).upper() for c in CLASS_NAMES]
        predicted_class = classes[prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        prob_dict = {classes[i]: round(float(probabilities[i]), 4) for i in range(len(classes))}
        
        is_ambiguous = 0.40 <= confidence <= 0.60
        recommended_action = (
            "ESKALASI KE SUPERVISOR (Human-in-the-Loop): Dokumen berada di zona ambigu."
            if is_ambiguous else
            "OTOMATISASI DISETUJUI: Klasifikasi memiliki keyakinan tinggi."
        )
        
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        
        return PredictionResponse(
            status="success",
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            is_ambiguous_warning_zone=is_ambiguous,
            recommended_action=recommended_action,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat memproses inferensi: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
