from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import os
import sys

# Proje ana dizinini path'e ekle (src modüllerine erişim için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.schemas import TransactionRequest, AnalysisResponse
from src.fzsl.fzsl_model import FZSLModel, FZSLPredictor
from src.fzsl.zsl_encoder import TextClassEncoder
from src.fzsl.class_descriptions import FRAUD_CLASS_DESCRIPTIONS, SEEN_CLASSES, UNSEEN_CLASS

app = FastAPI(
    title="FraudDetection FZSL API",
    description="Federated Zero-Shot Learning ile Anomali Tespiti ve Yeni Nesil Dolandırıcılık Yakalama Sistemi",
    version="1.0.0"
)

# Global değişkenler
scaler = None
predictor = None
stats = {
    "total_requests": 0,
    "normal_count": 0,
    "fraud_count": 0,
    "unknown_new_fraud_count": 0
}

@app.on_event("startup")
async def startup_event():
    global scaler, predictor
    print("[API] Baslatiliyor, bilesenler yukleniyor...")

    # 1. Scaler hazirligi (Egitim verisinden)
    csv_path = "data/creditcard.csv"
    if not os.path.exists(csv_path):
        print(f"[UYARI] {csv_path} bulunamadi. Scaler hazirlanamiyor.")
    else:
        print("[API] Veriseti yuklenip scaler fit ediliyor...")
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != "Class"]
        X = df[feature_cols].values.astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(X)
        print("[API] Scaler hazir.")

    # 2. Text Embedding'lerin hazirlanmasi
    print("[API] Text embedding'ler SBERT ile hesaplaniyor...")
    text_encoder = TextClassEncoder(embed_dim=128, backend="local")
    all_class_order = SEEN_CLASSES + [UNSEEN_CLASS]
    all_text_emb_matrix, all_class_order = text_encoder.get_class_embedding_matrix(
        FRAUD_CLASS_DESCRIPTIONS, class_order=all_class_order
    )
    text_dim = all_text_emb_matrix.shape[1]

    # 3. Model yuklenmesi
    model_path = "checkpoints/fzsl_model.pt"
    if not os.path.exists(model_path):
        print(f"[UYARI] Model dosyasi bulunamadi: {model_path}")
        print("[UYARI] Lutfen once 'python demo.py --mode fzsl' komutunu calistirin.")
    else:
        # Input dim genellikle 30'dur (Time, V1..V28, Amount)
        input_dim = 30
        model = FZSLModel(
            input_dim=input_dim,
            text_dim=text_dim,
            proj_dim=128,
            temperature=0.07
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        # 4. Predictor baslatma
        predictor = FZSLPredictor(
            model=model,
            class_text_embeddings=all_text_emb_matrix,
            class_order=all_class_order,
            device="cpu",
            optimal_threshold=0.5 # Default (gerekirse pr_curve sonuclarina gore degistirilebilir)
        )
        print("[API] Model ve Predictor basariyla yuklendi.")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/stats")
async def get_stats():
    return stats

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_transaction(req: TransactionRequest):
    global stats

    if predictor is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model veya Scaler henuz hazir degil.")

    stats["total_requests"] += 1

    # Numpy array'e cevir
    features = [
        req.Time, req.V1, req.V2, req.V3, req.V4, req.V5, req.V6, req.V7, req.V8, req.V9,
        req.V10, req.V11, req.V12, req.V13, req.V14, req.V15, req.V16, req.V17, req.V18, req.V19,
        req.V20, req.V21, req.V22, req.V23, req.V24, req.V25, req.V26, req.V27, req.V28, req.Amount
    ]
    X_raw = np.array([features], dtype=np.float32)

    # Olceklendir
    X_scaled = scaler.transform(X_raw)

    # Tahmin yap
    pred_classes, sims = predictor.predict(X_scaled)
    pred_class = pred_classes[0]
    similarity_score = sims[0].max() # En yuksek benzerlik skoru

    # Yorumlama
    is_fraud = pred_class != "normal"
    fraud_type = pred_class

    # Yeni Fraud Alarm Mekanizmasi (UNKNOWN_NEW_FRAUD)
    # Eger tahmin 'fraud_type_3' ise (eğitimde hiç görülmeyen sınıf)
    if is_fraud and fraud_type == UNSEEN_CLASS:
        fraud_type = "UNKNOWN_NEW_FRAUD"
        message = "DIKKAT: Daha once hic gorulmemis, yeni bir fraud paterni (Zero-Shot) tespit edildi!"
        stats["unknown_new_fraud_count"] += 1
    elif is_fraud:
        message = f"Fraud tespit edildi. Bilinen patern: {fraud_type}."
        stats["fraud_count"] += 1
    else:
        message = "Islem normal gorunuyor."
        stats["normal_count"] += 1

    return AnalysisResponse(
        is_fraud=is_fraud,
        fraud_type=fraud_type,
        confidence=float(similarity_score),
        message=message
    )
