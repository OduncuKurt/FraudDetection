"""
dashboard_api.py
----------------
FraudDetection Dashboard için FastAPI Backend
Gerçek model olmasa bile bağımsız simülasyon modu destekler.

Başlatma:
    uvicorn backend.dashboard_api:app --reload --port 8001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import random
import time
import math
import numpy as np
from collections import deque
from typing import Optional

app = FastAPI(
    title="FraudDetection Dashboard API",
    description="Canlı Dashboard için Simülasyon + İstatistik API",
    version="2.0.0"
)

# CORS — Frontend'den erişim için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statik dosyaları sun (dashboard klasöründen)
_dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
if os.path.isdir(_dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")

@app.get("/")
async def root():
    idx = os.path.join(_dashboard_dir, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "FraudDetection Dashboard API", "dashboard": "/dashboard"}

# ---------------------------------------------------------------------------
#  Global Durum (In-Memory Simulation State)
# ---------------------------------------------------------------------------

_state = {
    "total": 0,
    "normal": 0,
    "fraud_known": 0,
    "fraud_unknown": 0,
    "start_time": time.time(),
    "recent_transactions": deque(maxlen=200),
    "fraud_alerts": deque(maxlen=50),
    "fraud_type_counts": {
        "fraud_type_0": 0,
        "fraud_type_1": 0,
        "fraud_type_2": 0,
        "UNKNOWN_NEW_FRAUD": 0,
    },
    "hourly_totals": [0] * 24,
    "hourly_fraud":  [0] * 24,
    "last_hour_idx": 0,
    "amounts_processed": 0.0,
    "fraud_amounts": 0.0,
}

# Gerçek FZSL model metrikleri (eğitim sonuçlarından alındı)
REAL_METRICS = {
    "centralized": {
        "precision": 0.9289, "recall": 0.9388, "f1": 0.9338,
        "roc_auc": 0.9991, "pr_auc": 0.7741
    },
    "federated": {
        "precision": 0.9373, "recall": 1.0000, "f1": 0.9676,
        "roc_auc": 1.0000, "pr_auc": 0.9942
    },
    "fzsl_full": {
        "precision": 0.9579, "recall": 0.9715, "f1": 0.9647,
        "roc_auc": 1.0000, "pr_auc": 0.9934
    },
    "fzsl_unseen_detection_rate": 0.9831,
}

# Fraud tipi açıklamaları (doğal dil)
FRAUD_DESCRIPTIONS = {
    "fraud_type_0": {
        "title": "Yüksek Değerli Dolandırıcılık",
        "description": "Çok yüksek tutarlı anormal alışveriş. Kart kopyalama veya büyük ölçekli çevrimiçi dolandırıcılık. V14, V4, V12 özellikleri en belirleyici.",
        "color": "#FF4757",
        "icon": "💳",
        "top_features": ["V14", "V4", "V12", "V3", "V10"],
        "avg_amount": 172.80,
    },
    "fraud_type_1": {
        "title": "Hesap Ele Geçirme",
        "description": "Orta tutarlı, davranışsal sapmalar. Meşru hesaba yetkisiz erişim. V3, V17, V7 özellikleri kritik.",
        "color": "#FFA502",
        "icon": "🔑",
        "top_features": ["V3", "V17", "V7", "V1", "V12"],
        "avg_amount": 96.03,
    },
    "fraud_type_2": {
        "title": "Mikro İşlem Testi",
        "description": "Çok küçük tutarlı kart test işlemleri. Büyük fraud öncesi kart aktiflik testi. V7, V3, V1 özellikleri belirleyici.",
        "color": "#ECCC68",
        "icon": "🔍",
        "top_features": ["V7", "V3", "V1", "V10", "V5"],
        "avg_amount": 2.22,
    },
    "UNKNOWN_NEW_FRAUD": {
        "title": "⚠️ Bilinmeyen Yeni Fraud Tipi",
        "description": "Zero-Shot Learning ile tespit! Bu fraud tipi eğitimde hiç görülmedi. Para aklama şüphesi — işlem miktarları normal görünse de gizli yapısal anomaliler mevcut.",
        "color": "#7C2FFF",
        "icon": "🚨",
        "top_features": ["V14", "V17", "V12", "V3", "V10"],
        "avg_amount": 87.03,
    },
}

# SHAP değerleri (gerçek eğitim verisinden alınan ortalamalar)
REAL_SHAP_TEMPLATES = {
    "fraud_type_0": {
        "V14": +0.312, "V4": +0.289, "V12": -0.241, "V3": +0.198, "V10": -0.187,
        "V11": +0.143, "V17": -0.121, "V1": -0.098, "V2": +0.076, "Amount": +0.065,
    },
    "fraud_type_1": {
        "V3": +0.287, "V17": -0.253, "V7": +0.231, "V1": -0.198, "V12": +0.167,
        "V14": -0.142, "V10": +0.119, "V4": +0.087, "V16": -0.073, "Amount": +0.054,
    },
    "fraud_type_2": {
        "V7": +0.341, "V3": +0.298, "V1": -0.276, "V10": +0.214, "V5": -0.189,
        "V14": +0.133, "V2": +0.112, "V12": -0.098, "Amount": -0.076, "V4": +0.057,
    },
    "UNKNOWN_NEW_FRAUD": {
        "V14": +0.298, "V17": -0.271, "V12": +0.247, "V3": +0.221, "V10": -0.198,
        "V16": -0.167, "V4": +0.143, "V1": -0.121, "V2": +0.087, "Time": +0.065,
    },
}


def _gen_transaction_id():
    return f"TXN-{random.randint(100000, 999999)}"


def _gen_transaction(force_fraud_type: Optional[str] = None):
    """Simüle edilmiş bir işlem üretir."""
    _state["total"] += 1
    hour = int((time.time() - _state["start_time"]) / 3600) % 24

    # Fraud oranı ~%0.17 (gerçek veri seti)
    roll = random.random()
    if force_fraud_type:
        fraud_type = force_fraud_type
        is_fraud = True
        is_unknown = (fraud_type == "UNKNOWN_NEW_FRAUD")
    elif roll < 0.0017:  # ~%0.17
        is_fraud = True
        is_unknown = random.random() < 0.20  # %20 ihtimal UNKNOWN
        if is_unknown:
            fraud_type = "UNKNOWN_NEW_FRAUD"
        else:
            # Gerçek dağılım: type0: 207, type1: 92, type2: 15
            r = random.random()
            if r < 0.655:
                fraud_type = "fraud_type_0"
            elif r < 0.946:
                fraud_type = "fraud_type_1"
            else:
                fraud_type = "fraud_type_2"
    else:
        is_fraud = False
        is_unknown = False
        fraud_type = "normal"

    # Tutar
    if fraud_type == "fraud_type_0":
        amount = round(random.lognormvariate(4.5, 1.2), 2)
    elif fraud_type == "fraud_type_2":
        amount = round(random.uniform(0.01, 9.99), 2)
    elif fraud_type == "UNKNOWN_NEW_FRAUD":
        amount = round(random.lognormvariate(4.2, 0.9), 2)
    elif is_fraud:
        amount = round(random.lognormvariate(4.0, 1.0), 2)
    else:
        amount = round(random.lognormvariate(3.2, 1.3), 2)

    confidence = round(random.uniform(0.82, 0.99), 4) if is_fraud else round(random.uniform(0.91, 0.99), 4)

    # V1..V28 simüle et
    v_features = {}
    for i in range(1, 29):
        if is_fraud and f"V{i}" in SHAP_TEMPLATE_FOR(fraud_type):
            base = SHAP_TEMPLATE_FOR(fraud_type)[f"V{i}"] * random.uniform(0.7, 1.4)
        else:
            base = random.gauss(0, 1)
        v_features[f"V{i}"] = round(base, 6)

    txn = {
        "id": _gen_transaction_id(),
        "timestamp": time.time(),
        "amount": amount,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
        "confidence": confidence,
        "features": v_features,
        "message": _make_message(is_fraud, is_unknown, fraud_type),
    }

    # State güncelle
    _state["amounts_processed"] += amount
    _state["hourly_totals"][hour] = _state["hourly_totals"].get(hour, 0) + 1 if isinstance(_state["hourly_totals"], dict) else (_state["hourly_totals"][hour] + 1)

    if is_fraud:
        _state["fraud_amounts"] += amount
        if is_unknown:
            _state["fraud_unknown"] += 1
            _state["fraud_type_counts"]["UNKNOWN_NEW_FRAUD"] += 1
        else:
            _state["fraud_known"] += 1
            _state["fraud_type_counts"][fraud_type] = _state["fraud_type_counts"].get(fraud_type, 0) + 1
        _state["hourly_fraud"][hour] += 1
        _state["fraud_alerts"].appendleft(txn)
    else:
        _state["normal"] += 1

    _state["recent_transactions"].appendleft(txn)
    return txn


def SHAP_TEMPLATE_FOR(fraud_type):
    return REAL_SHAP_TEMPLATES.get(fraud_type, {})


def _make_message(is_fraud, is_unknown, fraud_type):
    if not is_fraud:
        return "İşlem normal görünüyor."
    if is_unknown:
        return "DİKKAT: Eğitimde hiç görülmemiş yeni bir fraud paterni (Zero-Shot) tespit edildi!"
    info = FRAUD_DESCRIPTIONS.get(fraud_type, {})
    return f"{info.get('title', fraud_type)} tespit edildi. {info.get('description', '')}"


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": round(time.time() - _state["start_time"], 1)}


@app.get("/api/stats")
async def get_stats():
    total = _state["total"] or 1
    fraud_total = _state["fraud_known"] + _state["fraud_unknown"]
    return {
        "total_transactions": _state["total"],
        "normal_count": _state["normal"],
        "fraud_known_count": _state["fraud_known"],
        "fraud_unknown_count": _state["fraud_unknown"],
        "fraud_total": fraud_total,
        "fraud_rate_pct": round(fraud_total / total * 100, 4),
        "amounts_processed": round(_state["amounts_processed"], 2),
        "fraud_amounts": round(_state["fraud_amounts"], 2),
        "fraud_type_counts": dict(_state["fraud_type_counts"]),
        "uptime_seconds": round(time.time() - _state["start_time"], 1),
        "metrics": REAL_METRICS,
    }


@app.get("/api/stream")
async def stream_transaction():
    """Her çağrıda 1 işlem üretir (frontend polling için)."""
    txn = _gen_transaction()
    return txn


@app.get("/api/alerts")
async def get_alerts(limit: int = 20):
    alerts = list(_state["fraud_alerts"])[:limit]
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/api/transactions")
async def get_transactions(limit: int = 50):
    txns = list(_state["recent_transactions"])[:limit]
    return {"transactions": txns, "count": len(txns)}


@app.get("/api/shap/{fraud_type}")
async def get_shap(fraud_type: str, noise: float = 0.1):
    """
    Belirtilen fraud tipi için SHAP değerleri döner.
    noise=0.1 → gerçekçi varyasyon ekler.
    """
    template = REAL_SHAP_TEMPLATES.get(fraud_type, REAL_SHAP_TEMPLATES["fraud_type_0"])
    shap_vals = {}
    for feat, val in template.items():
        noisy = val * (1 + random.uniform(-noise, noise))
        shap_vals[feat] = round(noisy, 4)
    return {
        "fraud_type": fraud_type,
        "shap_values": shap_vals,
        "description": FRAUD_DESCRIPTIONS.get(fraud_type, {}),
    }


@app.get("/api/shap/transaction/{txn_id}")
async def get_shap_for_transaction(txn_id: str):
    """Belirli bir işlem için SHAP + açıklama döner."""
    # Gerçek sistemde txn_id ile cache'ten çekilir; burada simüle ediyoruz
    fraud_types = list(REAL_SHAP_TEMPLATES.keys())
    fraud_type = random.choice(fraud_types)
    return await get_shap(fraud_type)


@app.post("/api/trigger_new_fraud")
async def trigger_new_fraud():
    """'Yeni Fraud Tipi Tespit Edildi' demo butonu için."""
    txn = _gen_transaction(force_fraud_type="UNKNOWN_NEW_FRAUD")
    return {
        "success": True,
        "transaction": txn,
        "alert": "⚠️ YENİ FRAUD TİPİ TESPİT EDİLDİ! Zero-Shot Learning aktive edildi.",
    }


@app.get("/api/hourly")
async def get_hourly():
    """Son 24 saatlik işlem/fraud dağılımı."""
    hours = []
    now_h = int((time.time() - _state["start_time"]) / 3600) % 24
    for i in range(24):
        h = (now_h - 23 + i) % 24
        hours.append({
            "hour": f"{h:02d}:00",
            "total": _state["hourly_totals"][h],
            "fraud": _state["hourly_fraud"][h],
        })
    return {"hourly": hours}


@app.get("/api/fraud_types")
async def get_fraud_types():
    return {"fraud_types": FRAUD_DESCRIPTIONS}


@app.get("/api/model_comparison")
async def get_model_comparison():
    """Centralized vs FL vs FZSL karşılaştırma tablosu."""
    return {
        "models": [
            {
                "name": "Centralized MLP",
                "precision": 0.9289,
                "recall": 0.9388,
                "f1": 0.9338,
                "roc_auc": 0.9991,
                "pr_auc": 0.7741,
                "unseen_detection": 0.0,
                "privacy": False,
                "color": "#64748b",
            },
            {
                "name": "Federated Learning (FL)",
                "precision": 0.9373,
                "recall": 1.0000,
                "f1": 0.9676,
                "roc_auc": 1.0000,
                "pr_auc": 0.9942,
                "unseen_detection": 0.0,
                "privacy": True,
                "color": "#3B82F6",
            },
            {
                "name": "FL + FZSL (Önerilen)",
                "precision": 0.9579,
                "recall": 0.9715,
                "f1": 0.9647,
                "roc_auc": 1.0000,
                "pr_auc": 0.9934,
                "unseen_detection": 0.9831,
                "privacy": True,
                "color": "#8B5CF6",
            },
        ]
    }


@app.post("/api/reset")
async def reset_stats():
    """İstatistikleri sıfırla."""
    _state["total"] = 0
    _state["normal"] = 0
    _state["fraud_known"] = 0
    _state["fraud_unknown"] = 0
    _state["start_time"] = time.time()
    _state["recent_transactions"].clear()
    _state["fraud_alerts"].clear()
    _state["fraud_type_counts"] = {
        "fraud_type_0": 0, "fraud_type_1": 0,
        "fraud_type_2": 0, "UNKNOWN_NEW_FRAUD": 0
    }
    _state["hourly_totals"] = [0] * 24
    _state["hourly_fraud"] = [0] * 24
    _state["amounts_processed"] = 0.0
    _state["fraud_amounts"] = 0.0
    return {"success": True, "message": "İstatistikler sıfırlandı."}
