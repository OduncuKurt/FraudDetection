# FraudShield — Federated Zero-Shot Learning for Credit Card Fraud Detection

> **Erasmus Internship Portfolio Project** — Extended from undergraduate thesis (2025–2026)

A production-grade fraud detection system combining **Federated Learning (FL)**, **Zero-Shot Learning (ZSL)**, and **Explainable AI (XAI)** with a real-time monitoring dashboard featuring Differential Privacy, Concept Drift detection, and multi-client FL visualization.

---

## 🔥 Live Dashboard

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (first time)
python demo.py --mode fzsl

# 3. Start the dashboard
python -m uvicorn backend.dashboard_api:app --reload --port 8000

# 4. Open in browser
# → http://localhost:8000
```

---

## 🏗️ System Architecture

```
Raw Transaction Data (creditcard.csv — 284K transactions)
        │
        ▼
  [Data Preprocessing]
  data_prep.py — IID / Non-IID client split
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
[Centralized MLP]               [Federated Learning (FedAvg)]
 train_centralized.py             train_fl.py — 4 clients
        │                                      │
        └──────────────────────────────────────┘
                          │
                          ▼
             [FZSL — Zero-Shot Learning]
              src/fzsl/
              - KMeans clustering → 4 fraud subtypes
              - fraud_type_3 held out as UNSEEN class
              - InfoNCE loss (CLIP-style contrastive)
              - Detects unseen fraud via text prototypes
                          │
                          ▼
                [XAI — Explainability]
                SHAP GradientExplainer + Counterfactual
                          │
                          ▼
             [Real-Time Dashboard API]
             backend/dashboard_api.py
             - Concept Drift (KL Divergence)
             - Differential Privacy (ε budget)
             - FL Client Visualization
```

---

## 📊 Model Performance

> All models trained on the same ULB creditcard.csv dataset.
> Centralized and FL: threshold=0.5 (default). FZSL: optimal F1 threshold=0.7902.

| Metric | Centralized MLP | Federated (FedAvg, 5 rounds) | **FL + FZSL (Ours)** |
|--------|:--------------:|:-----------------------------:|:--------------------:|
| ROC-AUC | 0.9743 | 0.9813 | **1.0000** |
| PR-AUC | 0.6586 | 0.7020 | **0.9934** |
| F1 Score | 0.0868 | 0.1721 | **0.9647** |
| Precision | 0.0456 | 0.0953 | **0.9579** |
| Recall | 0.9082 | 0.8878 | **0.9715** |
| Data Privacy | ❌ Raw data shared | ✅ Weights only | ✅ Weights only |
| Zero-Shot Detection | ❌ Not supported | ❌ Not supported | **✅ 98.3%** |
| New Fraud Types | ❌ Cannot detect | ❌ Cannot detect | **✅ Via text description** |

### Why Federated Learning?
In centralized training, raw transaction data is sent to a single server — a **GDPR violation risk**. In Federated Learning, each bank trains locally and only shares model weights. Raw data **never leaves the bank**.

### Why Zero-Shot Learning?
Real-world fraud patterns evolve daily. Traditional models cannot detect fraud types they have never seen during training. FZSL detects `fraud_type_3` (Transaction Laundering) — a type **completely absent from training** — with **98.3% success** using only a natural language description.

---

## 🧠 How FZSL Works

1. Fraud samples are clustered into **4 subtypes** using KMeans on V1–V28 PCA latent space
2. `fraud_type_3` is **completely excluded from training** (the unseen/zero-shot class)
3. A natural language description is written for each class and embedded via **TF-IDF + RandomProjection**
4. `TransactionEncoder` + `TextProjection` are trained with **InfoNCE (contrastive) loss** — CLIP-style
5. At inference, `fraud_type_3`'s text description is embedded and matched against transaction embeddings
6. The model detects this unseen type with **98.3% accuracy** — without ever seeing a single example

---

## 🔐 Fraud Subtype Taxonomy

Subtypes are discovered via **unsupervised clustering** (not pre-labeled), then mapped to known attack vectors from the literature.

| Cluster | Literature Match | Key Statistics | Reference |
|---------|-----------------|----------------|-----------|
| `fraud_type_0` | CNP / Card Cloning | n=207, mean=$172.80, dominant: V14, V4 | Bhattacharyya et al. (2011) |
| `fraud_type_1` | Account Takeover (ATO) | n=92, mean=$96.03, early time window | FATF (2022) |
| `fraud_type_2` | Card Probing / Carding | n=15, mean=$2.22, time_std≈53 min | Aleskerov et al. (1997) |
| `fraud_type_3` | Transaction Laundering | n=178, mean=$87.03, time_std≈13.8h | FATF (2022); FinCEN (2014) |

> **Note:** `fraud_type_3` is the **zero-shot unseen class** — held out from all training. The model identifies it solely via its natural language description.

---

## 🌊 Concept Drift Detection

The dashboard monitors for **distribution shift** in real-time using sliding-window KL divergence:

- **Baseline**: Feature distribution from the first 200 transactions
- **Window**: Last 150 transactions
- **Method**: Approximate KL divergence (Gaussian assumption) per V-feature
- **Threshold**: Alert when overall drift score > 40%
- **API**: `GET /api/drift`

---

## 🔐 Differential Privacy — GDPR Layer

Real Gaussian mechanism implementation for gradient protection:

```
ε (epsilon) = √(2 · ln(1.25/δ)) · sensitivity / σ
```

- **σ (noise scale)**: Gaussian noise added to gradients — configurable via slider
- **ε budget**: Cumulative privacy cost tracked per FL round
- **δ**: Fixed at 1e-5 (probability of privacy breach)
- **Interactive**: Noise slider shows real-time accuracy/privacy tradeoff
- **API**: `GET /api/privacy/budget`

---

## 🌐 Dashboard Features

| Panel | Description |
|-------|-------------|
| **Live Transaction Feed** | Real-time stream at 1 tx/sec from the credit card dataset |
| **Fraud Alert Panel** | Clickable fraud alerts with SHAP explanation modal |
| **Transaction Flow Chart** | Real-time line chart (Normal vs Fraud) |
| **Fraud Type Distribution** | Donut chart showing fraud type breakdown |
| **XAI — SHAP Explanation** | GradientExplainer SHAP per fraud type + per transaction |
| **Concept Drift Monitor** | KL divergence gauge + feature heatmap |
| **Differential Privacy** | ε budget gauge + accuracy/privacy tradeoff + noise slider |
| **FL Network** | SVG topology of 4 bank clients + animated aggregation rounds |
| **Model Comparison** | Side-by-side Centralized vs FL vs FL+FZSL metrics |
| **Performance Metrics** | Bar chart of Precision/Recall/F1/AUC across all models |

---

## 📡 API Reference

The backend exposes a **FastAPI** REST API at `http://localhost:8000`.  
Interactive Swagger docs: **http://localhost:8000/docs**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stream` | GET | Next transaction (real model inference) |
| `/api/stats` | GET | Global statistics and model metrics |
| `/api/shap/{fraud_type}` | GET | SHAP values for a fraud type |
| `/api/drift` | GET | Concept drift scores (KL divergence) |
| `/api/privacy/budget` | GET | ε budget usage and DP parameters |
| `/api/privacy/noise_impact` | GET | Accuracy impact for given σ |
| `/api/fl/clients` | GET | Per-client FL stats (accuracy, samples) |
| `/api/fl/simulate_round` | POST | Trigger FL aggregation round + consume ε |
| `/api/trigger_new_fraud` | POST | Simulate zero-shot fraud type detection |
| `/api/explain/{txn_id}` | GET | Full XAI report (SHAP + counterfactual) |
| `/api/model_comparison` | GET | Centralized vs FL vs FZSL metrics |
| `/api/reset` | POST | Reset statistics |

---

## 🗂️ Project Structure

```
FraudShield/
├── data/                   ← creditcard.csv (not in repo — see below)
├── checkpoints/            ← Trained models (gitignored)
├── outputs/                ← SHAP plots, PR curves
├── results/                ← Training logs
├── src/
│   ├── model.py            ← MLP architecture (FL backbone)
│   ├── data_prep.py        ← Preprocessing + IID/Non-IID split
│   ├── fl_utils.py         ← FedAvg, local training, evaluation
│   ├── train_centralized.py
│   ├── train_fl.py
│   ├── inference.py        ← FraudAnalyzer (real-time inference)
│   ├── system.py           ← Unified FL + FZSL + SHAP pipeline
│   ├── shap_analysis.py    ← SHAP DeepExplainer
│   ├── xai_analysis.py     ← Permutation Importance
│   └── fzsl/
│       ├── class_descriptions.py  ← Fraud subtype text descriptions
│       ├── fraud_subtypes.py      ← KMeans clustering
│       ├── zsl_encoder.py         ← TF-IDF + RandomProjection encoder
│       ├── fzsl_model.py          ← TransactionEncoder + TextProjection
│       └── train_fzsl.py          ← Full FZSL training pipeline
├── backend/
│   ├── dashboard_api.py    ← Main FastAPI app (all endpoints)
│   ├── xai_engine.py       ← GradientExplainer + Counterfactual
│   ├── shap_narrator.py    ← English SHAP explanation generator
│   ├── explainer.py        ← XAI utilities
│   └── schemas.py          ← Pydantic request/response schemas
├── dashboard/
│   ├── index.html          ← Dashboard UI (English)
│   ├── app.js              ← Chart.js + polling + FL animation
│   └── style.css           ← Dark mode design system
├── demo.py                 ← Quick-start training script
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- 4 GB RAM minimum (model loading + SHAP)

### Install

```bash
pip install -r requirements.txt
```

### Dataset

`data/creditcard.csv` is not included due to file size. Download from Kaggle:

> 🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it at `data/creditcard.csv`.

### Train

```bash
# Full pipeline (recommended)
python demo.py --mode fzsl

# Individual components
python demo.py --mode centralized
python demo.py --mode fl
python demo.py --mode all
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | PyTorch (FL training, FZSL contrastive learning) |
| **Explainability** | SHAP (GradientExplainer), Counterfactual |
| **Privacy** | Gaussian Differential Privacy (Gaussian mechanism) |
| **Backend API** | FastAPI + Pydantic |
| **Frontend** | Vanilla JS + Chart.js (no framework overhead) |
| **Dataset** | IEEE/Kaggle Credit Card Fraud (284K transactions) |
| **FL Algorithm** | FedAvg (McMahan et al., 2017) |
| **ZSL Method** | InfoNCE contrastive loss (CLIP-style) |

---

## 📚 References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2017). *Credit card fraud detection: a realistic modeling and a novel learning strategy.* IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784–3797.

2. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). *Communication-efficient learning of deep networks from decentralized data.* AISTATS.

3. Radford, A., et al. (2021). *Learning transferable visual models from natural language supervision.* ICML. (CLIP — contrastive learning architecture)

4. Dwork, C., & Roth, A. (2014). *The algorithmic foundations of differential privacy.* Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407.

5. Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011). Data mining for credit card fraud: A comparative study. *Decision Support Systems*, 50(3), 602–613.

6. Aleskerov, E., Freisleben, B., & Rao, B. (1997). CARDWATCH: A neural network based database mining system for credit card fraud detection. *IEEE/IAFE Conference on Computational Intelligence for Financial Engineering*, 220–226.

7. FATF (2022). *Money Laundering from Financial Fraud.* Financial Action Task Force Report.

8. FinCEN (2014). *Advisory on Recognizing Activity that May be Associated with Internet-Based Payment Systems.* FIN-2014-A005.

---

## 📄 License

MIT License — free to use for academic and research purposes.
