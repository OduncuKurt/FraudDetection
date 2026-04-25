# FraudDetection — Federated Zero-Shot Learning ile Kredi Karti Sahtekarligi Tespiti

Bu proje, kredi karti islemlerinde sahtekarligi tespit etmek icin
**Federated Learning (FL)**, **Zero-Shot Learning (ZSL)** ve **Explainable AI (XAI)**
yontemlerini bir arada kullanan bir sistem gelistirmektedir.

---

## Mimari

```
Ham Islem Verisi (creditcard.csv)
        |
        v
  [Veri On-Isleme]
  data_prep.py — IID / Non-IID client split
        |
        +──────────────────────────────────────────+
        |                                          |
        v                                          v
[Centralized MLP]                    [Federated Learning (FedAvg)]
 train_centralized.py                  train_fl.py
        |                                          |
        +──────────────────────────────────────────+
                          |
                          v
             [FZSL — Zero-Shot Learning]
              src/fzsl/train_fzsl.py
              - KMeans ile fraud alt tipleri
              - Gorulmemis fraud tipini aciklama
                metni uzerinden tespit eder
                          |
                          v
                [XAI — Aciklanabilirlik]
                SHAP + Permutation Importance
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

> **Not:** `data/creditcard.csv` dosyasi buyuk oldugu icin repoda yer almamaktadir.
> Asagidaki adresten indirip `data/` klasorune koyun:
> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Calistirma

### Hizli Baslangi (demo.py)

```bash
# Sadece FZSL (ana sistem — onerilen)
python demo.py --mode fzsl

# Merkezi model
python demo.py --mode centralized

# Federated Learning
python demo.py --mode fl

# Hepsini sirali calistir
python demo.py --mode all
```

### Direkt modül olarak

```bash
python -m src.train_centralized
python -m src.train_fl
python -m src.fzsl.train_fzsl
```

---

## Klasor Yapisi

```
FraudDetection/
├── data/               ← creditcard.csv buraya (gitignore'da)
├── checkpoints/        ← Egitilen modeller (gitignore'da)
├── outputs/            ← Grafik ciktilari (SHAP, PR curve vb.)
├── results/            ← Egitim log dosyalari (gitignore'da)
├── src/
│   ├── model.py            ← MLP mimarisi
│   ├── data_prep.py        ← Veri hazirlama + IID/Non-IID split
│   ├── fl_utils.py         ← FedAvg, local training, evaluation
│   ├── train_centralized.py
│   ├── train_fl.py
│   ├── shap_analysis.py    ← SHAP DeepExplainer
│   ├── xai_analysis.py     ← Permutation Importance
│   ├── logger.py
│   └── fzsl/
│       ├── class_descriptions.py  ← Fraud alt tipi aciklamalari
│       ├── fraud_subtypes.py      ← KMeans clustering
│       ├── zsl_encoder.py         ← TF-IDF + RandomProjection
│       ├── fzsl_model.py          ← TransactionEncoder + TextProjection
│       └── train_fzsl.py          ← Tam FZSL pipeline
├── demo.py
└── requirements.txt
```

---

## FZSL Nasil Calisir?

1. Fraud ornekleri **KMeans** ile 4 alt tipe ayrilir (`fraud_type_0..3`)
2. `fraud_type_3` **egitimde hic kullanilmaz** (unseen/gorulmemis sinif)
3. Her sinif icin dogal dil aciklamasi yazilir ve **TF-IDF + RandomProjection** ile embed edilir
4. `TransactionEncoder` + `TextProjection` **InfoNCE loss** ile egitilir (CLIP mimarisi)
5. Test aninda `fraud_type_3` aciklamasi embed edilir ve model bu tipi
   **hic gormeden** %98 dogrulukla tespit eder

---

## Temel Sonuclar (FZSL)

| Metrik | Deger |
|--------|-------|
| ROC-AUC | 1.0000 |
| PR-AUC | 0.9934 |
| F1 (optimal threshold) | 0.9647 |
| Precision | 0.9579 |
| Recall | 0.9715 |
| **Gorulmemis fraud tespit (zero-shot)** | **%98.3** |

---

## Bagimliliklar

```
pandas, numpy, scikit-learn, torch,
shap, matplotlib, sentence-transformers (opsiyonel)
```
