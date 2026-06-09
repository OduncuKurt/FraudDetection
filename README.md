# FraudDetection — Federated Zero-Shot Learning ile Kredi Kartı Sahtekarlığı Tespiti

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

## FZSL Nasıl Çalışır?

1. Fraud örnekleri **KMeans** ile 4 alt tipe ayrılır (`fraud_type_0..3`)
2. `fraud_type_3` **eğitimde hiç kullanılmaz** (unseen/görülmemiş sınıf)
3. Her sınıf için doğal dil açıklaması yazılır ve **TF-IDF + RandomProjection** ile embed edilir
4. `TransactionEncoder` + `TextProjection` **InfoNCE loss** ile eğitilir (CLIP mimarisi)
5. Test anında `fraud_type_3` açıklaması embed edilir ve model bu tipi
   **hiç görmeden** %98 doğrulukla tespit eder

---

## Veri Seti ve Fraud Tipi Metodolojisi

**Veri seti:** ULB Credit Card Fraud Detection (Kaggle, MLG-ULB grubu)  
284.807 gerçek kredi kartı işlemi; 492 fraud (%0,17), 284.315 normal.

> Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2017).
> *Credit card fraud detection: a realistic modeling and a novel learning strategy.*
> IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784–3797.

### Fraud Alt Tiplerinin Akademik Dayanağı

Bu projede fraud alt tipleri **önceden etiketlenmemiş**; bunun yerine veri güdümlü bir
yaklaşımla keşfedilmiştir. V1–V28 PCA latent uzayında KMeans (k=4) kümelemesi uygulanmış,
orte her küme işlem tutarı, zaman dağılımı ve baskın latent özellik yönleri açısından
analiz edilmiştir. Elde edilen istatistiksel örüntüler, kredi kartı sahtekarlığı
literatüründeki bilinen saldırı vektörleriyle ilişkilendirilmiştir.

| Küme Etiketi  | Literatür Karşılığı           | Temel İstatistik                        | Referans                        |
|---------------|-------------------------------|------------------------------------------|---------------------------------|
| `fraud_type_0`| CNP / Card Cloning            | n=207, mean=$172.80, baskın: V14, V4    | Bhattacharyya et al. (2011)     |
| `fraud_type_1`| Account Takeover (ATO)        | n=92, mean=$96.03, erken zaman dilimi   | FATF (2022)                     |
| `fraud_type_2`| Card Probing / Carding        | n=15, mean=$2.22, time_std≈53 dk        | Aleskerov et al. (1997)         |
| `fraud_type_3`| Transaction Laundering        | n=178, mean=$87.03, time_std≈13.8 saat  | FATF (2022); FinCEN (2014)      |

> **Not:** `fraud_type_3` FZSL değerlendirmesinde **unseen (görülmemiş)** sınıf olarak
> kullanılmaktadır. Model bu tipi eğitimde hiç görmeden yalnızca doğal dil açıklaması
> üzerinden tespit etmekte ve %98.3 başarı oranına ulaşmaktadır.

---

## Model Karşılaştırması

> Tüm modeller aynı ULB creditcard.csv verisi üzerinde eğitilmiştir.  
> Centralized ve FL modelleri: threshold=0.5 (default), FZSL: optimal F1 threshold=0.7902

| Metrik | Centralized MLP | Federated (FedAvg, 5 tur, Non-IID) | **FZSL (Önerimiz)** |
|--------|:--------------:|:----------------------------------:|:-------------------:|
| ROC-AUC | 0.9743 | 0.9813 | **1.0000** |
| PR-AUC | 0.6586 | 0.7020 | **0.9934** |
| F1 | 0.0868 | 0.1721 | **0.9647** |
| Precision | 0.0456 | 0.0953 | **0.9579** |
| Recall | 0.9082 | 0.8878 | **0.9715** |
| Veri Gizliliği | ❌ Ham veri paylaşımı | ✅ Yalnızca ağırlıklar | ✅ Yalnızca ağırlıklar |
| Zero-Shot Tespit | ❌ Desteklemiyor | ❌ Desteklemiyor | **✅ %98.3** |
| Yeni Fraud Tipi | ❌ Tanıyamaz | ❌ Tanıyamaz | **✅ Doğal dil açıklamasıyla** |

> **Neden FL?** Merkezi modelde ham işlem verisi tek sunucuya gönderilir (GDPR ihlali riski).  
> Federated Learning'de her banka yalnızca model ağırlıklarını paylaşır, veri yerel kalır.  
> 
> **Neden FZSL?** Gerçek dünyada her gün yeni fraud tipleri ortaya çıkar. Klasik modeller  
> eğitimde görmedikleri tipleri tanıyamaz. FZSL, sadece doğal dil açıklaması ile görülmemiş  
> `fraud_type_3` (Transaction Laundering) tipini **%98.3** başarıyla tespit etmiştir.

### FZSL Detaylı Metrikler

| Metrik | Değer |
|--------|-------|
| ROC-AUC | 1.0000 |
| PR-AUC | 0.9934 |
| F1 (optimal threshold=0.7902) | 0.9647 |
| Precision | 0.9579 |
| Recall | 0.9715 |
| **Görülmemiş fraud tespiti (zero-shot)** | **%98.3** |

---


## Bağımlılıklar

```
pandas, numpy, scikit-learn, torch,
shap, matplotlib, sentence-transformers (opsiyonel)
```

---

## Referanslar

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2017). Credit card fraud detection: a realistic modeling and a novel learning strategy. *IEEE Transactions on Neural Networks and Learning Systems*, 29(8), 3784–3797.
2. Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011). Data mining for credit card fraud: A comparative study. *Decision Support Systems*, 50(3), 602–613.
3. Aleskerov, E., Freisleben, B., & Rao, B. (1997). CARDWATCH: A neural network based database mining system for credit card fraud detection. *IEEE/IAFE Conference on Computational Intelligence for Financial Engineering*, 220–226.
4. FATF (2022). *Money Laundering from Financial Fraud*. Financial Action Task Force Report. https://www.fatf-gafi.org
5. FinCEN (2014). *Advisory on Recognizing Activity that May be Associated with Internet-Based Payment Systems*. FIN-2014-A005. Financial Crimes Enforcement Network.
