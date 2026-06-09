"""
train_fzsl.py
-------------
Federated Zero-Shot Learning (FZSL) eğitim ve değerlendirme pipeline'ı.

Pipeline:
─────────────────────────────────────────────────────────────────────────────
1.  creditcard.csv yükle
2.  Fraud'ları KMeans ile 4 cluster'a ayır (fraud_subtypes)
3.  fraud_type_3 → UNSEEN (eğitimde kullanılmaz, zero-shot test için saklanır)
4.  Sınıf açıklamalarını SBERT ile embed et
5.  FZSL modelini InfoNCE loss ile eğit (sadece SEEN sınıflar üzerinde)
6.  Zero-shot değerlendirme:
    - Seen sınıflar üzerinde performans
    - Unseen (fraud_type_3) üzerinde zero-shot performans
7.  Sonuçları karşılaştır ve kaydet
─────────────────────────────────────────────────────────────────────────────
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from src.fzsl.fraud_subtypes import (
    load_creditcard_dataset,
    generate_fraud_subtypes,
    build_seen_unseen_datasets,
    summarize_fraud_subtypes,
)
from src.fzsl.class_descriptions import (
    FRAUD_CLASS_DESCRIPTIONS,
    SEEN_CLASSES,
    UNSEEN_CLASS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from src.fzsl.zsl_encoder import TextClassEncoder
from src.fzsl.fzsl_model import FZSLModel, FZSLPredictor
from src.logger import setup_logger


# ─────────────────────────────────────────────────────────────────────────────
#  Veri hazırlama
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DROP_COLS = ["Class", "fraud_cluster", "fraud_subtype", "fzsl_split"]


def get_feature_cols(df):
    return [c for c in df.columns if c not in FEATURE_DROP_COLS]


def prepare_seen_training_data(train_df, scaler=None):
    """
    Sadece SEEN sınıfları içeren eğitim verisi hazırlar.
    SEEN_CLASSES = normal, fraud_type_0, fraud_type_1, fraud_type_2
    """
    mask = train_df["fraud_subtype"].isin(SEEN_CLASSES)
    df_seen = train_df[mask].copy()

    feature_cols = get_feature_cols(df_seen)
    X = df_seen[feature_cols].values.astype(np.float32)
    y = df_seen["fraud_subtype"].map(CLASS_TO_IDX).values.astype(np.int64)

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler, feature_cols


def prepare_test_data(test_df, scaler, feature_cols):
    """
    Test veri seti (hem seen hem unseen sınıfları içerir).
    Dönüş: X, fraud_subtype listesi, binary label (0=normal, 1=fraud)
    """
    X = test_df[feature_cols].values.astype(np.float32)
    X = scaler.transform(X)
    subtypes = test_df["fraud_subtype"].values
    y_binary = test_df["Class"].values  # 0=normal, 1=fraud
    return X, subtypes, y_binary


# ─────────────────────────────────────────────────────────────────────────────
#  InfoNCE / NT-Xent Loss (CLIP-style)
# ─────────────────────────────────────────────────────────────────────────────

def infonce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits : [batch_size, num_seen_classes]   (scaled cosine similarities)
    labels : [batch_size]  long  (class indices 0..num_seen_classes-1)
    """
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
#  Eğitim döngüsü
# ─────────────────────────────────────────────────────────────────────────────

def train_fzsl(
    model: FZSLModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seen_text_embeddings: np.ndarray,   # [num_seen_classes, text_dim]
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
) -> FZSLModel:
    """
    FZSL modelini InfoNCE loss ile eğitir.

    seen_text_embeddings : SBERT çıktısı, her SEEN sınıf için bir satır
                           Sıra CLASS_TO_IDX'e göre olmalı
    """
    model.to(device)

    text_tensor = torch.tensor(seen_text_embeddings, dtype=torch.float32).to(device)
    # shape: [num_seen_classes, text_dim]

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    # --- WeightedRandomSampler: fraud sinifini oversample et (900:1 dengesizlik) ---
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts.astype(float) + 1e-8)
    sample_weights = torch.tensor(
        [class_weights[label] for label in y_train], dtype=torch.float32
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    print(f"\n[FZSL Train] {len(X_train)} ornek | {epochs} epoch | batch={batch_size}")
    print(f"[FZSL Train] Sinif dagilimi: { {IDX_TO_CLASS[i]: int(c) for i, c in enumerate(class_counts)} }")
    print(f"[FZSL Train] WeightedSampler aktif: fraud ornekleri oversample ediliyor")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X, text_tensor)   # [B, num_seen_classes]
            loss = infonce_loss(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping → stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.5f}  "
                  f"temp={model.temperature.exp().item():.4f}")

    # En iyi ağırlıkları yükle
    model.load_state_dict(best_state)
    print(f"\n[FZSL Train] En iyi loss: {best_loss:.5f}")
    return model, loss_history


# ─────────────────────────────────────────────────────────────────────────────
#  Değerlendirme
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, fraud_proba: np.ndarray,
                            min_recall: float = 0.80):
    """
    PR curve uzerinde iki optimal nokta bulur:
    1. En yuksek F1 skorunu veren threshold
    2. recall >= min_recall iken en yuksek Precision veren threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, fraud_proba)

    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-9
    )
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]

    valid_mask = recalls[:-1] >= min_recall
    if valid_mask.any():
        best_prec_idx = np.where(valid_mask)[0][np.argmax(precisions[:-1][valid_mask])]
        best_prec_threshold = thresholds[best_prec_idx]
    else:
        best_prec_threshold = best_f1_threshold
        best_prec_idx = best_f1_idx

    print(f"\n[Threshold Opt] Best F1 @ threshold={best_f1_threshold:.4f}")
    print(f"  Precision={precisions[best_f1_idx]:.4f}  "
          f"Recall={recalls[best_f1_idx]:.4f}  "
          f"F1={f1_scores[best_f1_idx]:.4f}")
    print(f"[Threshold Opt] Best Precision (recall>={min_recall}) @ threshold={best_prec_threshold:.4f}")
    print(f"  Precision={precisions[best_prec_idx]:.4f}  "
          f"Recall={recalls[best_prec_idx]:.4f}  "
          f"F1={f1_scores[best_prec_idx]:.4f}")

    return {
        "f1_optimal": {
            "threshold": float(best_f1_threshold),
            "precision": float(precisions[best_f1_idx]),
            "recall":    float(recalls[best_f1_idx]),
            "f1":        float(f1_scores[best_f1_idx]),
        },
        "precision_optimal": {
            "threshold": float(best_prec_threshold),
            "precision": float(precisions[best_prec_idx]),
            "recall":    float(recalls[best_prec_idx]),
            "f1":        float(f1_scores[best_prec_idx]),
        },
    }


def plot_pr_curve(y_true: np.ndarray, fraud_proba: np.ndarray,
                  opt_thresholds: dict, save_path: str):
    """Precision-Recall curve + optimal threshold noktalarini kaydeder."""
    precisions, recalls, _ = precision_recall_curve(y_true, fraud_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, color="#6366f1", linewidth=2, label="FZSL PR Curve")
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline (fraud rate={baseline:.4f})")

    f1_opt = opt_thresholds["f1_optimal"]
    ax.scatter(f1_opt["recall"], f1_opt["precision"], color="#ef4444", s=120, zorder=5,
               label=f"F1-optimal (thr={f1_opt['threshold']:.3f}, F1={f1_opt['f1']:.3f})")

    p_opt = opt_thresholds["precision_optimal"]
    ax.scatter(p_opt["recall"], p_opt["precision"], color="#22c55e", s=120, zorder=5,
               label=f"Prec-optimal (thr={p_opt['threshold']:.3f}, P={p_opt['precision']:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("FZSL Precision-Recall Curve", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] PR curve kaydedildi: {save_path}")


def evaluate_zsl_binary(predictor: FZSLPredictor, X: np.ndarray, y_binary: np.ndarray,
                         subtypes: np.ndarray, label: str = "", threshold: float = 0.5):
    """Binary fraud detection degerlendirmesi."""
    fraud_proba = predictor.is_fraud_proba(X)
    preds_binary = (fraud_proba >= threshold).astype(int)

    acc   = accuracy_score(y_binary, preds_binary)
    prec  = precision_score(y_binary, preds_binary, zero_division=0)
    rec   = recall_score(y_binary, preds_binary, zero_division=0)
    f1    = f1_score(y_binary, preds_binary, zero_division=0)

    # ROC-AUC sadece iki sınıf varsa
    roc = pr = 0.0
    if len(np.unique(y_binary)) > 1:
        roc = roc_auc_score(y_binary, fraud_proba)
        pr  = average_precision_score(y_binary, fraud_proba)

    cm = confusion_matrix(y_binary, preds_binary)

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  Threshold = {threshold:.4f}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  PR-AUC    : {pr:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {cm}")

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": roc, "pr_auc": pr, "confusion_matrix": cm,
            "threshold": threshold}


def evaluate_zsl_unseen(predictor: FZSLPredictor, X_unseen: np.ndarray,
                         all_classes: list):
    """
    Unseen sınıf (fraud_type_3) için zero-shot sınıflandırma değerlendirmesi.

    Başarı kriteri: X_unseen örnekleri 'normal' yerine bir fraud sınıfına atanmalı.
    """
    pred_classes, sims = predictor.predict(X_unseen)
    pred_arr = np.array(pred_classes)

    # Kaç tanesi herhangi bir fraud sınıfına atandı?
    fraud_preds = np.array([c != "normal" for c in pred_classes])
    fraud_detect_rate = fraud_preds.mean()

    # En çok hangi sınıfa atandı?
    unique, counts = np.unique(pred_arr, return_counts=True)
    distribution = dict(zip(unique, counts))

    # Ortalama similarity skorları
    sims_mean = sims.mean(axis=0)
    sim_per_class = {cls: sims_mean[i] for i, cls in enumerate(all_classes)}

    print(f"\n{'─'*50}")
    print(f"  UNSEEN Sınıf Zero-Shot Değerlendirmesi (fraud_type_3)")
    print(f"{'─'*50}")
    print(f"  Örnek sayısı         : {len(X_unseen)}")
    print(f"  Fraud tespit oranı   : {fraud_detect_rate:.4f}")
    print(f"  Tahmin dağılımı      : {distribution}")
    print(f"  Ortalama Similarity Skorları:")
    for cls, sim in sorted(sim_per_class.items(), key=lambda x: -x[1]):
        marker = " ← (unseen target)" if cls == UNSEEN_CLASS else ""
        print(f"    {cls:<18}: {sim:.4f}{marker}")

    return {
        "fraud_detect_rate": fraud_detect_rate,
        "distribution": distribution,
        "mean_similarities": sim_per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Görselleştirme
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(loss_history: list, save_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="#6366f1", linewidth=2)
    plt.title("FZSL Eğitim Loss Eğrisi (InfoNCE)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Loss eğrisi kaydedildi: {save_path}")


def plot_similarity_heatmap(predictor: FZSLPredictor, samples_by_class: dict,
                             save_path: str):
    """
    Her sınıfın örnek ortalama similarity skorlarını ısı haritasında gösterir.
    samples_by_class: {class_name: X_np}
    """
    all_classes = predictor.class_order
    class_labels = list(samples_by_class.keys())
    sim_matrix = np.zeros((len(class_labels), len(all_classes)))

    for i, (cls_name, X) in enumerate(samples_by_class.items()):
        if len(X) == 0:
            continue
        _, sims = predictor.predict(X)
        sim_matrix[i] = sims.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.8)
    ax.set_xticks(range(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_title("FZSL: Ortalama Cosine Similarity (Gerçek Sınıf × Tahmin Sınıfı)",
                 fontsize=11)
    for i in range(len(class_labels)):
        for j in range(len(all_classes)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color="black")

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Similarity ısı haritası kaydedildi: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Ana pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logger(__file__)
    csv_path = "data/creditcard.csv"
    device = "cpu"
    proj_dim = 128
    epochs = 30
    lr = 1e-3
    batch_size = 256
    os.makedirs("outputs", exist_ok=True)

    # ── 1. Veri yükle ──────────────────────────────────────────────────────
    print("=" * 60)
    print("1. Dataset yükleniyor...")
    print("=" * 60)
    df = load_creditcard_dataset(csv_path)
    print(f"   Toplam kayıt: {len(df)}  |  Fraud: {df['Class'].sum()}  |  "
          f"Normal: {(df['Class'] == 0).sum()}")

    # ── 2. Fraud subtype'ları üret ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Fraud subtype'ları KMeans ile üretiliyor (4 cluster)...")
    print("=" * 60)
    df_labeled, kmeans, cluster_scaler = generate_fraud_subtypes(df, n_clusters=4)

    summary = summarize_fraud_subtypes(df_labeled)
    print("\nFraud Subtype Özeti:")
    print(summary.to_string(index=False))

    # ── 3. Seen/Unseen ayrımı ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"3. Seen/Unseen Ayrimi  ->  Unseen: {UNSEEN_CLASS}")
    print("=" * 60)
    train_df, test_df, df_marked = build_seen_unseen_datasets(
        df_labeled, unseen_subtype=UNSEEN_CLASS
    )
    print(f"   Train (seen): {len(train_df)} sample")
    print(f"   Test  (all) : {len(test_df)} sample")
    print(f"\n   FZSL Split Distribution:\n{df_marked['fzsl_split'].value_counts().to_string()}")

    # ── 4. Veri ön-işleme ──────────────────────────────────────────────────
    X_train, y_train, scaler, feature_cols = prepare_seen_training_data(train_df)
    X_test, subtypes_test, y_binary_test = prepare_test_data(test_df, scaler, feature_cols)

    input_dim = X_train.shape[1]
    print(f"\n   Input dim: {input_dim}  |  Train size: {len(X_train)}")
    print(f"   Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"     {IDX_TO_CLASS[idx]:<20}: {cnt}")

    # ── 5. Sınıf açıklamalarını SBERT ile embed et ─────────────────────────
    print("\n" + "=" * 60)
    print("4. Class descriptions embedded with SBERT...")
    print("=" * 60)
    text_encoder = TextClassEncoder(embed_dim=128, backend="local")

    # Tüm sınıflar (seen + unseen) için embedding → inference'da kullanılacak
    all_class_order = SEEN_CLASSES + [UNSEEN_CLASS]
    all_text_emb_matrix, all_class_order = text_encoder.get_class_embedding_matrix(
        FRAUD_CLASS_DESCRIPTIONS, class_order=all_class_order
    )
    text_dim = all_text_emb_matrix.shape[1]

    # Sadece seen sınıflar → training loss için
    seen_text_emb_matrix, _ = text_encoder.get_class_embedding_matrix(
        FRAUD_CLASS_DESCRIPTIONS, class_order=SEEN_CLASSES
    )

    print(f"   Toplam sınıf embedding boyutu: {all_text_emb_matrix.shape}")

    # ── 6. FZSL modelini eğit ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. FZSL model training (InfoNCE loss)...")
    print("=" * 60)
    fzsl_model = FZSLModel(
        input_dim=input_dim,
        text_dim=text_dim,
        proj_dim=proj_dim,
        temperature=0.07,
    )

    fzsl_model, loss_history = train_fzsl(
        model=fzsl_model,
        X_train=X_train,
        y_train=y_train,
        seen_text_embeddings=seen_text_emb_matrix,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    plot_loss_curve(loss_history, "outputs/fzsl_loss_curve.png")

    # ── 7. Predictor oluştur (seen + unseen class embedding'leriyle) ────────
    print("\n" + "=" * 60)
    print("6. Zero-shot predictor initializing (seen + unseen)...")
    print("=" * 60)
    # Predictor'ı önce varsayılan 0.5 threshold ile oluştur
    predictor = FZSLPredictor(
        model=fzsl_model,
        class_text_embeddings=all_text_emb_matrix,
        class_order=all_class_order,
        device=device,
        optimal_threshold=0.5,  # aşağıda PR eğrisinden optimize edilecek
    )

    # ── 7. Değerlendirme ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. Evaluation")
    print("=" * 60)

    # -- Threshold optimizasyonu (tum test seti uzerinde)
    print("\nOptimal threshold aranıyor...")
    full_fraud_proba = predictor.is_fraud_proba(X_test)
    opt_thresholds = find_optimal_threshold(y_binary_test, full_fraud_proba, min_recall=0.80)
    best_threshold = opt_thresholds["f1_optimal"]["threshold"]

    # Bulunan optimal threshold'u predictor'a kaydet
    predictor.optimal_threshold = best_threshold
    print(f"   Predictor optimal threshold güncellendi: {best_threshold:.4f}")

    # Tum test seti — optimal threshold (tek referans noktası)
    evaluate_zsl_binary(
        predictor, X_test, y_binary_test, subtypes_test,
        label="Full Test Set @ Optimal F1 Threshold",
        threshold=best_threshold
    )

    # 8c. Sadece seen siniflar — optimal threshold
    seen_mask = np.isin(subtypes_test, SEEN_CLASSES)
    X_seen_test   = X_test[seen_mask]
    y_seen_test   = y_binary_test[seen_mask]
    sub_seen_test  = subtypes_test[seen_mask]
    evaluate_zsl_binary(
        predictor, X_seen_test, y_seen_test, sub_seen_test,
        label="Seen Classes @ Optimal F1 Threshold",
        threshold=best_threshold
    )

    # 8d. Sadece unseen fraud — zero-shot ana degerlendirme
    unseen_mask = subtypes_test == UNSEEN_CLASS
    X_unseen = X_test[unseen_mask]
    if len(X_unseen) > 0:
        evaluate_zsl_unseen(predictor, X_unseen, all_class_order)
    else:
        print(f"\n[WARNING] No samples found for {UNSEEN_CLASS} in test set!")

    # ── 9. Görselleştirme ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("8. Generating visualizations...")
    print("=" * 60)

    # PR Curve
    plot_pr_curve(
        y_binary_test, full_fraud_proba, opt_thresholds,
        save_path="outputs/fzsl_pr_curve.png"
    )

    # Loss curve
    plot_loss_curve(loss_history, "outputs/fzsl_loss_curve.png")

    # Similarity heatmap
    samples_by_class = {}
    for cls in all_class_order:
        mask = subtypes_test == cls
        if mask.sum() > 0:
            X_cls = X_test[mask][:200]
            samples_by_class[cls] = X_cls

    plot_similarity_heatmap(
        predictor, samples_by_class,
        save_path="outputs/fzsl_similarity_heatmap.png"
    )

    # ── 10. Model kaydet ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("9. Saving model...")
    print("=" * 60)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(fzsl_model.state_dict(), "checkpoints/fzsl_model.pt")
    print("   Kaydedildi: checkpoints/fzsl_model.pt")

    print("\n" + "=" * 60)
    print("[DONE] FZSL Pipeline completed!")
    print("   Outputs: outputs/fzsl_loss_curve.png")
    print("            outputs/fzsl_similarity_heatmap.png")
    print("   Model  : checkpoints/fzsl_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
