import numpy as np
import torch
from sklearn.metrics import average_precision_score


def predict_proba_torch(model, X, device="cpu"):
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    return probs


def compute_permutation_importance(model, X, y, feature_names, metric_fn=None, n_repeats=3, device="cpu"):
    """
    Permutation importance:
    Bir feature karıştırıldığında performans ne kadar düşüyor?
    Daha çok düşüş = daha önemli feature
    """

    if metric_fn is None:
        metric_fn = average_precision_score  # PR-AUC benzeri mantık için iyi başlangıç

    baseline_probs = predict_proba_torch(model, X, device=device)
    baseline_score = metric_fn(y, baseline_probs)

    importances = []

    for j, feature_name in enumerate(feature_names):
        drops = []

        for _ in range(n_repeats):
            X_permuted = X.copy()
            shuffled_col = X_permuted[:, j].copy()
            np.random.shuffle(shuffled_col)
            X_permuted[:, j] = shuffled_col

            permuted_probs = predict_proba_torch(model, X_permuted, device=device)
            permuted_score = metric_fn(y, permuted_probs)

            drop = baseline_score - permuted_score
            drops.append(drop)

        mean_drop = float(np.mean(drops))
        importances.append((feature_name, mean_drop))

    importances.sort(key=lambda x: x[1], reverse=True)

    return baseline_score, importances


def print_top_features(importances, top_k=10):
    print(f"\nTop {top_k} most important features:")
    for rank, (feature, score_drop) in enumerate(importances[:top_k], start=1):
        print(f"{rank}. {feature:<10} importance_drop={score_drop:.6f}")


def explain_single_prediction(model, X, y, feature_names, sample_index, top_k=10, device="cpu"):
    """
    Basit local explanation:
    Örneğin feature değerlerini büyüklük sırasına göre gösterir.
    Bu tam SHAP değil ama ilk lokal yorum için yararlı.
    """

    probs = predict_proba_torch(model, X[sample_index:sample_index + 1], device=device)
    pred_prob = probs[0]
    true_label = y[sample_index]

    sample = X[sample_index]
    abs_sorted_idx = np.argsort(np.abs(sample))[::-1]

    print("\nSingle sample explanation")
    print(f"Sample index    : {sample_index}")
    print(f"True label      : {true_label}")
    print(f"Predicted prob  : {pred_prob:.6f}")

    print(f"\nTop {top_k} largest-magnitude feature values in this sample:")
    for i in abs_sorted_idx[:top_k]:
        print(f"{feature_names[i]:<10} value={sample[i]:.6f}")