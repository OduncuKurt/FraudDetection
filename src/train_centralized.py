# Threshold deneyi halen aktif. Unutma!
from src.data_prep import load_and_preprocess_data
from src.model import MLPFraudDetector
from src.fl_utils import evaluate_model, get_dataloader
import torch
from sklearn.metrics import confusion_matrix
from src.xai_analysis import compute_permutation_importance, print_top_features, explain_single_prediction
import numpy as np
from src.shap_analysis import (
    compute_shap_values_deep,
    print_mean_abs_shap,
    print_local_shap_explanation,
)
import shap
import matplotlib.pyplot as plt
import os

def evaluate_thresholds(model, X_test, y_test, thresholds=None, device="cpu"):
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.8, 0.9]

    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        metrics = evaluate_model(model, X_test, y_test, threshold=threshold, device=device)

        print(f"\n=== Threshold: {threshold} ===")
        print("Confusion Matrix:")
        print(cm)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


def train_centralized(model, X, y, epochs=10, lr=1e-3, batch_size=256, device="cpu"):
    model.to(device)
    model.train()

    loader = get_dataloader(X, y, batch_size=batch_size, shuffle=True)

    fraud_count = max((y == 1).sum(), 1)
    normal_count = max((y == 0).sum(), 1)
    pos_weight = torch.tensor([normal_count / fraud_count], dtype=torch.float32).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model


def print_confusion_matrix(model, X_test, y_test, threshold=0.5, device="cpu"):
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    print("Confusion Matrix:")
    print(cm)


def main():
    csv_path = "data/creditcard.csv"
    device = "cpu"

    print("Loading data...")
    # X_train, X_test, y_train, y_test, _, feature_names = load_and_preprocess_data(csv_path)
    X_train, X_test, y_train, y_test, _, feature_names = load_and_preprocess_data(csv_path)

    input_dim = X_train.shape[1]
    model = MLPFraudDetector(input_dim=input_dim)

    print("Training centralized model...")
    model = train_centralized(
        model=model,
        X=X_train,
        y=y_train,
        epochs=10,
        lr=1e-3,
        batch_size=256,
        device=device
    )

    print_confusion_matrix(model, X_test, y_test, threshold=0.5, device=device)

    metrics = evaluate_model(model, X_test, y_test, threshold=0.5, device=device)

    print("Centralized model metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    evaluate_thresholds(model, X_test, y_test, thresholds=[0.5, 0.7, 0.8, 0.9], device=device)

    baseline_score, importances = compute_permutation_importance(
    model=model,
    X=X_test,
    y=y_test,
    feature_names=feature_names,
    n_repeats=3,
    device=device
    )
    print(f"\nBaseline PR-AUC used for permutation importance: {baseline_score:.6f}")
    print_top_features(importances, top_k=10)

    # --- SHAP ANALYSIS ---
    # Arka plan için train setinden küçük bir örnek
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    X_background = X_train[bg_idx]

    # Açıklanacak örnekler:
    # birkaç fraud + birkaç normal örnek seçelim
    fraud_idx = np.where(y_test == 1)[0][:10]
    normal_idx = np.where(y_test == 0)[0][:10]
    selected_idx = np.concatenate([fraud_idx, normal_idx])

    X_explain = X_test[selected_idx]
    y_explain = y_test[selected_idx]

    explainer, shap_values = compute_shap_values_deep(
        model=model,
        X_background=X_background,
        X_explain=X_explain,
        device=device,
        check_additivity=False
    )

    print_mean_abs_shap(shap_values, feature_names, top_k=10)

    # İlk fraud örneği
    if len(fraud_idx) > 0:
        print_local_shap_explanation(
            shap_values=shap_values,
            X_explain=X_explain,
            y_true=y_explain,
            feature_names=feature_names,
            sample_index=0,
            top_k=10
        )

    # İlk normal örneği (fraud örneklerinden sonra geliyor)
    if len(normal_idx) > 0:
        print_local_shap_explanation(
            shap_values=shap_values,
            X_explain=X_explain,
            y_true=y_explain,
            feature_names=feature_names,
            sample_index=len(fraud_idx),
            top_k=10
        )

    os.makedirs("outputs", exist_ok=True)

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()