import numpy as np
import torch
import shap


def compute_shap_values_deep(model, X_background, X_explain, device="cpu", check_additivity=False):
    """
    PyTorch MLP için SHAP DeepExplainer.
    - X_background: arka plan örnekleri (örn. train'den 100 örnek)
    - X_explain: açıklanacak örnekler (örn. test'ten 20 örnek)
    """

    model.to(device)
    model.eval()

    background_tensor = torch.tensor(X_background, dtype=torch.float32).to(device)
    explain_tensor = torch.tensor(X_explain, dtype=torch.float32).to(device)

    explainer = shap.DeepExplainer(model, background_tensor)
    shap_values = explainer.shap_values(explain_tensor, check_additivity=check_additivity)

    # Binary single-output modelde bazen direkt ndarray, bazen liste dönebilir
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return explainer, shap_values


def print_mean_abs_shap(shap_values, feature_names, top_k=10):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_k} features by mean(|SHAP|):")
    for i, (fname, score) in enumerate(ranked[:top_k], start=1):
        print(f"{i}. {fname:<10} mean_abs_shap={score:.6f}")


def print_local_shap_explanation(shap_values, X_explain, y_true, feature_names, sample_index=0, top_k=10):
    sample_shap = shap_values[sample_index]
    sample_x = X_explain[sample_index]
    true_label = y_true[sample_index]

    ranked_idx = np.argsort(np.abs(sample_shap))[::-1]

    print("\nLocal SHAP explanation")
    print(f"Sample index : {sample_index}")
    print(f"True label   : {true_label}")

    print(f"\nTop {top_k} SHAP contributors:")
    for idx in ranked_idx[:top_k]:
        print(
            f"{feature_names[idx]:<10} "
            f"feature_value={sample_x[idx]:>10.6f} "
            f"shap_value={sample_shap[idx]:>10.6f}"
        )