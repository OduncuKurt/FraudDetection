"""
xai_engine.py — v2
-------------------
Fraud Detection için XAI motoru.

Birincil yöntem: Gradient × Input (GxI)
  - shap kütüphanesi gerekmez
  - Per-transaction, gerçek zamanlı (ms)
  - Pozitif = fraud ihtimalini artıran özellik
  - Negatif = fraud ihtimalini azaltan özellik
  - Akademik literatürde geçerli bir attribution yöntemi

İkincil yöntem: shap.GradientExplainer (isteğe bağlı, daha yavaş)

Aynı zamanda: Gradient-based Counterfactual açıklama
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class XAIEngine:
    def __init__(self, fl_model: nn.Module, scaler, feature_names: list,
                 background_data: np.ndarray, device: str = "cpu"):
        self.model = fl_model.to(device).eval()
        self.scaler = scaler
        self.feature_names = feature_names
        self.device = device
        self.n_features = len(feature_names)
        self._bg = background_data[:100]

    # ─── 1. Gradient × Input (Birincil, hızlı, shap kütüphanesi gerektirmez) ─
    def shap_values(self, x_scaled: np.ndarray) -> Optional[dict]:
        """
        Gradient × Input attribution.
        Signed: pozitif → fraud ihtimalini artıran özellik
                negatif → fraud ihtimalini azaltan özellik
        Normalleştirilmemiş, gerçek değerler döner.
        """
        try:
            x_arr = x_scaled.flatten()
            x_t = torch.tensor(x_arr, dtype=torch.float32,
                               device=self.device, requires_grad=True)
            logit = self.model(x_t.unsqueeze(0))
            prob = torch.sigmoid(logit)
            prob.squeeze().backward()

            grads = x_t.grad.detach().cpu().numpy()
            # Signed gradient × input
            gxi = grads * x_arr
            return {self.feature_names[i]: float(gxi[i])
                    for i in range(self.n_features)}
        except Exception as e:
            print(f"[XAI] GxI hata: {e}")
            return None

    # ─── 2. Counterfactual (gradient descent) ─────────────────────────────────
    def counterfactual(self, x_scaled: np.ndarray,
                       target_prob: float = 0.40,
                       steps: int = 200,
                       lr: float = 0.05) -> dict:
        """
        "X değerleri ne olursa sistem bu işlemi NORMAL görür?"
        Gradient descent ile minimum değişim hesapla.
        """
        x_t = torch.tensor(x_scaled.reshape(1, -1),
                            dtype=torch.float32, device=self.device)
        with torch.no_grad():
            orig_prob = float(torch.sigmoid(self.model(x_t)).item())

        if orig_prob < 0.5:
            return {
                "original_prob": round(orig_prob, 4),
                "counterfactual_prob": round(orig_prob, 4),
                "changes": [],
                "success": True,
                "verdict": "İşlem zaten normal olarak sınıflandırılıyor.",
            }

        x_cf = x_t.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=lr)
        lambda_reg = 0.5

        for _ in range(steps):
            optimizer.zero_grad()
            prob = torch.sigmoid(self.model(x_cf))
            loss = prob + lambda_reg * torch.mean((x_cf - x_t.detach()) ** 2)
            loss.backward()
            optimizer.step()
            if prob.item() < target_prob:
                break

        cf_arr = x_cf.detach().cpu().numpy().flatten()
        orig_arr = x_scaled.flatten()
        with torch.no_grad():
            cf_prob = float(torch.sigmoid(self.model(x_cf)).item())

        diffs = []
        for i, fname in enumerate(self.feature_names):
            delta = float(cf_arr[i] - orig_arr[i])
            if abs(delta) > 0.05:
                diffs.append({
                    "feature": fname,
                    "original": round(float(orig_arr[i]), 4),
                    "counterfactual": round(float(cf_arr[i]), 4),
                    "change": round(delta, 4),
                    "change_pct": f"{delta / max(abs(float(orig_arr[i])), 0.001) * 100:.1f}%",
                })
        diffs.sort(key=lambda x: -abs(x["change"]))
        success = cf_prob < 0.5
        top3 = [d["feature"] for d in diffs[:3]]

        if success:
            verdict = (
                f"Bu işlem {', '.join(top3)} özelliklerindeki değişimlerle "
                f"NORMAL (fraud değil) olarak sınıflandırılabilirdi. "
                f"Fraud olasılığı %{orig_prob*100:.1f}'den %{cf_prob*100:.1f}'e düşerdi."
            )
        else:
            verdict = (
                f"Bu işlem çok güçlü fraud sinyalleri içeriyor "
                f"(ulaşılan olasılık: %{cf_prob*100:.1f}). "
                f"En çok değişim gerektiren özellikler: {', '.join(top3)}."
            )

        return {
            "original_prob": round(orig_prob, 4),
            "counterfactual_prob": round(cf_prob, 4),
            "changes": diffs[:8],
            "n_features_changed": len(diffs),
            "success": success,
            "verdict": verdict,
        }

    # ─── 3. Tam XAI Raporu ────────────────────────────────────────────────────
    def full_explain(self, x_scaled: np.ndarray, fraud_type: str,
                     amount: float, time_sec: float,
                     fl_probability: float, fzsl_fraud_prob: float,
                     confidence: float, similarity_scores: dict) -> dict:
        shap_dict = self.shap_values(x_scaled)
        cf = self.counterfactual(x_scaled)

        contributions = []
        if shap_dict:
            total_abs = sum(abs(v) for v in shap_dict.values()) or 1.0
            top10 = sorted(shap_dict.items(), key=lambda x: -abs(x[1]))[:10]
            contributions = [
                {
                    "feature": f,
                    "shap": round(v, 4),
                    "contribution_pct": round(abs(v) / total_abs * 100, 1),
                    "direction": "fraud↑" if v > 0 else "fraud↓",
                    "feature_value": round(
                        float(x_scaled.flatten()[self.feature_names.index(f)]), 4
                    ) if f in self.feature_names else 0,
                }
                for f, v in top10
            ]

        return {
            "shap_values": shap_dict,
            "contributions": contributions,
            "counterfactual": cf,
            "metadata": {
                "fraud_type": fraud_type,
                "amount": amount,
                "fl_probability": fl_probability,
                "shap_method": "Gradient×Input",
            },
        }
