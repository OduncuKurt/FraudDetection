"""
system.py
---------
Unified Fraud Detection System: FL + FZSL + XAI

Egitim akisi:
  1. FL (FedAvg) -> global MLP backbone (30-dim -> 32-dim)
  2. FL backbone frozen -> FZSL egitimi (32-dim -> fraud tipi)
  3. SHAP explainer kurulumu

Tahmin akisi:
  islem (30-dim) -> FL backbone -> 32-dim
      -> FL head -> fraud olasiligi (binary)
      -> FZSL   -> fraud tipi (zero-shot, gorulmemis tipler dahil)
      -> SHAP   -> hangi ozellik neden etkiledi
"""

import os
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
)
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import MLPFraudDetector
from src.fl_utils import train_local, fedavg
from src.fzsl.fraud_subtypes import (
    load_creditcard_dataset, generate_fraud_subtypes,
    build_seen_unseen_datasets,
)
from src.fzsl.class_descriptions import (
    FRAUD_CLASS_DESCRIPTIONS, SEEN_CLASSES, UNSEEN_CLASS,
    CLASS_TO_IDX, IDX_TO_CLASS,
)
from src.fzsl.zsl_encoder import TextClassEncoder
from src.fzsl.fzsl_model import FZSLModel, FZSLPredictor

CHECKPOINT_PATH = os.path.join("checkpoints", "fraud_system.pkl")
FEATURE_DROP = ["Class", "fraud_cluster", "fraud_subtype", "fzsl_split"]


# ---------------------------------------------------------------------------
#  Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

def _get_feature_cols(df):
    return [c for c in df.columns if c not in FEATURE_DROP]


def _infonce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def _find_optimal_threshold(y_true, proba, min_recall=0.80):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    return float(thresholds[np.argmax(f1s)])


# ---------------------------------------------------------------------------
#  Ana sinif
# ---------------------------------------------------------------------------

class FraudDetectionSystem:
    """
    FL + FZSL + XAI'yi tek bir arayuz altinda birlestiren sistem.

    Kullanim:
        sys = FraudDetectionSystem()
        sys.train("data/creditcard.csv")
        result = sys.predict(X_single)
        sys.save()

        # Sonraki oturumda:
        sys = FraudDetectionSystem.load()
        result = sys.predict(X_single)
    """

    def __init__(self):
        self.fl_model       = None   # MLPFraudDetector (FedAvg ile egitilmis)
        self.fzsl_model     = None   # FZSLModel (FL backbone uzerine)
        self.predictor      = None   # FZSLPredictor (zero-shot inference)
        self.scaler         = None   # StandardScaler
        self.feature_names  = None   # list[str]
        self.fl_threshold   = 0.5    # binary karar esigi
        self.class_order    = SEEN_CLASSES + [UNSEEN_CLASS]
        self.shap_background = None  # SHAP arka plan ornekleri
        self.is_trained     = False

    # -------------------------------------------------------------------
    #  Ozellik cikarimi (FL backbone)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def _fl_features(self, X: np.ndarray, device="cpu") -> np.ndarray:
        """30-dim ham veri -> 32-dim FL backbone ozellikleri."""
        self.fl_model.eval()
        t = torch.tensor(X, dtype=torch.float32).to(device)
        return self.fl_model.get_features(t).cpu().numpy()

    @torch.no_grad()
    def _fl_proba(self, X: np.ndarray, device="cpu") -> np.ndarray:
        """30-dim ham veri -> fraud olasiligi (sigmoid)."""
        self.fl_model.eval()
        t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = self.fl_model(t)
        return torch.sigmoid(logits).cpu().numpy().flatten()

    # -------------------------------------------------------------------
    #  FL egitimi (FedAvg)
    # -------------------------------------------------------------------

    def _train_fl(self, X_train, y_train, input_dim,
                  num_clients=4, num_rounds=5, local_epochs=2,
                  lr=1e-3, device="cpu"):
        print("\n[FL] Federated Learning basladi...")

        # Non-IID label-skew split
        from src.data_prep import split_into_clients_noniid_label_skew
        clients = split_into_clients_noniid_label_skew(
            X_train, y_train,
            num_clients=num_clients,
            fraud_ratios=[0.5, 0.25, 0.15, 0.10],
        )

        global_model = MLPFraudDetector(input_dim=input_dim)

        for rnd in range(num_rounds):
            local_states = []
            for cid, (cx, cy) in enumerate(clients, 1):
                state = train_local(global_model, cx, cy,
                                    epochs=local_epochs, lr=lr, device=device)
                local_states.append(state)
                print(f"  Round {rnd+1}/{num_rounds}  Client {cid}/{num_clients}", end="\r")

            global_model.load_state_dict(fedavg(local_states))

        print(f"\n[FL] {num_rounds} tur tamamlandi.")
        return global_model

    # -------------------------------------------------------------------
    #  FZSL egitimi (FL backbone uzerine)
    # -------------------------------------------------------------------

    def _train_fzsl(self, X_train_raw, y_fzsl, seen_text_emb, text_dim,
                    epochs=30, lr=1e-3, batch_size=256, device="cpu"):
        """
        X_train_raw : 30-dim olcekli veri
        y_fzsl      : CLASS_TO_IDX ile kodlanmis etiketler (seen siniflar)
        """
        print("\n[FZSL] FL backbone ozellikleri cikariliyor...")
        X_fl = self._fl_features(X_train_raw, device=device)   # [N, 32]
        print(f"[FZSL] FL ozellikleri: {X_fl.shape}")

        # FL backbone donuk (frozen) - FZSL sadece projection head ogrenir
        fzsl_model = FZSLModel(
            input_dim=MLPFraudDetector.FEATURE_DIM,  # 32
            text_dim=text_dim,
            proj_dim=128,
            temperature=0.07,
        ).to(device)

        text_tensor = torch.tensor(seen_text_emb, dtype=torch.float32).to(device)

        # WeightedSampler: sinif dengesizligini gider
        counts = np.bincount(y_fzsl)
        weights_per_class = 1.0 / (counts.astype(float) + 1e-8)
        sample_w = torch.tensor([weights_per_class[l] for l in y_fzsl], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

        ds = TensorDataset(
            torch.tensor(X_fl, dtype=torch.float32),
            torch.tensor(y_fzsl, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)

        optimizer = torch.optim.AdamW(fzsl_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss, best_state = float("inf"), None
        print(f"[FZSL] {epochs} epoch egitim basladi (input=32-dim FL features)...")

        for ep in range(epochs):
            fzsl_model.train()
            total = 0.0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits = fzsl_model(bx, text_tensor)
                loss = _infonce_loss(logits, by)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fzsl_model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            avg = total / len(loader)
            if avg < best_loss:
                best_loss = avg
                best_state = copy.deepcopy(fzsl_model.state_dict())
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={avg:.5f}")

        fzsl_model.load_state_dict(best_state)
        print(f"[FZSL] En iyi loss: {best_loss:.5f}")
        return fzsl_model

    # -------------------------------------------------------------------
    #  Ana egitim pipeline'i
    # -------------------------------------------------------------------

    def train(self, csv_path: str, device: str = "cpu",
              fl_rounds: int = 5, fzsl_epochs: int = 30):
        print("=" * 60)
        print("  Unified Fraud Detection System - Egitim")
        print("=" * 60)

        # 1. Veri yukle + fraud subtype'lari uret
        print("\n[1/5] Veri yukleniyor ve fraud alt tipleri olusturuluyor...")
        df = load_creditcard_dataset(csv_path)
        df_labeled, _, _ = generate_fraud_subtypes(df, n_clusters=4)
        train_df, test_df, _ = build_seen_unseen_datasets(df_labeled, unseen_subtype=UNSEEN_CLASS)

        # Feature kolonlari + scaler
        feat_cols = _get_feature_cols(train_df)
        self.feature_names = feat_cols

        # Seen siniflar icin egitim verisi
        seen_mask = train_df["fraud_subtype"].isin(SEEN_CLASSES)
        df_seen = train_df[seen_mask]

        X_all_raw = df_seen[feat_cols].values.astype(np.float32)
        self.scaler = StandardScaler()
        X_all_scaled = self.scaler.fit_transform(X_all_raw)
        y_binary = df_seen["Class"].values.astype(np.float32)
        y_fzsl   = df_seen["fraud_subtype"].map(CLASS_TO_IDX).values.astype(np.int64)

        # Test verisi
        X_test_raw = test_df[feat_cols].values.astype(np.float32)
        X_test = self.scaler.transform(X_test_raw)
        y_test_binary = test_df["Class"].values
        y_test_subtypes = test_df["fraud_subtype"].values

        input_dim = X_all_scaled.shape[1]

        # 2. FL egitimi
        print("\n[2/5] Federated Learning (FedAvg) basladi...")
        self.fl_model = self._train_fl(
            X_all_scaled, y_binary, input_dim,
            num_rounds=fl_rounds, device=device,
        )

        # FL binary threshold optimize et
        fl_proba_test = self._fl_proba(X_test, device=device)
        self.fl_threshold = _find_optimal_threshold(y_test_binary, fl_proba_test)
        print(f"[FL] Optimal threshold: {self.fl_threshold:.4f}")
        self._print_binary_metrics("FL Binary", y_test_binary, fl_proba_test, self.fl_threshold)

        # 3. Sinif aciklamalarini embed et
        print("\n[3/5] Sinif aciklamalari embed ediliyor (TF-IDF+RandomProjection)...")
        encoder = TextClassEncoder(embed_dim=128, backend="local")
        all_text_emb, self.class_order = encoder.get_class_embedding_matrix(
            FRAUD_CLASS_DESCRIPTIONS, class_order=SEEN_CLASSES + [UNSEEN_CLASS]
        )
        seen_text_emb, _ = encoder.get_class_embedding_matrix(
            FRAUD_CLASS_DESCRIPTIONS, class_order=SEEN_CLASSES
        )
        text_dim = all_text_emb.shape[1]

        # 4. FZSL egitimi (FL backbone ozellikleri uzerinde)
        print("\n[4/5] FZSL egitimi (FL backbone ozellikleri kullaniliyor)...")
        self.fzsl_model = self._train_fzsl(
            X_all_scaled, y_fzsl, seen_text_emb, text_dim,
            epochs=fzsl_epochs, device=device,
        )

        # FL features ile FZSL predictor olustur
        self.predictor = self._make_predictor(all_text_emb, device)

        # FZSL zero-shot degerlendirmesi
        self._evaluate_fzsl(X_test, y_test_binary, y_test_subtypes, device)

        # 5. SHAP arka plan kur
        print("\n[5/5] SHAP explainer kuruluyor...")
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_all_scaled), size=min(200, len(X_all_scaled)), replace=False)
        self.shap_background = X_all_scaled[bg_idx]

        self.is_trained = True
        print("\n[DONE] Sistem egitimi tamamlandi.")

    def _make_predictor(self, all_text_emb, device):
        """FL ozellikleri ile calisan FZSL predictor."""

        class FLBackedPredictor(FZSLPredictor):
            """FL backbone'u FZSL onunde kullanan predictor."""
            def __init__(self, fl_model, fzsl_model, text_emb, class_order, device):
                self.fl_model    = fl_model
                self.device      = device
                self.class_order = class_order
                fzsl_model.eval()
                with torch.no_grad():
                    t = torch.tensor(text_emb, dtype=torch.float32).to(device)
                    self.class_protos = fzsl_model.encode_texts(t).cpu().numpy()
                self.model = fzsl_model

            def predict(self, X_scaled):
                # X_scaled: 30-dim olcekli ham veri
                fl_feat = self._get_fl_feat(X_scaled)
                self.model.eval()
                with torch.no_grad():
                    t = torch.tensor(fl_feat, dtype=torch.float32).to(self.device)
                    tx_emb = self.model.encode_transactions(t).cpu().numpy()
                sims = tx_emb @ self.class_protos.T
                idx  = sims.argmax(axis=1)
                return [self.class_order[i] for i in idx], sims

            def _get_fl_feat(self, X_scaled):
                self.fl_model.eval()
                with torch.no_grad():
                    t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
                    return self.fl_model.get_features(t).cpu().numpy()

            def is_fraud_proba(self, X_scaled, fraud_classes=None):
                if fraud_classes is None:
                    fraud_classes = [c for c in self.class_order if c != "normal"]
                _, sims = self.predict(X_scaled)
                exp = np.exp(sims - sims.max(axis=1, keepdims=True))
                proba = exp / exp.sum(axis=1, keepdims=True)
                fidx = [i for i, c in enumerate(self.class_order) if c in fraud_classes]
                return proba[:, fidx].sum(axis=1)

        return FLBackedPredictor(
            self.fl_model, self.fzsl_model,
            all_text_emb, self.class_order, device
        )

    # -------------------------------------------------------------------
    #  Degerlendirme (ic)
    # -------------------------------------------------------------------

    def _print_binary_metrics(self, label, y_true, proba, threshold):
        preds = (proba >= threshold).astype(int)
        print(f"  [{label}] Precision={precision_score(y_true, preds, zero_division=0):.4f} "
              f"Recall={recall_score(y_true, preds, zero_division=0):.4f} "
              f"F1={f1_score(y_true, preds, zero_division=0):.4f} "
              f"ROC-AUC={roc_auc_score(y_true, proba):.4f}")

    def _evaluate_fzsl(self, X_test, y_binary, subtypes, device):
        fraud_proba = self.predictor.is_fraud_proba(X_test)
        thr = _find_optimal_threshold(y_binary, fraud_proba)
        print(f"\n[FZSL] Optimal threshold: {thr:.4f}")
        self._print_binary_metrics("FZSL Binary", y_binary, fraud_proba, thr)

        # Zero-shot (unseen)
        unseen_mask = subtypes == UNSEEN_CLASS
        if unseen_mask.sum() > 0:
            X_u = X_test[unseen_mask]
            preds, _ = self.predictor.predict(X_u)
            rate = np.mean([p != "normal" for p in preds])
            print(f"  [FZSL Zero-Shot] Gorulmemis fraud tespit orani: {rate:.4f} "
                  f"({int(rate * len(X_u))}/{len(X_u)} ornek)")

    # -------------------------------------------------------------------
    #  Tahmin API'si
    # -------------------------------------------------------------------

    def predict(self, X_scaled: np.ndarray, device: str = "cpu") -> dict:
        """
        Parametreler:
            X_scaled: [N, 30]  StandardScaler uygulanmis islem ozellikleri

        Donus:
            list[dict] - her islem icin:
                is_fraud, fl_probability, fraud_type, fraud_type_description, confidence
        """
        assert self.is_trained, "Once train() cagirin!"

        fl_proba   = self._fl_proba(X_scaled, device)
        fzsl_types, sims = self.predictor.predict(X_scaled)
        fzsl_proba = self.predictor.is_fraud_proba(X_scaled)

        results = []
        for i in range(len(X_scaled)):
            ftype = fzsl_types[i]
            # Nihai karar: FL OR FZSL
            is_fraud = bool(fl_proba[i] >= self.fl_threshold) or (ftype != "normal")
            conf     = float(np.max(sims[i]))

            results.append({
                "is_fraud":               is_fraud,
                "fl_probability":         float(fl_proba[i]),
                "fzsl_fraud_probability": float(fzsl_proba[i]),
                "fraud_type":             ftype,
                "fraud_type_description": FRAUD_CLASS_DESCRIPTIONS.get(ftype, ""),
                "similarity_scores":      {cls: float(sims[i][j])
                                           for j, cls in enumerate(self.class_order)},
                "confidence":             conf,
            })
        return results

    def explain(self, X_scaled: np.ndarray, sample_idx: int = 0,
                top_k: int = 10, device: str = "cpu") -> dict:
        """
        SHAP ile tahmin aciklamasi.

        Donus:
            {feature_name: shap_value} (top_k ozellik)
        """
        assert self.is_trained and self.shap_background is not None

        def model_fn(x):
            t = torch.tensor(x, dtype=torch.float32).to(device)
            with torch.no_grad():
                return torch.sigmoid(self.fl_model(t)).cpu().numpy()

        explainer   = shap.KernelExplainer(model_fn, self.shap_background[:50])
        shap_vals   = explainer.shap_values(X_scaled[sample_idx:sample_idx+1], nsamples=100)
        sv          = shap_vals[0].flatten()
        ranked_idx  = np.argsort(np.abs(sv))[::-1][:top_k]

        return {
            self.feature_names[i]: float(sv[i])
            for i in ranked_idx
        }

    # -------------------------------------------------------------------
    #  Kaydet / Yukle
    # -------------------------------------------------------------------

    def save(self, path: str = CHECKPOINT_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "fl_model_state":      self.fl_model.state_dict(),
            "fl_input_dim":        next(self.fl_model.parameters()).shape[1],
            "fzsl_model_state":    self.fzsl_model.state_dict(),
            "fzsl_cfg": {
                "input_dim": MLPFraudDetector.FEATURE_DIM,
                "text_dim":  self.fzsl_model.text_proj.proj[0].in_features,
                "proj_dim":  self.fzsl_model.tx_encoder.proj_dim,
            },
            "scaler":              self.scaler,
            "feature_names":       self.feature_names,
            "fl_threshold":        self.fl_threshold,
            "class_order":         self.class_order,
            "shap_background":     self.shap_background,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[System] Kaydedildi: {path}")

    @classmethod
    def load(cls, path: str = CHECKPOINT_PATH, device: str = "cpu"):
        with open(path, "rb") as f:
            data = pickle.load(f)

        sys = cls()
        sys.feature_names   = data["feature_names"]
        sys.fl_threshold    = data["fl_threshold"]
        sys.class_order     = data["class_order"]
        sys.shap_background = data["shap_background"]
        sys.scaler          = data["scaler"]

        # FL model
        sys.fl_model = MLPFraudDetector(input_dim=data["fl_input_dim"])
        sys.fl_model.load_state_dict(data["fl_model_state"])
        sys.fl_model.eval()

        # FZSL model
        cfg = data["fzsl_cfg"]
        sys.fzsl_model = FZSLModel(**cfg)
        sys.fzsl_model.load_state_dict(data["fzsl_model_state"])
        sys.fzsl_model.eval()

        # Text embeddinglari yeniden uret
        encoder = TextClassEncoder(embed_dim=cfg["proj_dim"], backend="local")
        all_text_emb, _ = encoder.get_class_embedding_matrix(
            FRAUD_CLASS_DESCRIPTIONS, class_order=sys.class_order
        )
        sys.predictor = sys._make_predictor(all_text_emb, device)
        sys.is_trained = True

        print(f"[System] Yuklendi: {path}")
        return sys
