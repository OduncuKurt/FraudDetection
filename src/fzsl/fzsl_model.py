"""
fzsl_model.py
-------------
FZSL (Federated Zero-Shot Learning) için model bileşenleri.

Mimari (CLIP-benzeri, tabular veriye uyarlanmış):
─────────────────────────────────────────────────
  Transaction branch:
      input_dim → Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Linear(proj_dim) → L2-norm

  Text branch (SBERT çıktısı):
      text_embed_dim → Linear(proj_dim) → L2-norm

  Eşleşme skoru: cosine_similarity = dot(tx_emb, text_emb)   [her ikisi de L2-normalize]
  Loss:  InfoNCE / NT-Xent  (CLIP'teki gibi)

Zero-shot inference:
  Tüm sınıf açıklamaları (seen + unseen) embed edilir.
  Yeni işlem için benzerlik hesaplanır → en yakın sınıf seçilir.
  Bu sayede eğitimde hiç görülmeyen fraud_type_3 bile tespit edilebilir.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransactionEncoder(nn.Module):
    """
    İşlem (transaction) özellik vektörünü ortak embedding uzayına projekte eder.

    Çıktı: L2-normalize edilmiş proj_dim boyutlu vektör.
    """

    def __init__(self, input_dim: int, proj_dim: int = 128):
        super().__init__()
        self.proj_dim = proj_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, dim=-1)   # L2 normalize


class TextProjection(nn.Module):
    """
    SBERT embedding'ini (text_dim) ortak embedding uzayına (proj_dim) projekte eder.

    Çıktı: L2-normalize edilmiş proj_dim boyutlu vektör.
    Bu katman eğitim sırasında öğrenilir → daha iyi alignment sağlanır.
    """

    def __init__(self, text_dim: int = 384, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        return F.normalize(out, dim=-1)   # L2 normalize


class FZSLModel(nn.Module):
    """
    Tam FZSL modeli: TransactionEncoder + TextProjection.

    Training:
        forward() → logits = (tx_emb @ text_emb.T) / temperature
        loss = CrossEntropy(logits, class_labels)   [InfoNCE]

    Inference (zero-shot):
        Tüm sınıfların text embedding'leri hesaplanır (seen + unseen).
        transaction → tx_emb → similarity with each class → argmax
    """

    def __init__(
        self,
        input_dim: int,
        text_dim: int = 384,
        proj_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.tx_encoder = TransactionEncoder(input_dim=input_dim, proj_dim=proj_dim)
        self.text_proj = TextProjection(text_dim=text_dim, proj_dim=proj_dim)
        self.temperature = nn.Parameter(
            torch.tensor(temperature), requires_grad=True
        )

    def forward(
        self,
        x_transactions: torch.Tensor,
        x_texts: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_transactions : [batch_size, input_dim]
        x_texts        : [num_classes, text_dim]

        Dönüş:
            logits : [batch_size, num_classes]  (scaled cosine similarity)
        """
        tx_emb = self.tx_encoder(x_transactions)       # [B, proj_dim]
        text_emb = self.text_proj(x_texts)             # [num_classes, proj_dim]

        # Cosine similarity (her ikisi normalize edildiği için dot product yeterli)
        logits = tx_emb @ text_emb.T                   # [B, num_classes]
        logits = logits * self.temperature.exp().clamp(max=100)  # scaled
        return logits

    def encode_transactions(self, x: torch.Tensor) -> torch.Tensor:
        """Sadece transaction encoding (inference için)."""
        return self.tx_encoder(x)

    def encode_texts(self, x: torch.Tensor) -> torch.Tensor:
        """Sadece text projection (class prototype'lar için)."""
        return self.text_proj(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Yardımcı: numpy transaction array'inden probability prediction
# ─────────────────────────────────────────────────────────────────────────────

class FZSLPredictor:
    """
    Eğitilmiş FZSLModel ile zero-shot sınıflandırma yapar.

    Kullanım:
        predictor = FZSLPredictor(model, class_text_emb_np, class_order)
        classes, sims = predictor.predict(X_test_np)
    """

    def __init__(
        self,
        model: FZSLModel,
        class_text_embeddings: np.ndarray,   # [num_all_classes, text_dim]
        class_order: list,                   # class isimlerinin sırası
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.class_order = class_order

        # Text embedding'leri cihaza taşı ve model projection'dan geçir
        self.model.eval()
        with torch.no_grad():
            text_t = torch.tensor(class_text_embeddings, dtype=torch.float32).to(device)
            self.class_protos = self.model.encode_texts(text_t).cpu().numpy()
            # shape: [num_classes, proj_dim]

    def predict(self, X: np.ndarray):
        """
        X: [N, input_dim] numpy array

        Dönüş:
            pred_classes : list[str]  uzunluk N
            similarities : np.array [N, num_classes]
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            tx_embs = self.model.encode_transactions(X_tensor).cpu().numpy()
            # [N, proj_dim]

        # Cosine similarity (her ikisi L2-normalize edildi)
        sims = tx_embs @ self.class_protos.T   # [N, num_classes]
        pred_indices = sims.argmax(axis=1)
        pred_classes = [self.class_order[i] for i in pred_indices]

        return pred_classes, sims

    def is_fraud_proba(self, X: np.ndarray, fraud_classes: list = None):
        """
        Her işlem için fraud olasılığı hesaplar.

        fraud_classes: hangi sınıflar fraud sayılıyor (None → 'normal' dışı hepsi)
        Dönüş: np.array [N]  (0-1 arası fraud skoru)
        """
        if fraud_classes is None:
            fraud_classes = [c for c in self.class_order if c != "normal"]

        _, sims = self.predict(X)
        # Softmax → fraud sınıflarına ait softmax olasılıkları topla
        exp_sims = np.exp(sims - sims.max(axis=1, keepdims=True))
        proba = exp_sims / exp_sims.sum(axis=1, keepdims=True)

        fraud_indices = [i for i, c in enumerate(self.class_order) if c in fraud_classes]
        fraud_proba = proba[:, fraud_indices].sum(axis=1)

        return fraud_proba
