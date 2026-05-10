import torch
import torch.nn as nn


class MLPFraudDetector(nn.Module):
    """
    MLP tabanli fraud tespit modeli.

    Mimari:
        input_dim → 64 → 32 → 1  (binary classification)

    Katmanlar:
        self.net[:-1] → FL Backbone: 32-dim semantik ozellik vektoru
        self.net[-1]  → Siniflandirma kafasi: 32 → 1 (sigmoid ile prob)

    Not: get_features() metodu FZSL entegrasyonu icin 32-dim temsilleri dondurur.
    """

    FEATURE_DIM = 32   # backbone cikti boyutu (sabit referans)

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tam ileri gecis: logit dondurur (sigmoid uygulanmamis)."""
        return self.net(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        FL Backbone cikti: 32-dim semantik ozellik vektoru.

        Son siniflandirma katmani (Linear 32→1) atlanir.
        FZSL bu temsilleri TransactionEncoder'a girdi olarak kullanir.
        """
        return self.net[:-1](x)