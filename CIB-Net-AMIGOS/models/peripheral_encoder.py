import torch
import torch.nn as nn


class PeripheralEncoder(nn.Module):
    """
    2-layer MLP encoding summary statistics from AMIGOS peripheral channels
    (default 18 features = 6 stats x 3 channels [ECG_R, ECG_L, GSR]).
    """

    def __init__(self, in_features: int = 18, d_embed: int = 64, n_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, d_embed),
            nn.BatchNorm1d(d_embed),
            nn.GELU(),
        )
        self.aux_head = nn.Linear(d_embed, n_classes)

    def forward(self, periph_features: torch.Tensor):
        x = self.mlp(periph_features)
        z_periph = self.aux_head(x)
        return x.unsqueeze(1), z_periph

