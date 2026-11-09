import torch
import torch.nn as nn


class PerChannelCNNEncoder(nn.Module):
    def __init__(self, d_embed: int = 64, shared_weights: bool = True, n_channels: int = 14):
        super().__init__()
        self.shared_weights = shared_weights
        self.n_channels = n_channels
        self.d_embed = d_embed

        if shared_weights:
            self.encoder = self._build_encoder()
        else:
            self.encoders = nn.ModuleList([self._build_encoder() for _ in range(n_channels)])

    def _build_encoder(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Conv1d(64, self.d_embed, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.d_embed),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: `[B, n_channels, T]` -> returns `[B, n_channels, d_embed]`.
        """
        B, C, T = x.size()
        if self.shared_weights:
            x_reshaped = x.reshape(B * C, 1, T)
            out = self.encoder(x_reshaped)
            return out.view(B, C, self.d_embed)

        embeddings = [self.encoders[i](x[:, i, :].unsqueeze(1)).squeeze(-1) for i in range(self.n_channels)]
        return torch.stack(embeddings, dim=1)


class StageAModel(nn.Module):
    """EEG-only Stage A encoder + linear classifier for channel probe diagnostics."""

    def __init__(self, n_channels: int = 14, n_classes: int = 2, d_embed: int = 64, shared_weights: bool = True):
        super().__init__()
        self.encoder = PerChannelCNNEncoder(d_embed=d_embed, shared_weights=shared_weights, n_channels=n_channels)
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * d_embed, 64),
            nn.ELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        features = embeddings.view(embeddings.size(0), -1)
        return self.classifier(features)

