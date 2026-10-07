import torch
import torch.nn as nn

class PeripheralEncoder(nn.Module):
    def __init__(self, in_features=24, d_embed=64):
        """
        Simple 2-layer MLP to encode the 24 peripheral features (mean, std, range of 8 channels)
        into the unified token dimension (d_embed).
        """
        super(PeripheralEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, d_embed),
            nn.BatchNorm1d(d_embed),
            nn.GELU()
        )
        
        # Modality auxiliary head for Spectral Decoupling (z_periph)
        self.aux_head = nn.Linear(d_embed, 2)
        
    def forward(self, periph_features):
        """
        periph_features: [B, 24]
        Returns:
            token: [B, 1, d_embed] (unsqueeze for cross-attention)
            z_periph: [B, 2] (auxiliary logits)
        """
        x = self.mlp(periph_features)
        z_periph = self.aux_head(x)
        
        return x.unsqueeze(1), z_periph
