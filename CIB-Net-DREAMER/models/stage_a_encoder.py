import torch
import torch.nn as nn

class PerChannelCNNEncoder(nn.Module):
    def __init__(self, d_embed=64, shared_weights=True, n_channels=32):
        super(PerChannelCNNEncoder, self).__init__()
        
        self.shared_weights = shared_weights
        self.n_channels = n_channels
        self.d_embed = d_embed
        
        if shared_weights:
            self.encoder = self._build_encoder()
        else:
            self.encoders = nn.ModuleList([self._build_encoder() for _ in range(n_channels)])

    def _build_encoder(self):
        return nn.Sequential(
            # Input: [B, 1, T]
            nn.Conv1d(1, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            
            nn.AdaptiveAvgPool1d(64), # Downsample sequence to length 64
            
            nn.Conv1d(64, self.d_embed, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.d_embed),
            nn.ELU(),
            
            nn.AdaptiveAvgPool1d(1) # Global Average Pool -> [B, d_embed, 1]
        )

    def forward(self, x):
        """
        x: [B, N_channels, T]
        Returns: [B, N_channels, D_embed]
        """
        B, C, T = x.size()
        embeddings = []
        
        if self.shared_weights:
            # Reshape to treat each channel as a batch element for efficiency
            # x_reshaped: [B*C, 1, T]
            x_reshaped = x.view(B * C, 1, T)
            out = self.encoder(x_reshaped) # [B*C, D, 1]
            out = out.view(B, C, self.d_embed)
            return out
        else:
            for i in range(self.n_channels):
                # Extract channel i: [B, 1, T]
                xi = x[:, i, :].unsqueeze(1)
                ei = self.encoders[i](xi) # [B, D, 1]
                embeddings.append(ei.squeeze(-1))
                
            # Stack along channel dimension
            out = torch.stack(embeddings, dim=1) # [B, C, D]
            return out

class StageAModel(nn.Module):
    """
    Simple model using Stage A encoder + classifier to establish an encoder baseline
    and for running single-channel diagnostic probes.
    """
    def __init__(self, n_channels=32, n_classes=2, d_embed=64, shared_weights=True):
        super(StageAModel, self).__init__()
        self.encoder = PerChannelCNNEncoder(d_embed, shared_weights, n_channels)
        # Simple classification from concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * d_embed, 64),
            nn.ELU(),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        # x: [B, C, T]
        embeddings = self.encoder(x) # [B, C, D]
        # Flatten embeddings
        features = embeddings.view(embeddings.size(0), -1)
        out = self.classifier(features)
        return out
