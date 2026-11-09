import torch
import torch.nn as nn

class PerChannelCNNEncoder(nn.Module):
    """
    Stage A: Per-Channel Temporal 1D-CNN Encoder.
    Processes all 64 EEG channels independently using shared temporal weights.
    Eliminates Pezeshki spatial gradient starvation (no depthwise spatial squashing).
    
    Input: [B, 64, T] (where T = 640 for 4-second window @ 160 Hz)
    Output: [B, 64, d_embed] (channel embeddings, e.g. d_embed = 64)
    """
    def __init__(self, n_channels=64, d_embed=64, shared_weights=True):
        super(PerChannelCNNEncoder, self).__init__()
        self.n_channels = n_channels
        self.d_embed = d_embed
        self.shared_weights = shared_weights
        
        def make_branch():
            return nn.Sequential(
                # Block 1: Broad temporal features (slow rhythms, mu/alpha)
                nn.Conv1d(1, 16, kernel_size=25, stride=2, padding=12),
                nn.BatchNorm1d(16),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                
                # Block 2: Intermediate temporal features (beta rhythm)
                nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                
                # Block 3: High-frequency transient features
                nn.Conv1d(32, d_embed, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(d_embed),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(1) # Pool across remaining time dimension
            )
            
        if self.shared_weights:
            self.temporal_encoder = make_branch()
        else:
            self.temporal_encoders = nn.ModuleList([make_branch() for _ in range(n_channels)])
            
    def forward(self, x):
        """
        x: [B, n_channels, T]
        returns: [B, n_channels, d_embed]
        """
        B, C, T = x.shape
        if self.shared_weights:
            # Reshape into [B * C, 1, T] for batch 1D convolution
            x_reshaped = x.view(B * C, 1, T)
            feat = self.temporal_encoder(x_reshaped) # [B * C, d_embed, 1]
            out = feat.view(B, C, self.d_embed)
        else:
            feats = []
            for c in range(C):
                chan_x = x[:, c:c+1, :] # [B, 1, T]
                f = self.temporal_encoders[c](chan_x).squeeze(-1) # [B, d_embed]
                feats.append(f)
            out = torch.stack(feats, dim=1) # [B, C, d_embed]
            
        return out

