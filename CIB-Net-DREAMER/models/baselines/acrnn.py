import torch
import torch.nn as nn
import torch.nn.functional as F

class ACRNN(nn.Module):
    """
    Simplified ACRNN (Attention-based Convolutional Recurrent Neural Network)
    Implements Channel-wise attention -> CNN -> RNN.
    """
    def __init__(self, n_channels=32, n_classes=2):
        super(ACRNN, self).__init__()
        
        # Channel-wise attention (Spatial attention)
        self.channel_attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2),
            nn.ELU(),
            nn.Linear(n_channels // 2, n_channels),
            nn.Sigmoid()
        )
        
        # Spatial-temporal feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2)
        )
        
        # RNN for temporal dynamics
        self.rnn = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, n_classes)
        )

    def forward(self, x):
        # x is [B, 32, 8064]
        B, C, T = x.size()
        
        # Compute channel attention over the mean signal
        avg_pool = torch.mean(x, dim=2) # [B, C]
        attn_weights = self.channel_attention(avg_pool).unsqueeze(-1) # [B, C, 1]
        
        # Apply attention
        x = x * attn_weights
        
        # CNN
        feat = self.cnn(x) # [B, 128, T']
        
        # Prepare for RNN: [B, T', Features]
        feat = feat.permute(0, 2, 1) 
        
        # RNN
        rnn_out, _ = self.rnn(feat) # [B, T', 128]
        
        # Global Avg Pool over time
        rnn_out = torch.mean(rnn_out, dim=1) # [B, 128]
        
        out = self.classifier(rnn_out)
        return out
