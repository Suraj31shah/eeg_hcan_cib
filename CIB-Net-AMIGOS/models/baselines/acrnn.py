import torch
import torch.nn as nn


class ACRNN(nn.Module):
    """
    Attention-based Convolutional Recurrent Neural Network (channel attention -> 1D CNN -> BiLSTM).
    """

    def __init__(self, n_channels: int = 14, n_classes: int = 2, **kwargs):
        super().__init__()
        mid = max(2, n_channels // 2)
        self.channel_attention = nn.Sequential(
            nn.Linear(n_channels, mid),
            nn.ELU(),
            nn.Linear(mid, n_channels),
            nn.Sigmoid(),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=16, stride=4, padding=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(
            input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=2)
        attn_weights = self.channel_attention(avg_pool).unsqueeze(-1)
        x = x * attn_weights
        feat = self.cnn(x).permute(0, 2, 1)
        rnn_out, _ = self.rnn(feat)
        rnn_out = torch.mean(rnn_out, dim=1)
        return self.classifier(rnn_out)

