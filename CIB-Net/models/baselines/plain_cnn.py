import torch
import torch.nn as nn

class PlainCNN(nn.Module):
    def __init__(self, n_channels=32, n_classes=2):
        super(PlainCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Input: [B, 32, 8064]
            nn.Conv1d(n_channels, 64, kernel_size=32, stride=4, padding=16),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x is [B, 32, 8064]
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return out
