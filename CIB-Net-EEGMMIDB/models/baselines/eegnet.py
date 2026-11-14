import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class EEGNet(nn.Module):
    """
    Standard EEGNet implementation adapted for 64-channel EEGMMIDB.
    Expects input shape: [B, 64, 640] -> reshaped internally to [B, 1, 64, 640]
    """
    def __init__(self, n_channels=64, n_classes=2, sample_rate=160, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        kernel_1 = sample_rate // 2 # 80 points (0.5s)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.block2 = nn.Sequential(
            # Depthwise conv across all 64 channels
            ConstrainedConv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        
        self.block3 = nn.Sequential(
            # Separable conv
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        
        # Adaptive pooling to ensure fixed size regardless of temporal window length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 16))
        
        self.classifier = nn.Sequential(
            nn.Linear(F2 * 16, n_classes)
        )

    def forward(self, x, return_feat=False):
        # x is [B, 64, T] -> reshape to [B, 1, 64, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        
        feat = x.view(x.size(0), -1)
        out = self.classifier(feat)
        if return_feat:
            return out, feat
        return out

