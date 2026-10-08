import torch
import torch.nn as nn

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

class EEGNet(nn.Module):
    """
    Standard EEGNet implementation.
    Expects input shape: [B, C, T] -> Reshaped to [B, 1, C, T] internally
    """
    def __init__(self, n_channels=32, n_classes=2, sample_rate=128, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        kernel_1 = sample_rate // 2
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.block2 = nn.Sequential(
            # Depthwise conv
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
        
        # Calculate output size dynamically
        # Since T could vary depending on windowing strategy, we do a dummy pass
        dummy_x = torch.zeros(1, 1, n_channels, sample_rate * 4) # Assume 4s window 
        # But wait, T is passed implicitly. We can just use AdaptiveAvgPool2d to force a fixed size!
        # Or calculate it for 512 samples. Let's use 512 since window_sec=4.0 and sample_rate=128
        
        dummy_x = torch.zeros(1, 1, n_channels, 512)
        dummy_out = self.block3(self.block2(self.block1(dummy_x)))
        flatten_size = dummy_out.view(1, -1).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, n_classes)
        )

    def forward(self, x):
        # x is [B, 32, 8064] -> reshape to [B, 1, 32, 8064]
        x = x.unsqueeze(1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
