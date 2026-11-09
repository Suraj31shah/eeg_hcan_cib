import torch
import torch.nn as nn


class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(
            input,
            self.weight.clamp(min=-1.0, max=1.0),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EEGNet(nn.Module):
    """
    Standard EEGNet implementation dynamically sized from `sample_rate * window_sec`.
    Expects `[B, C, T]`.
    """

    def __init__(
        self,
        n_channels: int = 14,
        n_classes: int = 2,
        sample_rate: int = 128,
        window_sec: float = 4.0,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        super().__init__()
        kernel_1 = max(2, int(sample_rate) // 2)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            ConstrainedConv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )

        n_samples = int(round(sample_rate * window_sec))
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy_out = self.block3(self.block2(self.block1(dummy)))
            flatten_size = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(nn.Linear(flatten_size, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

