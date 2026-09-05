"""
A small CNN classifier for ELA maps.

Deliberately not a big pretrained backbone (ResNet/EfficientNet): ELA
residuals are single-channel, low-level statistical texture maps, not
natural-image content, so ImageNet features buy you very little here --
and a from-scratch backbone would badly overfit 360 samples anyway. This
architecture is sized to the dataset: a handful of conv blocks that can
learn "is there a spatially localized, texture-inconsistent patch" without
enough capacity to just memorize training images.

Swap this for a pretrained backbone once you're training on real CASIA v2
(tens of thousands of images) -- the roadmap's README notes that upgrade
path explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ELACNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 16 -> 4, fixed regardless of input size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # single logit; sigmoid applied at inference/loss time
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)  # (B,) raw logits


if __name__ == "__main__":
    model = ELACNN()
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print("output shape:", out.shape, " params:", f"{n_params:,}")
