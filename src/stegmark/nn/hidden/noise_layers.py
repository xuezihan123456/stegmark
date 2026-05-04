from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class IdentityNoise(nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        return image


class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.01) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, image: Tensor) -> Tensor:
        noise = torch.randn_like(image) * self.sigma
        return torch.clamp(image + noise, 0.0, 1.0)


class PixelDropout(nn.Module):
    def __init__(self, probability: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=probability)

    def forward(self, image: Tensor) -> Tensor:
        return cast(Tensor, self.dropout(image))
