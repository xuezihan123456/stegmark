from __future__ import annotations

import torch
from torch import Tensor, nn

from stegmark.nn.hidden.layers import conv_block


class HiddenEncoder(nn.Module):
    def __init__(self, message_bits: int, channels: int = 64) -> None:
        super().__init__()
        self.message_bits = message_bits
        self.features = nn.Sequential(
            conv_block(3, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
        )
        self.fusion = nn.Sequential(
            conv_block(channels + 3 + message_bits, channels),
            nn.Conv2d(channels, 3, kernel_size=1),
        )

    def forward(self, image: Tensor, message: Tensor) -> Tensor:
        batch_size, _, height, width = image.shape
        features = self.features(image)
        message_map = message.view(batch_size, self.message_bits, 1, 1).expand(
            batch_size,
            self.message_bits,
            height,
            width,
        )
        fused = torch.cat([features, image, message_map], dim=1)
        residual = self.fusion(fused)
        return torch.clamp(image + residual, 0.0, 1.0)
