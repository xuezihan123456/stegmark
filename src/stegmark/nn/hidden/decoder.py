from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from stegmark.nn.hidden.layers import conv_block


class HiddenDecoder(nn.Module):
    def __init__(self, message_bits: int, channels: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            conv_block(3, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, message_bits)

    def forward(self, image: Tensor) -> Tensor:
        features = self.backbone(image)
        pooled = self.pool(features).flatten(1)
        return cast(Tensor, self.head(pooled))
