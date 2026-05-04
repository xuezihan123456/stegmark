from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def hidden_image_loss(encoded: Tensor, cover: Tensor) -> Tensor:
    return F.mse_loss(encoded, cover)


def hidden_message_loss(logits: Tensor, target_bits: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, target_bits)


def hidden_total_loss(
    encoded: Tensor,
    cover: Tensor,
    logits: Tensor,
    target_bits: Tensor,
    *,
    image_weight: float = 1.0,
    message_weight: float = 5.0,
) -> tuple[Tensor, Tensor, Tensor]:
    image_component = hidden_image_loss(encoded, cover)
    message_component = hidden_message_loss(logits, target_bits)
    total = (image_weight * image_component) + (message_weight * message_component)
    return image_component, message_component, total
