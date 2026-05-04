from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Adam

from stegmark.nn.hidden.decoder import HiddenDecoder
from stegmark.nn.hidden.encoder import HiddenEncoder
from stegmark.training.losses import hidden_total_loss


@dataclass(frozen=True)
class HiddenTrainerConfig:
    message_bits: int
    image_size: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-3
    image_weight: float = 1.0
    message_weight: float = 5.0
    device: str = "cpu"


class HiddenTrainer:
    def __init__(self, config: HiddenTrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.encoder = HiddenEncoder(message_bits=config.message_bits).to(self.device)
        self.decoder = HiddenDecoder(message_bits=config.message_bits).to(self.device)
        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.learning_rate,
        )

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        image = batch["image"].to(self.device)
        message = batch["message"].to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        encoded = self.encoder(image, message)
        logits = self.decoder(encoded)
        encoder_loss, decoder_loss, total_loss = hidden_total_loss(
            encoded,
            image,
            logits,
            message,
            image_weight=self.config.image_weight,
            message_weight=self.config.message_weight,
        )
        total_loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        return {
            "encoder_loss": float(encoder_loss.detach().cpu().item()),
            "decoder_loss": float(decoder_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }
