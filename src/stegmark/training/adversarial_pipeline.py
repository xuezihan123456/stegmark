from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class WatermarkTrainer:
        """Joint encoder/decoder/discriminator trainer with differentiable noise layers.

        Args:
            encoder: Module mapping (images, messages) -> encoded_images.
            decoder: Module mapping encoded_images -> message_logits.
            discriminator: Module mapping images -> real/fake logits (B, 1).
            noise_pool: NoisePool or any nn.Module applied between encoder and decoder.
            device: torch device string.
        """

        def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            discriminator: nn.Module,
            noise_pool: nn.Module,
            device: str = "cpu",
        ) -> None:
            self.device = torch.device(device)
            self.encoder = encoder.to(self.device)
            self.decoder = decoder.to(self.device)
            self.discriminator = discriminator.to(self.device)
            self.noise_pool = noise_pool.to(self.device)

            enc_dec_params = list(encoder.parameters()) + list(decoder.parameters())
            self.opt_enc_dec = torch.optim.Adam(enc_dec_params, lr=1e-4)
            self.opt_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------

        def _encoder_loss(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(encoded, cover)

        def _decoder_loss(
            self, logits: torch.Tensor, messages: torch.Tensor
        ) -> torch.Tensor:
            return F.binary_cross_entropy_with_logits(logits, messages)

        def _gan_loss_generator(self, fake_logits: torch.Tensor) -> torch.Tensor:
            """Generator wants discriminator to output 1 (real) for encoded images."""
            real_labels = torch.ones_like(fake_logits)
            return F.binary_cross_entropy_with_logits(fake_logits, real_labels)

        def _gan_loss_discriminator(
            self,
            real_logits: torch.Tensor,
            fake_logits: torch.Tensor,
        ) -> torch.Tensor:
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            return 0.5 * (
                F.binary_cross_entropy_with_logits(real_logits, real_labels)
                + F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
            )

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def train_step(
            self,
            images: torch.Tensor,
            messages: torch.Tensor,
        ) -> dict[str, float]:
            """Single joint training step.

            Args:
                images: Cover images (B, 3, H, W) in [0, 1].
                messages: Binary message tensors (B, message_bits) in {0, 1}.

            Returns:
                Dict with encoder_loss, decoder_loss, gan_loss, total.
            """
            images = images.to(self.device)
            messages = messages.to(self.device)

            # --- Train discriminator ---
            self.opt_disc.zero_grad(set_to_none=True)
            with torch.no_grad():
                encoded_detach = self.encoder(images, messages)
            real_logits = self.discriminator(images)
            fake_logits_d = self.discriminator(encoded_detach.detach())
            disc_loss = self._gan_loss_discriminator(real_logits, fake_logits_d)
            disc_loss.backward()
            self.opt_disc.step()

            # --- Train encoder + decoder ---
            self.opt_enc_dec.zero_grad(set_to_none=True)
            encoded = self.encoder(images, messages)
            noised = self.noise_pool(encoded)
            logits = self.decoder(noised)

            enc_loss = self._encoder_loss(encoded, images)
            dec_loss = self._decoder_loss(logits, messages)
            fake_logits_g = self.discriminator(encoded)
            gan_loss = self._gan_loss_generator(fake_logits_g)

            total = enc_loss + 5.0 * dec_loss + 0.5 * gan_loss
            total.backward()
            self.opt_enc_dec.step()

            return {
                "encoder_loss": float(enc_loss.detach().cpu().item()),
                "decoder_loss": float(dec_loss.detach().cpu().item()),
                "gan_loss": float(gan_loss.detach().cpu().item()),
                "total": float(total.detach().cpu().item()),
            }

        @torch.no_grad()
        def validate(
            self,
            images: torch.Tensor,
            messages: torch.Tensor,
        ) -> dict[str, float]:
            """Compute BER and PSNR on a validation batch.

            Returns:
                Dict with ber (bit error rate in [0, 1]) and psnr (dB).
            """
            images = images.to(self.device)
            messages = messages.to(self.device)

            encoded = self.encoder(images, messages)
            logits = self.decoder(encoded)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            ber = float((predicted != messages).float().mean().item())

            mse = F.mse_loss(encoded, images).item()
            psnr = float(10.0 * torch.log10(torch.tensor(1.0 / (mse + 1e-10))).item())

            return {"ber": ber, "psnr": psnr}

    __all__ = ["WatermarkTrainer", "TORCH_AVAILABLE"]
