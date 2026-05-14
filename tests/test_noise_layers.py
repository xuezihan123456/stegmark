from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

skip_no_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


# ---------------------------------------------------------------------------
# Tiny dummy encoder / decoder for WatermarkTrainer tests
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class _TinyEncoder(nn.Module):
        """(images, messages) -> encoded images, same spatial shape."""

        def __init__(self, msg_bits: int = 8) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3 + msg_bits, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        def forward(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
            b, c, h, w = images.shape
            msg_map = messages.view(b, -1, 1, 1).expand(b, -1, h, w)
            x = torch.cat([images, msg_map], dim=1)
            x = torch.relu(self.conv1(x))
            return torch.sigmoid(self.conv2(x))

    class _TinyDecoder(nn.Module):
        """encoded_images -> message logits (B, msg_bits)."""

        def __init__(self, msg_bits: int = 8) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, msg_bits)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.conv1(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    class _TinyDiscriminator(nn.Module):
        """images -> real/fake logit (B, 1)."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)


# ---------------------------------------------------------------------------
# Test 1: All noise layer forward passes preserve spatial shape
# ---------------------------------------------------------------------------


@skip_no_torch
def test_noise_layers_preserve_shape() -> None:
    from stegmark.training.noise_layers import (
        CropFlipRotate,
        DifferentiableJPEG,
        GaussianNoise,
        PrintScanSimulator,
        ScreenShootSimulator,
    )

    x = torch.rand(2, 3, 32, 32)
    layers = [
        GaussianNoise(sigma=0.05),
        DifferentiableJPEG(quality=75.0),
        PrintScanSimulator(jpeg_quality=70.0, noise_sigma=0.02, blur_kernel=3),
        ScreenShootSimulator(moire_amplitude=0.03, blur_kernel=3),
        CropFlipRotate(crop_ratio=0.9, flip_prob=0.5, rotate_prob=0.5),
    ]
    for layer in layers:
        out = layer(x)
        assert out.shape == x.shape, f"{layer.__class__.__name__} changed shape"


# ---------------------------------------------------------------------------
# Test 2: GaussianNoise with sigma=0 is identity
# ---------------------------------------------------------------------------


@skip_no_torch
def test_gaussian_noise_sigma_zero_is_identity() -> None:
    from stegmark.training.noise_layers import GaussianNoise

    layer = GaussianNoise(sigma=0.0)
    x = torch.rand(1, 3, 16, 16)
    out = layer(x)
    assert torch.allclose(out, x), "sigma=0 GaussianNoise should be identity"


# ---------------------------------------------------------------------------
# Test 3: DifferentiableJPEG output is in [0, 1]
# ---------------------------------------------------------------------------


@skip_no_torch
def test_differentiable_jpeg_output_range() -> None:
    from stegmark.training.noise_layers import DifferentiableJPEG

    for quality in [10.0, 50.0, 95.0]:
        layer = DifferentiableJPEG(quality=quality)
        x = torch.rand(2, 3, 32, 32)
        out = layer(x)
        assert out.min() >= 0.0, f"quality={quality}: output below 0"
        assert out.max() <= 1.0, f"quality={quality}: output above 1"


# ---------------------------------------------------------------------------
# Test 4: Gradients flow back through noise layers
# ---------------------------------------------------------------------------


@skip_no_torch
def test_gradient_backprop_through_noise_layers() -> None:
    from stegmark.training.noise_layers import (
        DifferentiableJPEG,
        GaussianNoise,
        PrintScanSimulator,
        ScreenShootSimulator,
    )

    layers = [
        GaussianNoise(sigma=0.05),
        DifferentiableJPEG(quality=75.0),
        PrintScanSimulator(),
        ScreenShootSimulator(),
    ]
    for layer in layers:
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, f"{layer.__class__.__name__}: grad is None"
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# Test 5: NoisePool runs without error
# ---------------------------------------------------------------------------


@skip_no_torch
def test_noise_pool_runs() -> None:
    from stegmark.training.noise_layers import GaussianNoise, NoisePool

    pool = NoisePool(
        layers=[GaussianNoise(sigma=0.01), GaussianNoise(sigma=0.02), GaussianNoise(sigma=0.03)],
        num_active=2,
    )
    x = torch.rand(2, 3, 32, 32)
    out = pool(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Test 6: WatermarkTrainer.train_step returns dict with required keys
# ---------------------------------------------------------------------------


@skip_no_torch
def test_watermark_trainer_train_step_keys() -> None:
    from stegmark.training.adversarial_pipeline import WatermarkTrainer
    from stegmark.training.noise_layers import GaussianNoise

    msg_bits = 8
    enc = _TinyEncoder(msg_bits)
    dec = _TinyDecoder(msg_bits)
    disc = _TinyDiscriminator()
    noise = GaussianNoise(sigma=0.02)

    trainer = WatermarkTrainer(enc, dec, disc, noise, device="cpu")
    images = torch.rand(2, 3, 32, 32)
    messages = torch.randint(0, 2, (2, msg_bits), dtype=torch.float32)

    result = trainer.train_step(images, messages)

    required_keys = {"encoder_loss", "decoder_loss", "gan_loss", "total"}
    assert required_keys == set(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"
    assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# Test 7: WatermarkTrainer.validate returns BER in [0, 1]
# ---------------------------------------------------------------------------


@skip_no_torch
def test_watermark_trainer_validate_ber_range() -> None:
    from stegmark.training.adversarial_pipeline import WatermarkTrainer
    from stegmark.training.noise_layers import GaussianNoise

    msg_bits = 8
    enc = _TinyEncoder(msg_bits)
    dec = _TinyDecoder(msg_bits)
    disc = _TinyDiscriminator()
    noise = GaussianNoise(sigma=0.0)

    trainer = WatermarkTrainer(enc, dec, disc, noise, device="cpu")
    images = torch.rand(2, 3, 32, 32)
    messages = torch.randint(0, 2, (2, msg_bits), dtype=torch.float32)

    metrics = trainer.validate(images, messages)

    assert "ber" in metrics, "validate must return 'ber'"
    assert "psnr" in metrics, "validate must return 'psnr'"
    assert 0.0 <= metrics["ber"] <= 1.0, f"BER out of range: {metrics['ber']}"
