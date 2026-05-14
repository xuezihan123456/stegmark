from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class DifferentiableJPEG(nn.Module):
        """Differentiable JPEG approximation using sigmoid quantization.

        Input: (B, 3, H, W) in [0, 1].
        quality: compression quality in [10, 95].
        """

        def __init__(self, quality: float = 75.0) -> None:
            super().__init__()
            self.quality = float(quality)

        def _quantize_approx(self, x: torch.Tensor, step: float) -> torch.Tensor:
            """Soft quantization: round via sigmoid approximation."""
            # Straight-through estimator style: forward=round, backward=identity
            rounded = torch.round(x / step) * step
            # Blend with smooth approximation for gradient flow
            smooth = x - step * (torch.sigmoid((x / step - torch.floor(x / step)) * 12.0) - 0.5)
            return rounded + (smooth - smooth.detach())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Map quality to quantization step: lower quality -> larger step
            step = (101.0 - self.quality) / 100.0 * 0.2 + 0.01
            x_q = self._quantize_approx(x, step)
            return x_q.clamp(0.0, 1.0)

    class GaussianNoise(nn.Module):
        """Add Gaussian noise with given sigma."""

        def __init__(self, sigma: float = 0.05) -> None:
            super().__init__()
            self.sigma = sigma

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.sigma == 0.0:
                return x
            noise = torch.randn_like(x) * self.sigma
            return (x + noise).clamp(0.0, 1.0)

    class PrintScanSimulator(nn.Module):
        """Simulate print-scan pipeline: color shift + blur + noise + JPEG."""

        def __init__(
            self,
            jpeg_quality: float = 70.0,
            noise_sigma: float = 0.02,
            blur_kernel: int = 3,
        ) -> None:
            super().__init__()
            self.jpeg = DifferentiableJPEG(quality=jpeg_quality)
            self.noise = GaussianNoise(sigma=noise_sigma)
            self.blur_kernel = blur_kernel
            # Learnable color correction matrix (3x3), initialized to identity
            self.color_matrix = nn.Parameter(torch.eye(3))

        def _apply_color_matrix(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 3, H, W) -> reshape to (B, H*W, 3) for matmul
            b, c, h, w = x.shape
            flat = x.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)
            shifted = flat @ self.color_matrix.T
            return shifted.reshape(b, h, w, 3).permute(0, 3, 1, 2).clamp(0.0, 1.0)

        def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
            k = self.blur_kernel
            if k <= 1:
                return x
            pad = k // 2
            # Simple box blur approximating Gaussian
            kernel = torch.ones(3, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
            return F.conv2d(x, kernel, padding=pad, groups=3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._apply_color_matrix(x)
            x = self._gaussian_blur(x)
            x = self.noise(x)
            x = self.jpeg(x)
            return x

    class ScreenShootSimulator(nn.Module):
        """Simulate screen-shot distortion: Moire pattern + blur."""

        def __init__(self, moire_amplitude: float = 0.03, blur_kernel: int = 3) -> None:
            super().__init__()
            self.moire_amplitude = moire_amplitude
            self.blur_kernel = blur_kernel

        def _moire_mask(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            # Sinusoidal mask in spatial domain
            row = torch.arange(h, dtype=x.dtype, device=x.device)
            col = torch.arange(w, dtype=x.dtype, device=x.device)
            mask_h = torch.sin(2 * 3.14159265 * row / 8.0).view(1, 1, h, 1)
            mask_w = torch.sin(2 * 3.14159265 * col / 8.0).view(1, 1, 1, w)
            moire = self.moire_amplitude * mask_h * mask_w
            return (x + moire).clamp(0.0, 1.0)

        def _blur(self, x: torch.Tensor) -> torch.Tensor:
            k = self.blur_kernel
            if k <= 1:
                return x
            pad = k // 2
            kernel = torch.ones(3, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
            return F.conv2d(x, kernel, padding=pad, groups=3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._moire_mask(x)
            x = self._blur(x)
            return x

    class CropFlipRotate(nn.Module):
        """Random geometric transforms: crop + horizontal flip + 90-degree rotation."""

        def __init__(
            self,
            crop_ratio: float = 0.9,
            flip_prob: float = 0.5,
            rotate_prob: float = 0.5,
        ) -> None:
            super().__init__()
            self.crop_ratio = crop_ratio
            self.flip_prob = flip_prob
            self.rotate_prob = rotate_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            # Random crop via affine grid
            if self.crop_ratio < 1.0:
                scale = self.crop_ratio + torch.rand(1).item() * (1.0 - self.crop_ratio)
                theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
                theta[:, 0, 0] = scale
                theta[:, 1, 1] = scale
                grid = F.affine_grid(theta, x.size(), align_corners=False)
                x = F.grid_sample(x, grid, align_corners=False, mode="bilinear", padding_mode="reflection")
            # Random horizontal flip
            if torch.rand(1).item() < self.flip_prob:
                x = torch.flip(x, dims=[3])
            # Random 90-degree rotation
            if torch.rand(1).item() < self.rotate_prob:
                k = int(torch.randint(1, 4, (1,)).item())
                x = torch.rot90(x, k, dims=[2, 3])
            return x

    class NoisePool(nn.Module):
        """Randomly apply a subset of noise layers each forward pass."""

        def __init__(
            self,
            layers: list[nn.Module] | None = None,
            num_active: int = 2,
        ) -> None:
            super().__init__()
            if layers is None:
                layers = [
                    GaussianNoise(sigma=0.03),
                    DifferentiableJPEG(quality=75.0),
                    CropFlipRotate(crop_ratio=0.95),
                    ScreenShootSimulator(moire_amplitude=0.02),
                ]
            self.layers = nn.ModuleList(layers)
            self.num_active = min(num_active, len(self.layers))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            indices = torch.randperm(len(self.layers))[: self.num_active].tolist()
            for idx in indices:
                x = self.layers[idx](x)
            return x

    __all__ = [
        "DifferentiableJPEG",
        "GaussianNoise",
        "PrintScanSimulator",
        "ScreenShootSimulator",
        "CropFlipRotate",
        "NoisePool",
        "TORCH_AVAILABLE",
    ]
