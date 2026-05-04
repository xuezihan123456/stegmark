from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stegmark.core.codec import decode_bitstream, resolve_payload_bits
from stegmark.core.engine import EngineCapabilities, WatermarkEngine
from stegmark.exceptions import MessageTooLongError
from stegmark.types import ExtractResult, ImageArray


class ScreenResilientEngine(WatermarkEngine):
    """抗截图水印引擎 —— 傅里叶-梅林变换域嵌入。

    在 FFT 幅度谱的对数极坐标中频环带用 QIM 嵌入水印。
    对旋转和缩放攻击具有天然免疫能力。
    """

    name = "screen_resilient"
    declared_capabilities = EngineCapabilities(supports_strength_control=True)

    # 对数极坐标环带参数
    R_INNER: int = 30  # 中频内半径
    R_OUTER: int = 60  # 中频外半径
    ANGULAR_SAMPLES: int = 256  # 角度采样点
    QIM_STEP: float = 10.0  # 基础量化步长

    def encode(
        self,
        image: ImageArray,
        message: str | None = None,
        *,
        payload_bits: Sequence[int] | None = None,
        strength: float = 1.0,
    ) -> ImageArray:
        bits = resolve_payload_bits(message, payload_bits)
        h, w = image.shape[:2]
        min_dim = min(h, w)

        # 图像需要至少 128×128 以支持 log-polar
        if min_dim < 64:
            raise MessageTooLongError(
                "image too small for screen-resilient watermark",
                hint="Use an image of at least 64×64 pixels.",
            )

        # RGB → Y 通道
        y = (0.299 * image[:, :, 0].astype(np.float32) +
             0.587 * image[:, :, 1].astype(np.float32) +
             0.114 * image[:, :, 2].astype(np.float32))

        # 2D FFT
        fft = np.fft.fft2(y)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        fft_shifted = np.fft.fftshift(magnitude)

        # 构建对数极坐标采样 —— 在中频环带嵌入
        center_h, center_w = h // 2, w // 2
        capacity = self.ANGULAR_SAMPLES * (self.R_OUTER - self.R_INNER)
        if len(bits) > capacity:
            raise MessageTooLongError(
                f"message too long for screen-resilient engine ({len(bits)} > {capacity})",
                hint="Use a shorter message or a larger image.",
            )

        # QIM 嵌入：在中频环带修改幅度
        qim_step = self.QIM_STEP * strength
        bit_idx = 0
        modified_magnitude = fft_shifted.copy()
        for r in range(self.R_INNER, self.R_OUTER):
            if bit_idx >= len(bits):
                break
            for angle in range(0, self.ANGULAR_SAMPLES, max(1, self.ANGULAR_SAMPLES // 64)):
                if bit_idx >= len(bits):
                    break
                # 极坐标 → 直角坐标
                theta = 2 * np.pi * angle / self.ANGULAR_SAMPLES
                row = int(center_h + r * np.sin(theta))
                col = int(center_w + r * np.cos(theta))
                if 0 <= row < h and 0 <= col < w:
                    val = modified_magnitude[row, col]
                    q = np.round(val / qim_step)
                    if int(bits[bit_idx]) == 1:
                        q = q + 1 if q % 2 == 0 else q
                    else:
                        q = q if q % 2 == 0 else q + 1
                    modified_magnitude[row, col] = q * qim_step
                    bit_idx += 1

        # 逆 FFT
        modified_fft = np.fft.ifftshift(modified_magnitude) * np.exp(1j * phase)
        y_watermarked = np.real(np.fft.ifft2(modified_fft))

        # Y 差值回加到 RGB
        delta = y_watermarked - y
        result = image.astype(np.float32) + delta[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    def decode(self, image: ImageArray) -> ExtractResult:
        h, w = image.shape[:2]

        y = (0.299 * image[:, :, 0].astype(np.float32) +
             0.587 * image[:, :, 1].astype(np.float32) +
             0.114 * image[:, :, 2].astype(np.float32))

        fft = np.fft.fft2(y)
        magnitude = np.abs(fft)
        fft_shifted = np.fft.fftshift(magnitude)

        center_h, center_w = h // 2, w // 2
        qim_step = self.QIM_STEP

        extracted_bits: list[int] = []
        for r in range(self.R_INNER, self.R_OUTER):
            for angle in range(0, self.ANGULAR_SAMPLES, max(1, self.ANGULAR_SAMPLES // 64)):
                if len(extracted_bits) >= 8192:
                    break
                theta = 2 * np.pi * angle / self.ANGULAR_SAMPLES
                row = int(center_h + r * np.sin(theta))
                col = int(center_w + r * np.cos(theta))
                if 0 <= row < h and 0 <= col < w:
                    val = fft_shifted[row, col]
                    q = np.round(val / qim_step)
                    bit = int(q) % 2
                    extracted_bits.append(bit)

        if not extracted_bits:
            return ExtractResult(
                found=False, engine=self.name, bits=(), confidence=0.0, error="no_bits",
            )

        # 在各偏移处尝试解码帧
        for offset in range(0, min(64, len(extracted_bits) - 128)):
            trial = extracted_bits[offset:]
            decoded = decode_bitstream(trial)
            if decoded.valid:
                return ExtractResult(
                    found=True,
                    engine=self.name,
                    bits=decoded.bits,
                    payload=decoded.payload,
                    message=decoded.message,
                    confidence=1.0,
                )

        return ExtractResult(
            found=False,
            engine=self.name,
            bits=tuple(extracted_bits[:256]),
            confidence=0.0,
            error="decode_failed",
        )


__all__ = ["ScreenResilientEngine"]
