from __future__ import annotations

import numpy as np

from stegmark.core.codec import decode_bitstream, resolve_payload_bits
from stegmark.core.engine import WatermarkEngine
from stegmark.exceptions import InvalidInputError
from stegmark.types import ExtractResult, ImageArray


class ReversibleEngine(WatermarkEngine):
    """可逆水印引擎 —— 基于 LSB 嵌入的可恢复水印。

    使用红色通道 LSB 进行嵌入，存储原始 LSB 以支持完全恢复。
    元数据格式：[4B num_bits][orig_lsb_bytes] 存储在底部蓝色通道 LSB 中。
    """

    name = "reversible"

    def encode(
        self,
        image: ImageArray,
        message: str | None = None,
        *,
        payload_bits: list[int] | None = None,
        strength: float = 1.0,
    ) -> ImageArray:
        del strength
        bits = list(resolve_payload_bits(message, payload_bits))
        if len(bits) < 24:
            raise InvalidInputError(
                "reversible engine requires at least 24 bits of payload",
                hint="Use a longer message.",
            )

        h, w = image.shape[:2]
        num_pixels = h * w
        if len(bits) > num_pixels:
            raise InvalidInputError(
                f"message too long for reversible engine ({len(bits)} bits > {num_pixels} pixels)",
                hint="Use a larger image or shorter message.",
            )

        result = image.copy()

        # 保存原始红色通道 LSB（用于无损恢复）
        red_channel = result[:, :, 0]
        orig_red_lsb = (red_channel & 1).ravel()

        # LSB 嵌入到红色通道
        flat_red = red_channel.ravel()
        for i in range(len(bits)):
            flat_red[i] = (flat_red[i] & 0xFE) | int(bits[i])
        result[:, :, 0] = flat_red.reshape(h, w)

        # 元数据存储在蓝色通道末尾 LSB：[4B num_bits][orig_red_lsb_bytes][4B meta_pixel_count]
        orig_lsb_bytes = np.packbits(orig_red_lsb).tobytes()
        # 计算元数据占用的像素数
        meta_byte_len = 4 + len(orig_lsb_bytes) + 4  # header + lsb + footer
        meta_pixel_count = meta_byte_len * 8  # 每字节 8 bits
        meta = len(bits).to_bytes(4, "big") + orig_lsb_bytes + meta_pixel_count.to_bytes(4, "big")
        meta_bits: list[int] = []
        for byte_val in meta:
            for shift in range(7, -1, -1):
                meta_bits.append((byte_val >> shift) & 1)

        blue_flat = result[:, :, 2].ravel()
        for i, bit in enumerate(meta_bits):
            idx = num_pixels - 1 - i
            if idx < 0:
                break
            blue_flat[idx] = (blue_flat[idx] & 0xFE) | bit
        result[:, :, 2] = blue_flat.reshape(h, w)

        return result

    def _read_metadata(self, image: ImageArray) -> tuple[int, np.ndarray | None]:
        """读取元数据，返回 (num_bits, orig_lsb_or_None)。"""
        h, w = image.shape[:2]
        total_pixels = h * w

        # 从蓝色通道末尾读取元数据（写入时从末尾向前写，先写 num_bits）
        blue_flat = image[:, :, 2].ravel()

        # 读取 num_bits (4 bytes = 32 bits)，从 total-1 向前读
        num_bits_bits: list[int] = []
        for i in range(32):
            idx = total_pixels - 1 - i
            if idx < 0:
                return 0, None
            num_bits_bits.append(int(blue_flat[idx]) & 1)
        # bits[0] 在 total-1（MSB），bits[31] 在 total-32（LSB）
        num_bits = 0
        for b in num_bits_bits:
            num_bits = (num_bits << 1) | b

        if num_bits <= 0 or num_bits > total_pixels:
            return 0, None

        # 读取 orig_lsb (packed bytes)
        orig_lsb_byte_count = (num_bits + 7) // 8
        lsb_bit_count = orig_lsb_byte_count * 8
        lsb_bits: list[int] = []
        for i in range(lsb_bit_count):
            idx = total_pixels - 32 - 1 - i
            if idx < 0:
                return num_bits, None
            lsb_bits.append(int(blue_flat[idx]) & 1)

        # Pack bits to bytes
        lsb_bytes = bytearray()
        for i in range(0, len(lsb_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(lsb_bits):
                    byte = (byte << 1) | lsb_bits[i + j]
                else:
                    byte <<= 1
            lsb_bytes.append(byte)

        orig_lsb = np.unpackbits(np.array(lsb_bytes, dtype=np.uint8))[:num_bits]
        return num_bits, orig_lsb.astype(np.uint8)

    def decode(self, image: ImageArray) -> ExtractResult:
        num_bits, _ = self._read_metadata(image)

        red_flat = image[:, :, 0].ravel()
        max_bits = num_bits if num_bits > 0 else min(len(red_flat), 8192)

        extracted_bits: list[int] = []
        for i in range(min(max_bits, len(red_flat))):
            extracted_bits.append(int(red_flat[i]) & 1)

        decoded = decode_bitstream(extracted_bits[:1024])
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

    def restore(self, image: ImageArray) -> ImageArray:
        num_bits, orig_lsb = self._read_metadata(image)
        if orig_lsb is None or num_bits <= 0:
            raise InvalidInputError(
                "no reversible metadata found in image",
                hint="This image was not embedded with the reversible engine.",
            )

        result = image.copy()
        # 恢复红色通道原始 LSB
        h, w = image.shape[:2]
        red_flat = result[:, :, 0].ravel()
        total_pixels = h * w
        for i in range(min(num_bits, total_pixels)):
            red_flat[i] = (red_flat[i] & 0xFE) | int(orig_lsb[i])
        result[:, :, 0] = red_flat.reshape(h, w)
        return result


__all__ = ["ReversibleEngine"]
