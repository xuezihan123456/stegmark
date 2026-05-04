from __future__ import annotations

"""StegMark WASM — 浏览器端隐形水印（Pyodide 兼容）。

纯 Python 实现，仅依赖 math/zlib（Pyodide 内置），无 numpy 依赖。
在浏览器中通过 Pyodide 运行时直接调用 embed_image() / extract_message()。
"""

import math
import zlib

FRAME_VERSION = 1
HEADER_BYTES = 3
CRC_BYTES = 4
MAX_PAYLOAD_BYTES = 65_535
BLOCK_SIZE = 8
BASE_DELTA = 12.0


def _encode_text_wasm(message: str) -> list[int]:
    payload = message.encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"message too long: {len(payload)} > {MAX_PAYLOAD_BYTES}")
    frame = bytearray()
    frame.append(FRAME_VERSION)
    frame.extend(len(payload).to_bytes(2, "big"))
    frame.extend(payload)
    frame.extend(zlib.crc32(payload).to_bytes(4, "big"))
    bits: list[int] = []
    for byte in bytes(frame):
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits_to_bytes_wasm(bits: list[int]) -> bytes:
    result = bytearray()
    for i in range(0, len(bits) - len(bits) % 8, 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)


def _decode_frame_wasm(bits: list[int]) -> tuple[bool, str, str]:
    if len(bits) < (HEADER_BYTES + CRC_BYTES) * 8:
        return False, "", "too_short"
    data = _bits_to_bytes_wasm(bits)
    plen = int.from_bytes(data[1:3], "big")
    pend = HEADER_BYTES + plen
    cend = pend + CRC_BYTES
    if len(data) < cend:
        return False, "", "truncated"
    payload = data[HEADER_BYTES:pend]
    if int.from_bytes(data[pend:cend], "big") != zlib.crc32(payload):
        return False, "", "crc_mismatch"
    try:
        return True, payload.decode("utf-8"), ""
    except UnicodeDecodeError:
        return False, "", "decode_error"


def _rgb_to_y(pixels):
    h, w = len(pixels), len(pixels[0])
    y = [[0.0] * w for _ in range(h)]
    for r in range(h):
        row = y[r]
        prow = pixels[r]
        for c in range(w):
            row[c] = 0.299 * prow[c][0] + 0.587 * prow[c][1] + 0.114 * prow[c][2]
    return y


def _dct_8x8(block):
    n = 8
    result = [[0.0] * n for _ in range(n)]
    for k in range(n):
        ak = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        for l_val in range(n):
            al = math.sqrt(1.0 / n) if l_val == 0 else math.sqrt(2.0 / n)
            total = 0.0
            for i in range(n):
                bi = block[i]
                cos_k = math.cos(math.pi * k * (2 * i + 1) / (2 * n))
                for j in range(n):
                    total += bi[j] * cos_k * math.cos(math.pi * l_val * (2 * j + 1) / (2 * n))
            result[k][l_val] = ak * al * total
    return result


def _idct_8x8(coeffs):
    n = 8
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                ak = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
                cos_i = math.cos(math.pi * k * (2 * i + 1) / (2 * n))
                for l_val in range(n):
                    al = math.sqrt(1.0 / n) if l_val == 0 else math.sqrt(2.0 / n)
                    total += ak * al * coeffs[k][l_val] * cos_i * math.cos(math.pi * l_val * (2 * j + 1) / (2 * n))
            result[i][j] = total
    return result


def embed_image(pixels, message, *, strength=1.0):
    bits = _encode_text_wasm(message)
    h, w = len(pixels), len(pixels[0])
    uh = h - (h % BLOCK_SIZE)
    uw = w - (w % BLOCK_SIZE)
    br, bc = uh // BLOCK_SIZE, uw // BLOCK_SIZE
    y = _rgb_to_y(pixels)
    delta = max(4.0, BASE_DELTA * strength)
    bi = 0
    for r in range(br):
        if bi >= len(bits):
            break
        for c in range(bc):
            if bi >= len(bits):
                break
            y0, y1 = r * BLOCK_SIZE, (r + 1) * BLOCK_SIZE
            x0, x1 = c * BLOCK_SIZE, (c + 1) * BLOCK_SIZE
            block = [[y[yy][xx] for xx in range(x0, x1)] for yy in range(y0, y1)]
            coeffs = _dct_8x8(block)
            mid = (coeffs[3][2] + coeffs[2][3]) / 2.0
            s = delta / 2.0 if bits[bi] == 1 else -delta / 2.0
            coeffs[3][2] = mid + s
            coeffs[2][3] = mid - s
            restored = _idct_8x8(coeffs)
            for yy in range(BLOCK_SIZE):
                for xx in range(BLOCK_SIZE):
                    y[y0 + yy][x0 + xx] = restored[yy][xx]
            bi += 1
    y_orig = _rgb_to_y(pixels)
    result = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
    for rr in range(h):
        for cc in range(w):
            dy = y[rr][cc] - y_orig[rr][cc]
            for ch in range(3):
                result[rr][cc][ch] = max(0, min(255, int(pixels[rr][cc][ch] + dy)))
    return result


def extract_message(pixels):
    h, w = len(pixels), len(pixels[0])
    uh = h - (h % BLOCK_SIZE)
    uw = w - (w % BLOCK_SIZE)
    br, bc = uh // BLOCK_SIZE, uw // BLOCK_SIZE
    y = _rgb_to_y(pixels)
    bits: list[int] = []
    for r in range(br):
        for c in range(bc):
            y0, y1 = r * BLOCK_SIZE, (r + 1) * BLOCK_SIZE
            x0, x1 = c * BLOCK_SIZE, (c + 1) * BLOCK_SIZE
            block = [[y[yy][xx] for xx in range(x0, x1)] for yy in range(y0, y1)]
            coeffs = _dct_8x8(block)
            bits.append(1 if coeffs[3][2] >= coeffs[2][3] else 0)
    valid, message, _ = _decode_frame_wasm(bits)
    return message if valid else None


__all__ = ["embed_image", "extract_message"]
