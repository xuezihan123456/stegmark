from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stegmark.core.image_io import load_image
from stegmark.core.native import BLOCK_SIZE, COEFF_A, COEFF_B, _block_view, _dct2, _rgb_to_ycbcr
from stegmark.types import ImageArray


def generate_diff_heatmap(
    original: ImageArray,
    watermarked: ImageArray,
    *,
    amplify: float = 20.0,
) -> Image:
    """生成差值放大热力图。

    红色 = 正差值（水印图更亮），蓝色 = 负差值（水印图更暗），白色 = 无差值。
    """
    diff = (watermarked.astype(np.float32) - original.astype(np.float32)) * amplify
    # 映射到 [0, 255]：diff=-1→0(蓝), diff=0→127(白), diff=1→255(红)
    magnitude = np.clip(diff + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(magnitude, mode="RGB")


def generate_dct_modification_map(
    original: ImageArray,
    watermarked: ImageArray,
    *,
    threshold: float = 1.0,
) -> Image:
    """生成 DCT 系数修改位置覆盖图。

    在原始图像上叠加绿色标记，标出 (3,2) 和 (2,3) 位置 DCT 系数差值超过阈值的 8×8 块。
    绿色越亮 = 修改量越大。
    """
    y_orig, _, _ = _rgb_to_ycbcr(original)
    y_wm, _, _ = _rgb_to_ycbcr(watermarked)

    blocks_orig = _block_view(y_orig)
    blocks_wm = _block_view(y_wm)

    if blocks_orig.size == 0:
        return Image.fromarray(original, mode="RGB")

    coeffs_orig = _dct2(blocks_orig)
    coeffs_wm = _dct2(blocks_wm)

    diff_a = np.abs(
        coeffs_orig[..., COEFF_A[0], COEFF_A[1]] - coeffs_wm[..., COEFF_A[0], COEFF_A[1]]
    )
    diff_b = np.abs(
        coeffs_orig[..., COEFF_B[0], COEFF_B[1]] - coeffs_wm[..., COEFF_B[0], COEFF_B[1]]
    )
    max_diff = np.maximum(diff_a, diff_b)

    # 生成绿色覆盖图：被修改的块标为绿色，强度正比于修改量
    overlay = Image.fromarray(original, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    block_rows, block_cols = max_diff.shape
    if max_diff.max() > 0:
        normalized = (max_diff / max_diff.max() * 180).astype(np.uint8)
        for r in range(block_rows):
            for c in range(block_cols):
                if max_diff[r, c] > threshold:
                    alpha = int(min(255, normalized[r, c] + 75))
                    x0 = c * BLOCK_SIZE
                    y0 = r * BLOCK_SIZE
                    draw.rectangle(
                        [x0, y0, x0 + BLOCK_SIZE - 1, y0 + BLOCK_SIZE - 1],
                        fill=(0, 255, 0, alpha),
                    )

    return overlay.convert("RGB")


def generate_frequency_analysis(image: ImageArray) -> Image:
    """生成频域幅度谱图像。

    对 Y 通道做 FFT，输出对数刻度的幅度谱（中心为低频）。
    """
    y_channel, _, _ = _rgb_to_ycbcr(image)
    fft = np.fft.fft2(y_channel.astype(np.float32))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))
    # 归一化到 [0, 255]
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    else:
        magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    return Image.fromarray(magnitude, mode="L").convert("RGB")


def generate_robustness_heatmap(
    original: ImageArray,
    watermarked: ImageArray,
    *,
    jpeg_quality: int = 75,
) -> Image:
    """生成鲁棒性热力预测图。

    模拟 JPEG 压缩后，计算每个 8×8 块的恢复失败风险。
    红色 = 高风险（水印信号弱），绿色 = 低风险（水印信号强）。
    """
    y_orig, _, _ = _rgb_to_ycbcr(original)
    y_wm, _, _ = _rgb_to_ycbcr(watermarked)

    blocks_orig = _block_view(y_orig)
    blocks_wm = _block_view(y_wm)

    if blocks_orig.size == 0:
        return Image.fromarray(original, mode="RGB")

    # 简化鲁棒性评估：JPEG 压缩主要影响高频 DCT 系数
    # 计算每个块中 (3,2)/(2,3) 系数差值作为信号强度
    coeffs_orig = _dct2(blocks_orig)
    coeffs_wm = _dct2(blocks_wm)

    signal_a = np.abs(coeffs_wm[..., COEFF_A[0], COEFF_A[1]] - coeffs_orig[..., COEFF_A[0], COEFF_A[1]])
    signal_b = np.abs(coeffs_wm[..., COEFF_B[0], COEFF_B[1]] - coeffs_orig[..., COEFF_B[0], COEFF_B[1]])
    signal_strength = np.minimum(signal_a, signal_b)  # 最弱链决定鲁棒性

    # 归一化并反转：低信号 → 高风险（红），高信号 → 低风险（绿）
    if signal_strength.max() > 0:
        risk = 1.0 - signal_strength / signal_strength.max()
    else:
        risk = np.ones_like(signal_strength, dtype=np.float32)

    # 生成 RGBA 热力图：红(255,0,0)～绿(0,255,0)
    block_rows, block_cols = risk.shape
    rgb = np.zeros((block_rows * BLOCK_SIZE, block_cols * BLOCK_SIZE, 3), dtype=np.uint8)
    for r in range(block_rows):
        for c in range(block_cols):
            r_val = int(risk[r, c] * 255)
            g_val = int((1.0 - risk[r, c]) * 255)
            y0, y1 = r * BLOCK_SIZE, (r + 1) * BLOCK_SIZE
            x0, x1 = c * BLOCK_SIZE, (c + 1) * BLOCK_SIZE
            rgb[y0:y1, x0:x1, 0] = r_val
            rgb[y0:y1, x0:x1, 1] = g_val

    return Image.fromarray(rgb, mode="RGB")


def generate_full_report(
    original_path: Path | str,
    watermarked_path: Path | str,
    output_dir: Path | str,
    *,
    amplify: float = 20.0,
) -> list[Path]:
    """生成完整的取证报告，保存 4 张 PNG 到 output_dir。

    返回保存的文件路径列表。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    original = load_image(original_path).array
    watermarked = load_image(watermarked_path).array

    files: list[Path] = []

    # 1. 差值热力图
    heatmap = generate_diff_heatmap(original, watermarked, amplify=amplify)
    p = out / "forensics_diff_heatmap.png"
    heatmap.save(p, format="PNG")
    files.append(p)

    # 2. DCT 修改位置图
    dct_map = generate_dct_modification_map(original, watermarked)
    p = out / "forensics_dct_modifications.png"
    dct_map.save(p, format="PNG")
    files.append(p)

    # 3. 频域分析（水印图）
    freq = generate_frequency_analysis(watermarked)
    p = out / "forensics_frequency_analysis.png"
    freq.save(p, format="PNG")
    files.append(p)

    # 4. 鲁棒性热力图
    robustness = generate_robustness_heatmap(original, watermarked)
    p = out / "forensics_robustness_heatmap.png"
    robustness.save(p, format="PNG")
    files.append(p)

    return files


__all__ = [
    "generate_diff_heatmap",
    "generate_dct_modification_map",
    "generate_frequency_analysis",
    "generate_robustness_heatmap",
    "generate_full_report",
]
