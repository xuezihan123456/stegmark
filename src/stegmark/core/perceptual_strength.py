from __future__ import annotations

import numpy as np

from stegmark.core.native import BLOCK_SIZE, FloatArray


def compute_jnd_map(y_channel: FloatArray) -> FloatArray:
    """计算 Y 通道每个 8×8 块的局部方差，返回归一化 JND 图。

    高纹理区域返回高值（接近 1.0），平坦区域返回低值（接近 0.0）。
    对均匀图像返回全零。
    """
    height, width = y_channel.shape
    usable_height = height - (height % BLOCK_SIZE)
    usable_width = width - (width % BLOCK_SIZE)
    if usable_height == 0 or usable_width == 0:
        return np.empty((0, 0), dtype=np.float32)

    trimmed = y_channel[:usable_height, :usable_width]
    block_rows = usable_height // BLOCK_SIZE
    block_cols = usable_width // BLOCK_SIZE

    # 通过 reshape 将 8×8 块展开为向量，批量计算方差
    blocks = (
        trimmed
        .reshape(block_rows, BLOCK_SIZE, block_cols, BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(block_rows, block_cols, BLOCK_SIZE * BLOCK_SIZE)
        .astype(np.float32, copy=False)
    )
    variances = np.var(blocks, axis=2)  # (block_rows, block_cols)

    # 归一化到 [0, 1]，全零时返回全零
    vmax = variances.max()
    if vmax <= 0:
        return np.zeros((block_rows, block_cols), dtype=np.float32)

    # 使用 log 压缩大范围方差
    log_var = np.log1p(variances)
    log_max = np.log1p(vmax)
    return (log_var / log_max).astype(np.float32, copy=False)


def adaptive_delta(
    jnd_map: FloatArray,
    bits_array: np.ndarray,
    base_delta: float,
    strength: float,
) -> np.ndarray:
    """根据 JND 图为每个块生成自适应 delta 值。

    delta[i] = base_delta * strength * (0.5 + jnd_map[i])

    平坦区域最少获得 0.5 倍 delta，高纹理区域最多获得 1.5 倍 delta。
    返回形状与 jnd_map 一致的逐块 delta 数组。
    """
    flat = jnd_map.reshape(-1)
    if flat.size == 0:
        return np.array([], dtype=np.float32)
    # 展开到与 bits 匹配的长度
    actual_count = min(len(flat), len(bits_array))
    adaptive = (0.5 + flat[:actual_count]).astype(np.float32, copy=False)
    return np.maximum(base_delta * strength * adaptive, 2.0).astype(np.float32, copy=False)


__all__ = ["compute_jnd_map", "adaptive_delta"]
