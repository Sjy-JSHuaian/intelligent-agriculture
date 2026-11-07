"""影像切片：将大尺寸影像切分为固定大小的子块，用于训练与推理。"""
from __future__ import annotations

import numpy as np


def tile_positions(height: int, width: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    """返回所有切片的左上角坐标 (y, x)。"""
    positions = []
    for y in range(0, max(1, height - tile_size + 1), stride):
        for x in range(0, max(1, width - tile_size + 1), stride):
            # 处理边界：右/下越界时向左上收缩，保证切片大小一致
            y = min(y, height - tile_size)
            x = min(x, width - tile_size)
            positions.append((y, x))
    return positions


def tile_array(arr: np.ndarray, tile_size: int, stride: int) -> list[np.ndarray]:
    """将数组（H, W, ...）切分为 tile_size 大小的子块列表。"""
    h, w = arr.shape[:2]
    tiles = []
    for y, x in tile_positions(h, w, tile_size, stride):
        tiles.append(arr[y:y + tile_size, x:x + tile_size])
    return tiles


def tile_pair(
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    stride: int,
    min_valid_ratio: float = 0.05,
    ignore_index: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """切分影像+掩膜，返回 [(tile_image, tile_mask, y, x), ...]。

    过滤掉有效像素（非 ignore_index）占比过低的切片，避免大量纯背景块。
    """
    h, w = image.shape[:2]
    out = []
    for y, x in tile_positions(h, w, tile_size, stride):
        tile_img = image[y:y + tile_size, x:x + tile_size]
        tile_mask = mask[y:y + tile_size, x:x + tile_size]
        valid = (tile_mask != ignore_index).mean()
        if valid >= min_valid_ratio:
            out.append((tile_img, tile_mask, y, x))
    return out
