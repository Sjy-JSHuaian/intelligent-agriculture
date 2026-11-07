"""LoveDA 数据集读取与统计。

目录结构（官方 LoveDA）：
    root/
      Train/ (或 train)
        Urban/images_png/*.png   (1024x1024 RGB)
        Urban/masks_png/*.png    (单通道，同名对应)
        Rural/images_png/*.png
        Rural/masks_png/*.png
      Val/ ...
      Test/ ...                  (可能无 masks_png，仅用于推理)

类别标签（官方 README，原始掩膜取值 0-7）：
    0 = no-data   （应忽略）
    1 = background
    2 = building
    3 = road
    4 = water
    5 = barren
    6 = forest
    7 = agriculture

训练时重映射：原始 1-7 -> 0-6（7 个语义类别），原始 0(no-data) -> 255(忽略)。
"""
from __future__ import annotations

import glob
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import read_image, read_mask

# 原始掩膜取值 -> 类别名（官方 README）
RAW_CLASSES = {
    0: "no-data",      # 忽略
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}

# 训练用标签（重映射后 0-6，共 7 个语义类别）
TRAIN_CLASSES = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "barren",
    5: "forest",
    6: "agriculture",
}

RAW_IGNORE = 0        # 原始掩膜中 no-data 的取值
IGNORE_INDEX = 255    # 重映射后训练时忽略值
NUM_CLASSES = 7       # 7 个语义类别

_SPLIT_ALIASES = {
    "train": ["Train", "train"],
    "val": ["Val", "val", "Valid", "valid", "Validation", "validation"],
    "test": ["Test", "test"],
}


def remap_mask(raw_mask: np.ndarray) -> np.ndarray:
    """原始掩膜(0-7) -> 训练掩膜(0-6, no-data=255)。"""
    mask = raw_mask.astype(np.int64) - 1   # 1-7 -> 0-6；0 -> -1
    mask[mask < 0] = IGNORE_INDEX          # no-data -> 255
    return mask


def _find_split_dir(root: str, split: str) -> str:
    """按名称大小写不敏感地定位 split 目录。"""
    for alias in _SPLIT_ALIASES[split]:
        candidate = os.path.join(root, alias)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"未找到 {split} 目录（候选: {_SPLIT_ALIASES[split]}），请确认 {root} 下已解压。")


def collect_samples(root: str, split: str, scenes: list[str] | None = None) -> list[tuple[str, str | None]]:
    """收集 (image_path, mask_path) 配对列表；无掩膜时 mask 为 None。

    通过递归查找 images_png 目录，自动适配 Urban/Rural 场景与大小写。
    """
    split_dir = _find_split_dir(root, split)
    image_files = sorted(glob.glob(os.path.join(split_dir, "**", "images_png", "*.png"), recursive=True))

    if scenes:
        image_files = [p for p in image_files
                       if any(s.lower() in os.path.normpath(p).lower().split(os.sep) for s in scenes)]

    pairs: list[tuple[str, str | None]] = []
    for img in image_files:
        mask = img.replace("images_png", "masks_png")
        pairs.append((img, mask if os.path.exists(mask) else None))
    return pairs


def compute_statistics(root: str, split: str, scenes: list[str] | None = None,
                       max_images: int | None = None) -> dict:
    """统计影像数量、尺寸、dtype 与各类别（原始取值 0-7）像素分布。"""
    pairs = collect_samples(root, split, scenes)
    if max_images is not None:
        pairs = pairs[:max_images]

    n_images = len(pairs)
    n_with_mask = sum(1 for _, m in pairs if m is not None)
    shapes = Counter()
    class_counts = Counter()
    dtype = None

    for img_path, mask_path in pairs:
        img = read_image(img_path)
        shapes[img.shape] += 1
        if dtype is None:
            dtype = str(img.dtype)
        if mask_path is not None:
            mask = read_mask(mask_path)
            counts = np.bincount(mask.ravel(), minlength=8)  # 取值 0-7，C 速度
            for v, c in enumerate(counts):
                if c:
                    class_counts[int(v)] += int(c)

    return {
        "split": split,
        "n_images": n_images,
        "n_with_mask": n_with_mask,
        "image_shapes": dict(shapes),
        "dtype": dtype,
        "class_pixel_counts": dict(sorted(class_counts.items())),
    }


def print_statistics(stats: dict) -> None:
    """以可读格式打印统计结果（原始取值 0-7）。"""
    print("=" * 60)
    print(f"[数据集统计] split={stats['split']}")
    print(f"  影像数量: {stats['n_images']}（含掩膜: {stats['n_with_mask']}）")
    print(f"  影像尺寸分布: {stats['image_shapes']}")
    print(f"  数据类型: {stats['dtype']}")
    total_px = sum(stats["class_pixel_counts"].values())
    print("  类别像素分布（原始取值）:")
    for cid in sorted(RAW_CLASSES):
        cnt = stats["class_pixel_counts"].get(cid, 0)
        ratio = cnt / total_px * 100 if total_px else 0.0
        flag = " (忽略)" if cid == RAW_IGNORE else ""
        print(f"    {cid} {RAW_CLASSES[cid]:12s} {cnt:12d}  {ratio:6.2f}%{flag}")
    # 汇总有效类别（重映射后）
    print("  重映射后有效类别占比:")
    valid_total = sum(stats["class_pixel_counts"].get(c, 0) for c in range(1, 8))
    for cid in sorted(TRAIN_CLASSES):
        raw_v = cid + 1
        cnt = stats["class_pixel_counts"].get(raw_v, 0)
        ratio = cnt / valid_total * 100 if valid_total else 0.0
        print(f"    {cid} {TRAIN_CLASSES[cid]:12s} {cnt:12d}  {ratio:6.2f}%")
    print("=" * 60)


class LoveDADataset(Dataset):
    """LoveDA 语义分割 Dataset。

    输出 image (CHW float) 与重映射后的 mask (HW long, 0-6, no-data=255)。
    """

    def __init__(self, pairs, transform, tile_size: tuple[int, int] = (256, 256)):
        self.pairs = [(img, mask) for img, mask in pairs if mask is not None]
        self.transform = transform
        self.tile_size = tile_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]
        image = read_image(img_path)              # HWC uint8 RGB
        mask = remap_mask(read_mask(mask_path))   # HW long (0-6, 255=ignore)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return {"image": image, "mask": mask, "name": os.path.basename(img_path)}
