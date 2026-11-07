"""数据增强（基于 albumentations），训练/验证两套流水线。"""
from __future__ import annotations

import albumentations as A

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD


def get_train_augmentation(tile_size: tuple[int, int] = (256, 256)) -> A.Compose:
    """训练增强：随机裁剪 256x256 + 随机翻转/旋转 90° + Normalize（ImageNet）。

    输入影像为 1024x1024 uint8，RandomCrop 保持原始分辨率（利于梯田边界识别）。
    """
    h, w = tile_size
    return A.Compose([
        A.RandomCrop(height=h, width=w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
    ])


def get_val_augmentation(tile_size: tuple[int, int] = (256, 256)) -> A.Compose:
    """验证/测试增强：Resize 到固定尺寸 + Normalize，不做几何变换。"""
    h, w = tile_size
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=1, mask_interpolation=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
    ])
