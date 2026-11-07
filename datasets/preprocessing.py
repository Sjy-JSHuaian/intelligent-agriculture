"""图像预处理：读取、归一化、缩放与 RGB 植被指数。

重要说明：
    本项目当前数据为高分辨率 RGB 影像，**不含近红外（NIR）波段**，
    因此无法计算真实 NDVI。本模块提供的植被指数（VARI / ExG / GLI / NDI）
    均基于 RGB 三通道，仅作为植被覆盖度的近似特征，报告中须明确标注
    “RGB 植被指数”，不得称作 NDVI。
"""
from __future__ import annotations

import numpy as np
import cv2

# ImageNet 统计量（用于深度学习模型输入的标准化，仅对 RGB 3 通道有效）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_EPS = 1e-6


def read_image(path: str) -> np.ndarray:
    """读取影像为 HWC uint8 数组（RGB 顺序）。

    使用 np.fromfile + imdecode 而非 cv2.imread，以支持中文路径（Windows）。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # OpenCV 默认 BGR
    if img is None:
        raise FileNotFoundError(f"无法读取影像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str) -> np.ndarray:
    """读取单通道掩膜为 HW uint8 数组（中文路径安全）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩膜: {path}")
    return mask


def to_float01(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]，保持 HWC。"""
    return img.astype(np.float32) / 255.0


def standardize_rgb(img: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """RGB 影像标准化：HWC uint8/float -> CHW float32 (x-mean)/std。"""
    x = img.astype(np.float32) / 255.0
    x = (x - mean) / std
    return x.transpose(2, 0, 1)  # HWC -> CHW


def resize_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """按 (w, h) 缩放影像，插值方式为双线性。"""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """按 (w, h) 缩放掩膜，最近邻插值以保持类别整数。"""
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
# RGB 植被指数（非 NDVI）
# --------------------------------------------------------------------------- #
def compute_vari(rgb: np.ndarray) -> np.ndarray:
    """VARI (Visible Atmospherically Resistant Index) = (G - R) / (G + R - B)。

    分母 G+R-B 可能趋近 0（偏蓝像素），导致数值爆炸，故裁剪到 [-1, 1]
    （VARI 名义取值范围）。rgb: float 数组 [0,1]，形状 (H, W, 3)。
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vari = (g - r) / (g + r - b + _EPS)
    return np.clip(vari, -1.0, 1.0)


def compute_exg(rgb: np.ndarray) -> np.ndarray:
    """ExG (Excess Green) = 2*G - R - B，取值约 [-2, 2]。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 2.0 * g - r - b


def compute_gli(rgb: np.ndarray) -> np.ndarray:
    """GLI (Green Leaf Index) = (2G - R - B) / (2G + R + B)，取值约 [-1, 1]。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (2.0 * g - r - b) / (2.0 * g + r + b + _EPS)


def compute_ndi(rgb: np.ndarray) -> np.ndarray:
    """NDI (Normalized Difference Index) = (G - R) / (G + R)，取值约 [-1, 1]。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (g - r) / (g + r + _EPS)


def compute_rgb_indices(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """一次性计算所有 RGB 植被指数，返回 {名称: 数组}。"""
    return {
        "VARI": compute_vari(rgb),
        "ExG": compute_exg(rgb),
        "GLI": compute_gli(rgb),
        "NDI": compute_ndi(rgb),
    }
