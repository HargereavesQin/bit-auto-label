#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MobileNetV3-Small 装甲板数字分类器（<=600KB ONNX）.

针对 Armor Transform Dataset 的 20x28 灰度小图特化：
  - 单通道输入，stem stride=1 保留分辨率（小图不能下采样太狠）
  - 仅 2 次 stride=2 下采样（28x20 -> 14x10 -> 7x5）
  - Inverted Residual + SE + h-swish/h-relu 混合激活
  - 末端 1x1 conv + AvgPool + 单层 FC，无大 FC 头

mix 推理链（Function.cpp + Classify.cpp）对齐：
  透视 ROI 20x28 灰度 -> OTSU 二值化 -> /255 -> ONNX
  输出 9 类 logits，C++ 侧 softmax + remap_incidies -> armor id
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROI_INPUT_WIDTH = 20
ROI_INPUT_HEIGHT = 28
ROI_INPUT_CHANNELS = 1
NUM_NETWORK_CLASSES = 9
MAX_ONNX_BYTES = 800 * 1024
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# mix Classify.cpp: remap_incidies[network_class] -> armor_id
MIX_REMAP_ARMOR_IDS = (1, 2, 3, 4, 5, 6, 0, 7, 8)

# Armor Transform Dataset 文件夹 -> 网络输出类 index
FOLDER_TO_NETWORK_CLASS = {
    "idx_1": 0,
    "idx_2": 1,
    "idx_3": 2,
    "idx_4": 3,
    "idx_6": 5,
    "idx_0": 6,
    "idx_7": 7,
    "idx_7 (负样本)": 8,
}

NETWORK_CLASS_NAMES = (
    "guard", "digit_1", "digit_2", "digit_3", "digit_4",
    "outpost", "base", "unknown",
)


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    network_class_id: int


# ============================================================================
# MobileNetV3 基础构件
# ============================================================================
class Hswish(nn.Module):
    """hard swish: x * relu6(x+3) / 6"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0) / 6.0


class Hsigmoid(nn.Module):
    """hard sigmoid: relu6(x+3) / 6"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0) / 6.0


class SEModule(nn.Module):
    """Squeeze-and-Excitation, 1x1 conv 无 bias, reduction=4."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = Hsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.act1(self.bn1(self.fc1(w)))
        w = self.act2(self.bn2(self.fc2(w)))
        return x * w


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual: 1x1 expand -> dw 3x3 -> SE -> 1x1 project.

    use_hs=True 用 h-swish, False 用 ReLU（早期层省算力）。
    identity shortcut 仅当 stride==1 且 inp==oup。
    """

    def __init__(
        self,
        inp: int,
        hidden: int,
        oup: int,
        stride: int,
        use_se: bool = True,
        use_hs: bool = False,
        kernel: int = 3,
    ):
        super().__init__()
        self.identity = (stride == 1 and inp == oup)
        act = Hswish if use_hs else nn.ReLU
        layers: list[nn.Module] = []
        if inp != hidden:
            layers += [
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                act(inplace=True) if act is nn.ReLU else act(),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, kernel, stride, kernel // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            act(inplace=True) if act is nn.ReLU else act(),
        ]
        if use_se:
            layers.append(SEModule(hidden))
        layers += [
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.identity:
            return x + out
        return out


# ============================================================================
# MobileNetV3-Small (Armor 20x28 灰度特化)
# ============================================================================
class MobileNetV3SmallArmor(nn.Module):
    """MobileNetV3-Small 变体，1x28x20 灰度输入，9 类输出。

    结构（width_mult=1.0）:
      stem  : Conv 1->24  3x3 s1, BN, hswish        -> 24x28x20
      block1: IBN 24->24  exp=96  s1 SE relu        -> 24x28x20
      block2: IBN 24->32  exp=128 s2    relu        -> 32x14x10
      block3: IBN 32->64  exp=192 s2 SE hswish      -> 64x7x5
      block4: IBN 64->80  exp=224 s1 SE hswish      -> 80x7x5
      head  : Conv 80->160 1x1, BN, hswish          -> 160x7x5
      pool  : AdaptiveAvgPool -> 160
      fc    : Linear 160 -> num_classes
    约 135K 参数，FP32 ONNX 约 540-580 KB。
    """

    def __init__(
        self,
        num_classes: int = NUM_NETWORK_CLASSES,
        input_height: int = ROI_INPUT_HEIGHT,
        input_width: int = ROI_INPUT_WIDTH,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width

        def c(x: int) -> int:
            return max(int(x * width_mult), 1)

        c_stem = c(24)
        self.stem = nn.Sequential(
            nn.Conv2d(ROI_INPUT_CHANNELS, c_stem, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_stem),
            Hswish(),
        )
        self.block1 = InvertedResidual(c_stem, c(96), c(24), 1, use_se=True, use_hs=False)
        self.block2 = InvertedResidual(c(24), c(128), c(32), 2, use_se=False, use_hs=False)
        self.block3 = InvertedResidual(c(32), c(192), c(64), 2, use_se=True, use_hs=True)
        self.block4 = InvertedResidual(c(64), c(224), c(80), 1, use_se=True, use_hs=True)
        self.head = nn.Sequential(
            nn.Conv2d(c(80), c(160), 1, bias=False),
            nn.BatchNorm2d(c(160)),
            Hswish(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c(160), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# 向后兼容别名：旧脚本（train_mlp_conv.py / test_mlp_conv_pt.py /
# 下游部署脚本）仍 import MlpConv。
MlpConv = MobileNetV3SmallArmor


# ============================================================================
# 参数 / 大小估算
# ============================================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_fp32_bytes(model: nn.Module) -> int:
    return count_parameters(model) * 4


# ============================================================================
# 预处理（mix 同款：resize -> OTSU -> /255）
# ============================================================================
def preprocess_roi_image(
    image_bgr_or_gray: np.ndarray,
    input_width: int = ROI_INPUT_WIDTH,
    input_height: int = ROI_INPUT_HEIGHT,
) -> np.ndarray:
    """mix 同款预处理：resize -> OTSU -> /255，返回 float32 [H,W]。"""
    if image_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray
    resized = cv2.resize(gray, (input_width, input_height))
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0


def tensor_from_roi(image_bgr_or_gray: np.ndarray) -> torch.Tensor:
    """单样本 tensor [1, H, W]。"""
    normalized = preprocess_roi_image(image_bgr_or_gray)
    return torch.from_numpy(normalized).float().unsqueeze(0)


# ============================================================================
# 数据集发现
# ============================================================================
def discover_training_samples(dataset_root: Path) -> list[SampleRecord]:
    """扫描 Armor Transform Dataset，按 mix 类别 index 收集样本。"""
    records: list[SampleRecord] = []
    for folder_name, class_id in FOLDER_TO_NETWORK_CLASS.items():
        folder_path = dataset_root / folder_name
        if not folder_path.is_dir():
            continue
        for image_path in _iter_images(folder_path):
            records.append(SampleRecord(image_path=image_path, network_class_id=class_id))
    return records


def _iter_images(folder_path: Path) -> Iterable[Path]:
    for extension in IMAGE_EXTENSIONS:
        yield from sorted(folder_path.glob(f"*{extension}"))
        yield from sorted(folder_path.glob(f"*{extension.upper()}"))


def build_class_names(num_classes: int = NUM_NETWORK_CLASSES) -> list[str]:
    return list(NETWORK_CLASS_NAMES[:num_classes])


# ============================================================================
# 部署产物：preprocess.json / label.txt / ONNX
# ============================================================================
def save_preprocess_config(output_dir: Path) -> Path:
    """写入 preprocess.json，供 MLPClassifier / 部署读取。"""
    payload = {
        "mean": [0.0],
        "std": [1.0],
        "input_size": [ROI_INPUT_WIDTH, ROI_INPUT_HEIGHT],
        "channels": ROI_INPUT_CHANNELS,
        "architecture": "mobilenet_v3_small_armor",
        "mix_remap_armor_ids": list(MIX_REMAP_ARMOR_IDS),
    }
    config_path = output_dir / "preprocess.json"
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2, ensure_ascii=False)
    return config_path


def save_label_file(output_dir: Path, num_classes: int = NUM_NETWORK_CLASSES) -> Path:
    label_path = output_dir / "label.txt"
    labels = build_class_names(num_classes)
    with open(label_path, "w", encoding="utf-8") as label_file:
        label_file.write("\n".join(labels))
    return label_path


def export_onnx_model(
    model: nn.Module,
    onnx_path: Path,
    input_height: int = ROI_INPUT_HEIGHT,
    input_width: int = ROI_INPUT_WIDTH,
    opset_version: int = 11,
) -> Path:
    """导出 OpenCV dnn 可读的 ONNX（legacy exporter，单文件）。"""
    model.eval()
    dummy = torch.zeros(1, ROI_INPUT_CHANNELS, input_height, input_width)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamo=False,
    )
    return onnx_path


def assert_onnx_size_under_limit(onnx_path: Path, limit_bytes: int = MAX_ONNX_BYTES) -> int:
    size_bytes = onnx_path.stat().st_size
    if size_bytes > limit_bytes:
        raise ValueError(
            f"ONNX 大小 {size_bytes} bytes 超过限制 {limit_bytes} bytes "
            f"({limit_bytes // 1024} KB)"
        )
    return size_bytes
