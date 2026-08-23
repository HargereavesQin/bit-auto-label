"""能量机关 SAM3 自动标注可复用模块。

流水线：
1. YOLO pose ONNX 仅取 bbox（不采用网络关键点）
2. SAM3 文本提示在 ROI 内分割「圆盘」与「R」两个轮廓
3. 以 R 质心为极点、圆盘轮廓求极轴五点

公开 API::

    from buff_sam3_auto import BuffAutoStrategy, default_onnx_path

    strategy = BuffAutoStrategy(
        onnx_path=default_onnx_path(),
        sam_checkpoint="path/to/sam3.pt",
    )
    annotations, debug = strategy.annotate(image_bgr)
"""

from __future__ import annotations

from .detector import BBoxDetection, OrtBBoxDetector
from .geometry import (
    contour_centroid,
    select_circle_contour,
    select_r_component,
    solve_five_keypoints,
)
from .paths import default_bpe_candidates, default_onnx_path, package_root, weights_dir
from .segmenter import RSegmenter
from .strategy import BuffAutoStrategy

__all__ = [
    "BBoxDetection",
    "BuffAutoStrategy",
    "OrtBBoxDetector",
    "RSegmenter",
    "contour_centroid",
    "default_bpe_candidates",
    "default_onnx_path",
    "package_root",
    "select_circle_contour",
    "select_r_component",
    "solve_five_keypoints",
    "weights_dir",
]

__version__ = "1.0.0"
