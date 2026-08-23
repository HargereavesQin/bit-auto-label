#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO-Pose 标签 -> MLP ROI 数据集.

读已有 YOLO-pose 标签里的 4 关键点（不跑 YOLO 推理），按 mix.cpp 的逻辑：
传统视觉椭圆拟合修正关键点 -> 透视拉正 -> 中心裁剪（去掉两侧灯条）-> 可选轻度增强 -> OTSU 二值化，
把装甲板中间黑白图案存成 ROI 图。不分类，全部存一个文件夹，供分类网络训练。

用法:
    python scripts/yolo_2_mlp.py <数据集根目录> --out <输出目录>
                        [--no-refine] [--no-binarize] [--size 20 28] [--min-vis 0.3]
                        [--config config/yolo_2_mlp_aug.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.dataset_layout import resolve_open_layout  # noqa: E402
from core.mlp_classifier import MLPClassifier  # noqa: E402

# ============== 模块级常量（无魔法值） ==============
YOLO_POSE_BBOX_VALUE_COUNT = 5          # class_id, cx, cy, w, h
YOLO_POSE_VALUES_PER_KEYPOINT = 3       # x, y, visibility
EXPECTED_KEYPOINT_COUNT = 4             # 装甲板 4 个灯条端点
DEFAULT_ROI_OUTPUT_WIDTH = 20           # 40
DEFAULT_ROI_OUTPUT_HEIGHT = 28          
DEFAULT_KEYPOINT_VISIBILITY_THRESHOLD = 0.3
IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
POSE_SHAPE_TYPES = ("pose", "hierarchy_pose")
ROI_OUTPUT_SUFFIX = ".jpg"
DEFAULT_AUG_CONFIG_PATH = _REPO_ROOT / "config" / "yolo_2_mlp_aug.yaml"
# 写死的源数据集路径（YOLO split: images/{train,val,test} + labels/{train,val,test}）
DEFAULT_DATASET_ROOT = "/home/dreamchaser/Desktop/armor-label/autolabel/armor-label/datasets/pose_2026_sz"
YOLO_SPLIT_NAMES = ("train", "val", "test")
GRAY_MIDPOINT = 128.0
AUG_NO_OP_EPSILON = 1e-6
CLARITY_BLUR_KERNEL_SIZE = 3
CLARITY_AMOUNT_SCALE = 1.5              # clarity=1.0 时 unsharp 强度上限
BRIGHTNESS_AMOUNT_SCALE = 50.0          # brightness=1.0 约等效 +50 灰度
CONTRAST_BASE = 1.0                     # contrast 增量叠在 1.0 上


@dataclass(frozen=True)
class RoiAugmentationConfig:
    """轻度增强强度；全 0 时恒等变换，便于后续按效果调参。"""

    contrast: float = 0.0
    clarity: float = 0.0
    brightness: float = 0.0

    def is_noop(self) -> bool:
        return (
            abs(self.contrast) < AUG_NO_OP_EPSILON
            and abs(self.clarity) < AUG_NO_OP_EPSILON
            and abs(self.brightness) < AUG_NO_OP_EPSILON
        )


def load_augmentation_config(config_path: Optional[str]) -> RoiAugmentationConfig:
    """从 YAML 读取 augmentation 段；缺省或全 0 则不做增强。"""
    if not config_path:
        return RoiAugmentationConfig()
    path = Path(config_path)
    if not path.is_file():
        print(f"[WARN] 增强配置不存在，跳过增强: {path}", file=sys.stderr)
        return RoiAugmentationConfig()
    try:
        import yaml
    except ImportError:
        print("[WARN] 未安装 PyYAML，跳过增强配置", file=sys.stderr)
        return RoiAugmentationConfig()
    with open(path, "r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file) or {}
    section = payload.get("augmentation", {}) or {}
    return RoiAugmentationConfig(
        contrast=float(section.get("contrast", 0.0)),
        clarity=float(section.get("clarity", 0.0)),
        brightness=float(section.get("brightness", 0.0)),
    )


def apply_brightness_delta(image: np.ndarray, brightness: float) -> np.ndarray:
    """亮度增量；0 原样返回。"""
    if abs(brightness) < AUG_NO_OP_EPSILON:
        return image
    boosted = image.astype(np.float32) + brightness * BRIGHTNESS_AMOUNT_SCALE
    return np.clip(boosted, 0, 255).astype(np.uint8)


def apply_contrast_delta(image: np.ndarray, contrast: float) -> np.ndarray:
    """对比度增量叠在 1.0 上；0 原样返回。"""
    if abs(contrast) < AUG_NO_OP_EPSILON:
        return image
    scale = CONTRAST_BASE + contrast
    adjusted = scale * (image.astype(np.float32) - GRAY_MIDPOINT) + GRAY_MIDPOINT
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_clarity_delta(image: np.ndarray, clarity: float) -> np.ndarray:
    """Unsharp 锐化模拟清晰度；0 原样返回。"""
    if abs(clarity) < AUG_NO_OP_EPSILON:
        return image
    blurred = cv2.GaussianBlur(image, (CLARITY_BLUR_KERNEL_SIZE, CLARITY_BLUR_KERNEL_SIZE), 0)
    amount = clarity * CLARITY_AMOUNT_SCALE
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_roi_augmentation(image: np.ndarray, config: RoiAugmentationConfig) -> np.ndarray:
    """按配置顺序做亮度 -> 对比度 -> 清晰度；全 0 时零开销返回。"""
    if config.is_noop():
        return image
    enhanced = apply_brightness_delta(image, config.brightness)
    enhanced = apply_contrast_delta(enhanced, config.contrast)
    enhanced = apply_clarity_delta(enhanced, config.clarity)
    return enhanced


def binarize_roi_otsu(image: np.ndarray) -> np.ndarray:
    """OTSU 二值化；彩色先转灰度。"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary



def parse_yolo_pose_line(label_line: str, image_width: int, image_height: int,
                         visibility_threshold: float) -> Optional[tuple[np.ndarray, np.ndarray, bool]]:
    """解析一行 YOLO-pose 标签 -> (keypointsPixel[4,2], bboxXyxy[4], allVisible).

    归一化坐标反归一化到像素空间。关键点可见性低于阈值则该实例返回 None。
    """
    parts = label_line.strip().split()
    if len(parts) < YOLO_POSE_BBOX_VALUE_COUNT:
        return None
    try:
        class_id = int(float(parts[0]))
        center_x = float(parts[1]) * image_width
        center_y = float(parts[2]) * image_height
        box_width = float(parts[3]) * image_width
        box_height = float(parts[4]) * image_height
    except (ValueError, IndexError):
        return None

    keypoint_values = parts[YOLO_POSE_BBOX_VALUE_COUNT:]
    keypoint_count = len(keypoint_values) // YOLO_POSE_VALUES_PER_KEYPOINT
    if keypoint_count < EXPECTED_KEYPOINT_COUNT:
        return None

    keypoints_pixel = np.zeros((EXPECTED_KEYPOINT_COUNT, 2), dtype=np.float32)
    all_visible = True
    for keypoint_index in range(EXPECTED_KEYPOINT_COUNT):
        offset = keypoint_index * YOLO_POSE_VALUES_PER_KEYPOINT
        try:
            keypoint_x = float(keypoint_values[offset]) * image_width
            keypoint_y = float(keypoint_values[offset + 1]) * image_height
            keypoint_vis = float(keypoint_values[offset + 2])
        except (ValueError, IndexError):
            return None
        if keypoint_vis < visibility_threshold:
            all_visible = False
        keypoints_pixel[keypoint_index] = (keypoint_x, keypoint_y)

    bbox_xyxy = np.array([
        center_x - box_width / 2.0,
        center_y - box_height / 2.0,
        center_x + box_width / 2.0,
        center_y + box_height / 2.0,
    ], dtype=np.float32)
    return keypoints_pixel, bbox_xyxy, all_visible


def iter_yolo_pose_instances(label_path: str, image_width: int, image_height: int,
                             visibility_threshold: float) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """从 YOLO txt 标签逐实例产出 (keypointsPixel, bboxXyxy)。"""
    with open(label_path, "r", encoding="utf-8") as label_file:
        for line in label_file:
            parsed = parse_yolo_pose_line(line, image_width, image_height, visibility_threshold)
            if parsed is None or not parsed[2]:
                continue
            yield parsed[0], parsed[1]


def iter_json_pose_instances(label_path: str) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """从扁平 LabelMe-JSON 标签逐实例产出 (keypointsPixel, bboxXyxy)。

    JSON 里 keypoints 已是像素坐标;points 是 bbox 对角两点。
    """
    with open(label_path, "r", encoding="utf-8") as label_file:
        payload = json.load(label_file)
    for shape in payload.get("shapes", []):
        if shape.get("shape_type") not in POSE_SHAPE_TYPES:
            continue
        raw_points = shape.get("points")
        raw_keypoints = shape.get("keypoints")
        if not raw_points or not raw_keypoints or len(raw_keypoints) < EXPECTED_KEYPOINT_COUNT:
            continue
        corner_a, corner_b = raw_points[0], raw_points[1]
        bbox_xyxy = np.array([
            min(corner_a[0], corner_b[0]),
            min(corner_a[1], corner_b[1]),
            max(corner_a[0], corner_b[0]),
            max(corner_a[1], corner_b[1]),
        ], dtype=np.float32)
        keypoints_pixel = np.array(
            [[kp[0], kp[1]] for kp in raw_keypoints[:EXPECTED_KEYPOINT_COUNT]],
            dtype=np.float32,
        )
        yield keypoints_pixel, bbox_xyxy


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR -> 单通道灰度；已是单通道则原样返回。"""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def rectify_one_instance(image: np.ndarray, keypoints_pixel: np.ndarray,
                         bbox_xyxy: np.ndarray, use_refine: bool,
                         output_size: tuple[int, int], mode: str,
                         augmentation: RoiAugmentationConfig
                         ) -> tuple[Optional[np.ndarray], str]:
    """对单实例做 修正->透视->裁剪->增强->后处理，返回 (roi, source)。   
    增强在二值化/灰度化后处理前做，否则 0/255 图上对比度/清晰度几乎无效。
    """
    corrected_keypoints = None
    source_tag = "fallback"
    if use_refine:
        corrected_keypoints, _ = MLPClassifier.refine_keypoints_and_clean(
            image, keypoints_pixel, bbox_xyxy
        )
        if corrected_keypoints is not None:
            source_tag = "refined"

    # binary + 无增强时直接走 extract 内置 OTSU，避免多一次彩色->灰度转换
    do_binary_inline = mode == "binary" and not augmentation.is_noop()
    extract_binarize = mode == "binary" and augmentation.is_noop()

    rectified_roi = MLPClassifier.extract_armor_roi(
        image, bbox_xyxy, keypoints_pixel,
        output_size=output_size,
        corrected_kpts=corrected_keypoints,
        binarize=extract_binarize,
    )
    if rectified_roi is None:
        return None, source_tag

    # 非 binary 内置分支：先增强，再按 mode 后处理
    if not extract_binarize:
        rectified_roi = apply_roi_augmentation(rectified_roi, augmentation)
        if mode == "grayscale":
            rectified_roi = to_grayscale(rectified_roi)
        elif mode == "binary":  # do_binary_inline
            rectified_roi = binarize_roi_otsu(rectified_roi)
        # color: 保持 BGR
    return rectified_roi, source_tag


def save_roi_image(roi: np.ndarray, output_dir: Path, image_stem: str, instance_index: int) -> Path:
    """把单张 ROI 写盘，返回保存路径。"""
    if roi.ndim == 2:
        save_image = roi
    else:
        save_image = roi
    filename = f"{image_stem}_{instance_index:04d}{ROI_OUTPUT_SUFFIX}"
    save_path = output_dir / filename
    write_ok = cv2.imwrite(str(save_path), save_image)
    if not write_ok:
        raise IOError(f"写入失败: {save_path}")
    return save_path


class MlpRoiExtractor:
    """承载提取配置与统计，无全局可变状态。"""

    def __init__(self, output_dir: Path, use_refine: bool, mode: str,
                 output_size: tuple[int, int], visibility_threshold: float,
                 augmentation: Optional[RoiAugmentationConfig] = None):
        self.output_dir = output_dir
        self.use_refine = use_refine
        self.mode = mode
        self.output_size = output_size
        self.visibility_threshold = visibility_threshold
        self.augmentation = augmentation or RoiAugmentationConfig()
        self.saved_count = 0
        self.refined_count = 0
        self.fallback_count = 0
        self.skipped_count = 0

    def process_image(self, image_path: str, label_path: Optional[str],
                       label_format: Optional[str]) -> None:
        """处理单张图：读图 -> 遍历实例 -> 修正 -> 透视 -> 存盘。"""
        image = cv2.imread(image_path)
        if image is None:
            return
        image_height, image_width = image.shape[:2]
        image_stem = Path(image_path).stem

        instances = self._collect_instances(label_path, label_format,
                                             image_width, image_height)
        for instance_index, (keypoints_pixel, bbox_xyxy) in enumerate(instances, start=1):
            self._process_one(image, keypoints_pixel, bbox_xyxy,
                              image_stem, instance_index)

    def _collect_instances(self, label_path: Optional[str], label_format: Optional[str],
                           image_width: int, image_height: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """按标签格式收集本图的全部实例。"""
        if not label_path or not os.path.isfile(label_path):
            return []
        if label_format == "yolo" or label_path.lower().endswith(".txt"):
            return list(iter_yolo_pose_instances(label_path, image_width, image_height,
                                                 self.visibility_threshold))
        return list(iter_json_pose_instances(label_path))

    def _process_one(self, image: np.ndarray, keypoints_pixel: np.ndarray,
                     bbox_xyxy: np.ndarray, image_stem: str, instance_index: int) -> None:
        """处理并保存单个实例，更新统计。"""
        roi, source_tag = rectify_one_instance(
            image, keypoints_pixel, bbox_xyxy,
            use_refine=self.use_refine,
            output_size=self.output_size,
            mode=self.mode,
            augmentation=self.augmentation,
        )
        if roi is None:
            self.skipped_count += 1
            return
        try:
            save_roi_image(roi, self.output_dir, image_stem, instance_index)
        except IOError as error:
            print(f"[WARN] {error}", file=sys.stderr)
            self.skipped_count += 1
            return
        self.saved_count += 1
        if source_tag == "refined":
            self.refined_count += 1
        else:
            self.fallback_count += 1

    def report(self) -> str:
        """打印并返回汇总。"""
        aug = self.augmentation
        summary = (
            f"保存: {self.saved_count} | refined: {self.refined_count} | "
            f"fallback: {self.fallback_count} | skipped: {self.skipped_count} | "
            f"mode={self.mode} | "
            f"aug(contrast={aug.contrast}, clarity={aug.clarity}, brightness={aug.brightness})"
        )
        print(summary)
        return summary


def _filter_by_split(image_paths: list[str], split: str) -> list[str]:
    """按 YOLO split 名过滤图片路径；split='all' 或非 YOLO 布局时原样返回。"""
    if not split or split == "all":
        return image_paths
    split_marker = f"/images/{split}/"
    filtered = [p for p in image_paths if split_marker in p.replace("\\", "/")]
    return filtered or image_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器。"""
    parser = argparse.ArgumentParser(
        description="YOLO-Pose 标签 -> MLP ROI 数据集（不跑 YOLO，不分类）"
    )
    parser.add_argument("dataset_root", type=str, nargs="?",
                        default=DEFAULT_DATASET_ROOT,
                        help=f"数据集根目录（YOLO 或扁平 JSON），默认: {DEFAULT_DATASET_ROOT}")
    parser.add_argument("--out", type=str, required=True, help="ROI 输出目录")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "val", "test"],
                        help="只处理某个 split（YOLO 数据集有效），默认 all")
    parser.add_argument("--no-refine", action="store_true",
                        help="跳过传统视觉椭圆拟合修正，直接用 YOLO 关键点")
    parser.add_argument("--no-binarize", action="store_true",
                        help="保留彩色 ROI，不做 OTSU 二值化（等价 --mode color）")
    parser.add_argument("--grayscale", action="store_true",
                        help="输出单通道灰度图（保留真实亮度信息，不阈值；黑白电影感）")
    parser.add_argument("--mode", type=str,
                        choices=["binary", "grayscale", "color"],
                        default=None,
                        help="输出模式：binary=OTSU二值 / grayscale=单通道灰度 / color=彩色。"
                             "覆盖 --no-binarize / --grayscale")
    parser.add_argument("--size", type=int, nargs=2,
                        default=[DEFAULT_ROI_OUTPUT_WIDTH, DEFAULT_ROI_OUTPUT_HEIGHT],
                        metavar=("WIDTH", "HEIGHT"), help="输出 ROI 尺寸 W H，默认 20 28")
    parser.add_argument("--min-vis", type=float,
                        default=DEFAULT_KEYPOINT_VISIBILITY_THRESHOLD,
                        help="关键点可见性阈值，低于则跳过该实例")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_AUG_CONFIG_PATH),
        help=f"增强配置 YAML（默认 {DEFAULT_AUG_CONFIG_PATH}；全 0=不增强）",
    )
    return parser


def main() -> None:
    """入口：解析参数 -> 解析布局 -> 遍历图片 -> 提取 ROI。"""
    args = build_arg_parser().parse_args()
    dataset_root = args.dataset_root
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentation = load_augmentation_config(args.config)
    layout = resolve_open_layout(dataset_root)
    if not layout.image_paths:
        print(f"[ERROR] 未找到图片: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    # 按 --split 过滤（YOLO 数据集 images/<split>/...）
    image_paths = _filter_by_split(layout.image_paths, args.split)
    if not image_paths:
        print(f"[ERROR] split={args.split} 未匹配到图片: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    if args.mode is not None:
        roi_mode = args.mode
    elif args.grayscale:
        roi_mode = "grayscale"
    elif args.no_binarize:
        roi_mode = "color"
    else:
        roi_mode = "binary"

    extractor = MlpRoiExtractor(
        output_dir=output_dir,
        use_refine=not args.no_refine,
        mode=roi_mode,
        output_size=(args.size[0], args.size[1]),
        visibility_threshold=args.min_vis,
        augmentation=augmentation,
    )

    total_images = len(image_paths)
    print(f"数据集: {dataset_root} | split={args.split} | 图片数={total_images} | mode={roi_mode}")
    for processed_index, image_path in enumerate(image_paths, start=1):
        label_path = layout.annotation_map.get(image_path)
        label_format = layout.annotation_format.get(image_path)
        extractor.process_image(image_path, label_path, label_format)
        if processed_index % 50 == 0:
            print(f"进度 {processed_index}/{total_images} | 已保存 {extractor.saved_count}")

    print(f"输出目录: {output_dir}")
    print(f"增强配置: {args.config}")
    extractor.report()


if __name__ == "__main__":
    main()
