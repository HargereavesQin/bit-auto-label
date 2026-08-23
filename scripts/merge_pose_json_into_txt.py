#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 YOLO 分目录数据集中新增的 LabelMe JSON 位姿标注合并进已有 txt。

规则（默认 IoU >= 0.8 视为同一目标）:
  - JSON 与某 txt 行 bbox 重合 >= 阈值: 视为改 ID，用 JSON 行替换该 txt 行
  - JSON 与所有 txt 行重合 < 阈值: 视为新标注，追加到 txt
  - 其他未重合的 txt 行保持不变
  - 处理完成后删除 JSON

用法:
  python scripts/merge_pose_json_into_txt.py
  python scripts/merge_pose_json_into_txt.py --split train --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.constants import HIER_CLASS_NAMES, BUFF_CLASS_NAMES
from core.shape_registry import ensure_default_specs, get as spec_get
from core.constants import SHAPE_POSE, SHAPE_HIERARCHY

POSE_SHAPE_TYPES = {"pose", "hierarchy_pose", SHAPE_POSE, SHAPE_HIERARCHY}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MergeStats:
    json_files: int = 0
    shapes_total: int = 0
    appended: int = 0
    replaced: int = 0
    skipped: int = 0
    deleted_json: int = 0


def load_class_names(data_yaml: Path) -> list[str]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return list(names)
    return list(HIER_CLASS_NAMES) + list(BUFF_CLASS_NAMES)


def bbox_xyxy_from_yolo_parts(parts: list[str], w: int, h: int) -> Optional[np.ndarray]:
    if len(parts) < 5:
        return None
    try:
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
    except ValueError:
        return None
    return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32)


def bbox_xyxy_from_json_points(points: list) -> Optional[np.ndarray]:
    if not points or len(points) < 2:
        return None
    xs = [p[0] for p in points[:2]]
    ys = [p[1] for p in points[:2]]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def json_shape_to_dict(shape: dict) -> Optional[dict]:
    st = shape.get("shape_type") or shape.get("type") or ""
    if st not in POSE_SHAPE_TYPES and not shape.get("keypoints"):
        return None
    if not shape.get("keypoints") or len(shape.get("keypoints", [])) < 4:
        return None
    if not shape.get("points") or len(shape.get("points", [])) < 2:
        return None
    label = (shape.get("label") or "").strip()
    if not label:
        return None
    type_name = SHAPE_HIERARCHY if st in ("hierarchy_pose", SHAPE_HIERARCHY) else SHAPE_POSE
    return {
        "label": label,
        "type": type_name,
        "points": shape["points"][:2],
        "keypoints": shape["keypoints"],
        "kpt_names": shape.get("kpt_names", []),
    }


def shape_dict_to_yolo_line(shape_dict: dict, w: int, h: int, class_names: list[str]) -> Optional[str]:
    label = shape_dict["label"]
    if label not in class_names:
        return None
    cid = class_names.index(label)
    spec = spec_get(shape_dict["type"])
    if spec is None or spec.to_yolo_line is None:
        return None
    return spec.to_yolo_line(shape_dict, w, h, cid)


def read_image_size(img_path: Path, json_data: dict) -> tuple[int, int, Optional[np.ndarray]]:
    w = int(json_data.get("imageWidth") or 0)
    h = int(json_data.get("imageHeight") or 0)
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        ih, iw = img.shape[:2]
        if w <= 0 or h <= 0:
            w, h = iw, ih
        return w, h, img
    if w > 0 and h > 0:
        return w, h, None
    return 0, 0, None


def find_json_files(img_dir: Path, lbl_dir: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for directory in (img_dir, lbl_dir):
        if not directory.is_dir():
            continue
        for p in directory.glob("*.json"):
            if p.name in ("pose_classes.json", "data.json"):
                continue
            found.setdefault(p.stem, p)
    return sorted(found.values(), key=lambda p: p.name)


def merge_one_json(
    json_path: Path,
    img_dir: Path,
    lbl_dir: Path,
    class_names: list[str],
    iou_threshold: float,
    dry_run: bool,
) -> tuple[str, int, int, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stem = json_path.stem
    img_path = None
    for ext in IMAGE_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.is_file():
            img_path = candidate
            break
    if img_path is None:
        return "no_image", 0, 0, 0

    w, h, _ = read_image_size(img_path, data)
    if w <= 0 or h <= 0:
        return "bad_size", 0, 0, 0

    txt_path = lbl_dir / f"{stem}.txt"
    existing_lines: list[str] = []
    if txt_path.is_file():
        with open(txt_path, "r", encoding="utf-8") as f:
            existing_lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    line_bboxes: list[Optional[np.ndarray]] = []
    for line in existing_lines:
        parts = line.split()
        line_bboxes.append(bbox_xyxy_from_yolo_parts(parts, w, h))

    claimed: set[int] = set()
    appended = replaced = skipped = 0

    for shape in data.get("shapes", []):
        shape_dict = json_shape_to_dict(shape)
        if shape_dict is None:
            skipped += 1
            continue
        json_line = shape_dict_to_yolo_line(shape_dict, w, h, class_names)
        if not json_line:
            skipped += 1
            continue
        json_bbox = bbox_xyxy_from_json_points(shape_dict["points"])
        if json_bbox is None:
            skipped += 1
            continue

        best_idx = -1
        best_iou = 0.0
        for idx, bbox in enumerate(line_bboxes):
            if idx in claimed or bbox is None:
                continue
            iou = bbox_iou(json_bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            existing_lines[best_idx] = json_line
            line_bboxes[best_idx] = json_bbox
            claimed.add(best_idx)
            replaced += 1
        else:
            existing_lines.append(json_line)
            line_bboxes.append(json_bbox)
            appended += 1

    if not dry_run:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(existing_lines))
            if existing_lines:
                f.write("\n")
        json_path.unlink()

    return "ok", appended, replaced, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 JSON 位姿标注到已有 YOLO-pose txt")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/pose_2026_sz",
        help="YOLO 数据集根目录",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--iou", type=float, default=0.8, help="bbox IoU 阈值，>= 视为改 ID")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写 txt、不删 json")
    args = parser.parse_args()

    ensure_default_specs()

    dataset = args.dataset.resolve()
    data_yaml = dataset / "data.yaml"
    img_dir = dataset / "images" / args.split
    lbl_dir = dataset / "labels" / args.split
    class_names = load_class_names(data_yaml) if data_yaml.is_file() else list(HIER_CLASS_NAMES)

    json_files = find_json_files(img_dir, lbl_dir)
    stats = MergeStats(json_files=len(json_files))

    print(f"数据集: {dataset}")
    print(f"split: {args.split}")
    print(f"类别: {class_names}")
    print(f"IoU 阈值: {args.iou}")
    print(f"JSON 文件: {len(json_files)}")
    if args.dry_run:
        print("[dry-run] 不会写入 txt 或删除 json")

    for json_path in json_files:
        status, app, rep, skip = merge_one_json(
            json_path, img_dir, lbl_dir, class_names, args.iou, args.dry_run
        )
        stats.shapes_total += app + rep + skip
        stats.appended += app
        stats.replaced += rep
        stats.skipped += skip
        if status == "ok":
            if not args.dry_run:
                stats.deleted_json += 1
            print(f"  {json_path.name}: +{app} 替换{rep} 跳过{skip}")
        else:
            print(f"  {json_path.name}: [{status}] 跳过")

    print("---")
    print(f"处理 JSON: {stats.json_files}")
    print(f"新增行: {stats.appended}")
    print(f"替换行: {stats.replaced}")
    print(f"跳过 shape: {stats.skipped}")
    print(f"删除 JSON: {stats.deleted_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
