#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLP ROI 批量推理并按预测类别归档到 idx_* 文件夹.

映射（数据集无 idx_5）: class_0..4→idx_0..4, class_5→idx_6, class_6→idx_7

用法:
    python scripts/mlp_sort_roi.py
    python scripts/mlp_sort_roi.py --dry-run
    python scripts/mlp_sort_roi.py --save-log datasets/mlp_sort_log.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.mlp_classifier import MLPClassifier  # noqa: E402

DEFAULT_MODEL_PATH = _REPO_ROOT / "config" / "mlp_fp32.onnx"
DEFAULT_INPUT_DIR = _REPO_ROOT / "datasets" / "mlp_roi_sz"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "datasets" / "Armor Transform Dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LOW_CONF_FOLDER_NAME = "idx_unknown"
CLASS_ID_PREFIX = "class_"

# Armor Transform Dataset 无 idx_5：模型 class_id -> 目标文件夹
CLASS_ID_TO_FOLDER = {
    0: "idx_0",
    1: "idx_1",
    2: "idx_2",
    3: "idx_3",
    4: "idx_4",
    5: "idx_6",
    6: "idx_7",
}


def list_image_paths(input_dir: Path) -> list[Path]:
    """递归收集输入目录下全部图片。"""
    paths: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        paths.extend(input_dir.rglob(f"*{extension}"))
        paths.extend(input_dir.rglob(f"*{extension.upper()}"))
    return sorted(set(paths))


def load_image_for_predict(image_path: Path):
    """读取 ROI；灰度图转 BGR，供 3 通道模型 predict 使用。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def parse_class_id(class_name: str, labels: list[str]) -> int:
    """从 predict 返回的类别名解析整数 id。"""
    if class_name in labels:
        return labels.index(class_name)
    if class_name.startswith(CLASS_ID_PREFIX):
        try:
            return int(class_name[len(CLASS_ID_PREFIX):])
        except ValueError:
            pass
    if class_name.startswith("idx_"):
        try:
            return int(class_name[4:])
        except ValueError:
            pass
    try:
        return int(class_name)
    except ValueError:
        return -1


def target_folder_name(class_id: int) -> str:
    """模型 class_id -> Armor Transform Dataset 文件夹名。"""
    return CLASS_ID_TO_FOLDER.get(class_id, LOW_CONF_FOLDER_NAME)


def unique_destination(dest_dir: Path, filename: str) -> Path:
    """目标目录已有同名文件时追加 _1/_2… 避免覆盖。"""
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 1
    while True:
        candidate = dest_dir / f"{stem}_{index:04d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def sort_rois(
    classifier: MLPClassifier,
    image_paths: list[Path],
    output_dir: Path,
    *,
    use_copy: bool,
    min_confidence: float,
    dry_run: bool,
) -> dict:
    """逐张推理并归档，返回统计信息。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    failed_read = 0
    results: list[dict] = []

    transfer = shutil.copy2 if use_copy else shutil.move

    for index, image_path in enumerate(image_paths, start=1):
        image = load_image_for_predict(image_path)
        if image is None:
            failed_read += 1
            continue

        class_name, confidence, infer_ms = classifier.predict(image)
        class_id = parse_class_id(class_name, classifier.labels)

        if confidence < min_confidence or class_id < 0:
            folder_name = LOW_CONF_FOLDER_NAME
        else:
            folder_name = target_folder_name(class_id)

        dest_dir = output_dir / folder_name
        dest_path = unique_destination(dest_dir, image_path.name)

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            transfer(str(image_path), str(dest_path))

        stats[folder_name] += 1
        results.append({
            "source": str(image_path),
            "destination": str(dest_path),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(confidence),
            "infer_ms": float(infer_ms),
        })

        if index % 200 == 0:
            print(f"进度 {index}/{len(image_paths)} | 已归档 {index - failed_read}")

    return {
        "total": len(image_paths),
        "archived": len(results),
        "failed_read": failed_read,
        "folder_counts": dict(stats),
        "results": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLP 推理 ROI 并按 idx_* 文件夹归档"
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f"ONNX/PT 模型路径，默认 {DEFAULT_MODEL_PATH}")
    parser.add_argument("--in", dest="input_dir", type=str,
                        default=str(DEFAULT_INPUT_DIR),
                        help=f"ROI 输入目录，默认 {DEFAULT_INPUT_DIR}")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"归档根目录（其下 idx_0/idx_1/…），默认 {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--move", action="store_true",
                        help="移动文件（默认复制，保留源目录）")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="置信度低于此值的样本放入 idx_unknown")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计不写入文件")
    parser.add_argument("--save-log", type=str, default="",
                        help="保存每张图的推理结果 JSON")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.out)

    if not model_path.is_file():
        print(f"[ERROR] 模型不存在: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    image_paths = list_image_paths(input_dir)
    if not image_paths:
        print(f"[ERROR] 未找到图片: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"模型: {model_path}")
    print(f"输入: {input_dir} ({len(image_paths)} 张)")
    print(f"输出: {output_dir}")
    print(f"模式: {'移动' if args.move else '复制'}{' (dry-run)' if args.dry_run else ''}")

    classifier = MLPClassifier(str(model_path))
    print(
        f"输入尺寸 W×H={classifier.input_size} | "
        f"通道={classifier.input_channels} | 类别数={len(classifier.labels)}"
    )

    summary = sort_rois(
        classifier,
        image_paths,
        output_dir,
        use_copy=not args.move,
        min_confidence=args.min_conf,
        dry_run=args.dry_run,
    )

    print("\n归档统计:")
    for folder_name, count in sorted(summary["folder_counts"].items()):
        print(f"  {folder_name}: {count}")
    print(
        f"\n合计: {summary['archived']}/{summary['total']} | "
        f"读图失败: {summary['failed_read']}"
    )

    if args.save_log and not args.dry_run:
        log_path = Path(args.save_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            json.dump(summary, log_file, ensure_ascii=False, indent=2)
        print(f"日志: {log_path}")


if __name__ == "__main__":
    main()
