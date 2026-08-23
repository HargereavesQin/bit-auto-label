# -*- coding: utf-8 -*-
"""打开目录时的数据集布局解析（扁平 JSON / YOLO images+labels）。

判定优先级（避免「根目录 JSON+图」与 sibling images/labels 并存时误判）:
  1. 选中路径本身是 images/ 或 images/{train,val,test} → YOLO
  2. 选中路径根层已有图片 → 扁平（旁路 json/txt）
  3. 否则若存在标准 images/+labels/ → YOLO
  4. 否则扁平
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from core.constants import (
    ANNO_FORMAT_JSON,
    ANNO_FORMAT_YOLO,
    DATASET_LAYOUT_FLAT,
    DATASET_LAYOUT_YOLO,
    IMAGE_FILE_EXTENSIONS,
)
from core.yolo_to_json import (
    YOLO_SPLIT_NAMES,
    load_yolo_class_names_for_dataset,
    normalize_yolo_dataset_dirs,
)


@dataclass
class OpenLayoutResult:
    layout: str
    images_dir: str
    labels_dir: Optional[str]
    dataset_root: str
    image_paths: list[str] = field(default_factory=list)
    annotation_map: dict[str, Optional[str]] = field(default_factory=dict)
    annotation_format: dict[str, Optional[str]] = field(default_factory=dict)
    yolo_label_targets: dict[str, str] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)


def list_top_level_images(directory: str) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, filename)
        for filename in sorted(os.listdir(directory))
        if filename.lower().endswith(IMAGE_FILE_EXTENSIONS)
    ]


def count_top_level_images(directory: str) -> int:
    return len(list_top_level_images(directory))


def sidecar_json_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + ".json"


def sidecar_txt_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + ".txt"


def yolo_label_path_for_image(
    image_path: str, images_dir: str, labels_dir: str
) -> str:
    relative = os.path.relpath(image_path, images_dir)
    return os.path.join(labels_dir, os.path.splitext(relative)[0] + ".txt")


def _path_basename(path: str) -> str:
    return os.path.basename(os.path.abspath(path).rstrip(os.sep))


def _is_explicit_yolo_images_path(selected_path: str) -> bool:
    """用户明确选了 images/ 或 images/split，应走 YOLO。"""
    absolute = os.path.abspath(selected_path)
    base_name = _path_basename(absolute)
    parent = os.path.dirname(absolute)
    if base_name == "images" and os.path.isdir(os.path.join(parent, "labels")):
        return True
    if base_name in YOLO_SPLIT_NAMES:
        grand = os.path.dirname(parent)
        if (
            _path_basename(parent) == "images"
            and os.path.isdir(os.path.join(grand, "labels", base_name))
        ):
            return True
    return False


def _has_standard_yolo_tree(selected_path: str) -> bool:
    absolute = os.path.abspath(selected_path)
    return os.path.isdir(os.path.join(absolute, "images")) and os.path.isdir(
        os.path.join(absolute, "labels")
    )


def _register_annotation(
    image_path: str,
    yolo_txt_target: Optional[str],
    annotation_map: dict,
    annotation_format: dict,
    yolo_label_targets: dict,
) -> None:
    json_path = sidecar_json_path(image_path)
    flat_txt = sidecar_txt_path(image_path)
    if os.path.isfile(json_path):
        annotation_map[image_path] = json_path
        annotation_format[image_path] = ANNO_FORMAT_JSON
    elif yolo_txt_target and os.path.isfile(yolo_txt_target):
        annotation_map[image_path] = yolo_txt_target
        annotation_format[image_path] = ANNO_FORMAT_YOLO
    elif os.path.isfile(flat_txt):
        annotation_map[image_path] = flat_txt
        annotation_format[image_path] = ANNO_FORMAT_YOLO
    else:
        annotation_map[image_path] = None
        annotation_format[image_path] = None
    if yolo_txt_target:
        yolo_label_targets[image_path] = yolo_txt_target


def _collect_flat_layout(selected_path: str) -> OpenLayoutResult:
    absolute = os.path.abspath(selected_path)
    image_paths = list_top_level_images(absolute)
    annotation_map: dict[str, Optional[str]] = {}
    annotation_format: dict[str, Optional[str]] = {}
    yolo_targets: dict[str, str] = {}
    for image_path in image_paths:
        _register_annotation(
            image_path, None, annotation_map, annotation_format, yolo_targets
        )
    return OpenLayoutResult(
        layout=DATASET_LAYOUT_FLAT,
        images_dir=absolute,
        labels_dir=None,
        dataset_root=absolute,
        image_paths=image_paths,
        annotation_map=annotation_map,
        annotation_format=annotation_format,
        yolo_label_targets=yolo_targets,
    )


def _image_subdirs_for_yolo(images_dir: str, labels_dir: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for split_name in YOLO_SPLIT_NAMES:
        image_split = os.path.join(images_dir, split_name)
        label_split = os.path.join(labels_dir, split_name)
        if os.path.isdir(image_split):
            pairs.append((image_split, label_split))
    if pairs:
        return pairs
    return [(images_dir, labels_dir)]


def _collect_yolo_layout(selected_path: str) -> OpenLayoutResult:
    absolute = os.path.abspath(selected_path)
    images_dir, labels_dir, dataset_root = normalize_yolo_dataset_dirs(absolute, "")
    class_names, _ = load_yolo_class_names_for_dataset(images_dir, labels_dir)

    image_paths: list[str] = []
    annotation_map: dict[str, Optional[str]] = {}
    annotation_format: dict[str, Optional[str]] = {}
    yolo_targets: dict[str, str] = {}

    for image_subdir, _label_subdir in _image_subdirs_for_yolo(images_dir, labels_dir):
        for image_path in list_top_level_images(image_subdir):
            image_paths.append(image_path)
            target_txt = yolo_label_path_for_image(image_path, images_dir, labels_dir)
            _register_annotation(
                image_path,
                target_txt,
                annotation_map,
                annotation_format,
                yolo_targets,
            )

    return OpenLayoutResult(
        layout=DATASET_LAYOUT_YOLO,
        images_dir=images_dir,
        labels_dir=labels_dir,
        dataset_root=dataset_root,
        image_paths=image_paths,
        annotation_map=annotation_map,
        annotation_format=annotation_format,
        yolo_label_targets=yolo_targets,
        class_names=list(class_names or []),
    )


def resolve_open_layout(selected_path: str) -> OpenLayoutResult:
    if not selected_path or not os.path.isdir(selected_path):
        return OpenLayoutResult(
            layout=DATASET_LAYOUT_FLAT,
            images_dir=selected_path or "",
            labels_dir=None,
            dataset_root=selected_path or "",
        )

    # 用户明确点进 images/ 或 images/train → YOLO
    if _is_explicit_yolo_images_path(selected_path):
        result = _collect_yolo_layout(selected_path)
        if result.image_paths:
            return result

    # 根层已有图片（含与 images/labels 同级的极端情况）→ 扁平优先
    if count_top_level_images(selected_path) > 0:
        return _collect_flat_layout(selected_path)

    # 根目录无顶层图，但有标准 YOLO 树 → YOLO
    if _has_standard_yolo_tree(selected_path):
        result = _collect_yolo_layout(selected_path)
        if result.image_paths:
            return result

    return _collect_flat_layout(selected_path)
