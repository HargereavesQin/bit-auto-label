# -*- coding: utf-8 -*-
"""YOLO detect/pose 标签 → LabelMe JSON 转换与数据集扫描。"""
from __future__ import annotations

import os
import json
from typing import Optional

import yaml

from core.constants import SHAPE_POSE, SHAPE_RECTANGLE, SHAPE_POINT
from core.exporter import Exporter

def _search_paths(*dirs: str) -> list[str]:
    seen = set()
    out = []
    for d in dirs:
        if not d:
            continue
        d = os.path.abspath(d)
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
        parent = os.path.dirname(d)
        if parent and parent not in seen:
            seen.add(parent)
            out.append(parent)
    return out


YOLO_SPLIT_NAMES = ("train", "val", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
YAML_META_FILENAMES = ("data.yaml", "data.yml", "dataset.yaml")


def _parse_names_from_yaml_data(data: dict) -> list[str]:
    names_map = data.get("names")
    if isinstance(names_map, dict):
        ordered = []
        for i in sorted(int(k) for k in names_map.keys()):
            ordered.append(str(names_map[i]))
        return ordered
    if isinstance(names_map, list) and names_map:
        return [str(n) for n in names_map]
    return []


def find_dataset_meta_file(root_dir: str) -> Optional[str]:
    """在数据集根目录（与 images/ labels/ 同级）查找 classes.txt 或 data.yaml。"""
    if not root_dir or not os.path.isdir(root_dir):
        return None
    root_dir = os.path.abspath(root_dir)
    classes_txt = os.path.join(root_dir, "classes.txt")
    if os.path.isfile(classes_txt):
        return classes_txt
    for name in YAML_META_FILENAMES:
        yaml_path = os.path.join(root_dir, name)
        if os.path.isfile(yaml_path):
            return yaml_path
    return None


def _has_standard_yolo_layout(root: str) -> bool:
    return (
        os.path.isdir(os.path.join(root, "images"))
        and os.path.isdir(os.path.join(root, "labels"))
    )


def _infer_dataset_root(img_abs: str, lbl_abs: str) -> str:
    """从 images/labels 路径推断数据集根目录（避免与 normalize 互相递归）。"""
    img_base = os.path.basename(img_abs.rstrip(os.sep))
    lbl_base = os.path.basename(lbl_abs.rstrip(os.sep)) if lbl_abs else ""

    if img_base == "images":
        return os.path.dirname(img_abs)
    if lbl_base == "labels":
        return os.path.dirname(lbl_abs)

    for start in (img_abs, lbl_abs):
        if not start:
            continue
        cur = os.path.abspath(start)
        for _ in range(8):
            if _has_standard_yolo_layout(cur):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    if lbl_abs and lbl_abs != img_abs:
        try:
            return os.path.commonpath([img_abs, lbl_abs])
        except ValueError:
            return img_abs
    return img_abs


def normalize_yolo_dataset_dirs(img_dir: str, label_dir: str) -> tuple[str, str, str]:
    """
    规范化 YOLO 目录，支持用户选择数据集根目录或 images/ + labels/。

    Returns:
        (images_dir, labels_dir, dataset_root)
    """
    img_abs = os.path.abspath(img_dir)
    lbl_abs = os.path.abspath(label_dir) if label_dir else ""

    # 图片目录选的是数据集根（含 images/ + labels/ + 可选 data.yaml）
    if _has_standard_yolo_layout(img_abs):
        root = img_abs
        if not lbl_abs or lbl_abs == img_abs:
            return (
                os.path.join(root, "images"),
                os.path.join(root, "labels"),
                root,
            )

    if lbl_abs and _has_standard_yolo_layout(lbl_abs):
        root = lbl_abs
        return (
            os.path.join(root, "images"),
            os.path.join(root, "labels"),
            root,
        )

    # 标准：images/ + labels/
    if os.path.basename(img_abs.rstrip(os.sep)) == "images":
        root = os.path.dirname(img_abs)
        sibling_lbl = os.path.join(root, "labels")
        if lbl_abs and os.path.basename(lbl_abs.rstrip(os.sep)) == "labels":
            return img_abs, lbl_abs, root
        if os.path.isdir(sibling_lbl):
            return img_abs, sibling_lbl, root

    root = _infer_dataset_root(img_abs, lbl_abs or img_abs)
    return img_abs, lbl_abs or img_abs, root


def load_yolo_class_names(*search_dirs: str) -> list[str]:
    """从 classes.txt 或 data.yaml 读取类别名列表，按 class_id 排序。"""
    for base in _search_paths(*search_dirs):
        meta = find_dataset_meta_file(base)
        if not meta:
            continue
        if meta.endswith(".txt"):
            names = []
            with open(meta, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        names.append(line)
            if names:
                return names
        else:
            with open(meta, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            names = _parse_names_from_yaml_data(data)
            if names:
                return names

    return []


def load_yolo_class_names_for_dataset(img_dir: str, label_dir: str) -> tuple[list[str], Optional[str]]:
    """
    从与 images/ labels/ 同级的 classes.txt / data.yaml 加载类别名。

    Returns:
        (class_names, meta_file_path_or_none)
    """
    _, _, root = normalize_yolo_dataset_dirs(img_dir, label_dir)
    meta = find_dataset_meta_file(root)
    if meta:
        if meta.endswith(".txt"):
            names = []
            with open(meta, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        names.append(line)
            if names:
                return names, meta
        else:
            with open(meta, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            names = _parse_names_from_yaml_data(data)
            if names:
                return names, meta

    # 回退：在父级路径中搜索
    img_n, lbl_n, _ = normalize_yolo_dataset_dirs(img_dir, label_dir)
    names = load_yolo_class_names(root, img_n, lbl_n)
    return names, None


def resolve_yolo_dataset_root(img_dir: str, label_dir: str) -> str:
    """推断 YOLO 数据集根目录（与 images/ labels/ data.yaml 同级）。"""
    img_abs = os.path.abspath(img_dir)
    lbl_abs = os.path.abspath(label_dir) if label_dir else img_abs
    return _infer_dataset_root(img_abs, lbl_abs)


def _pair_images_and_labels(img_dir: str, label_dir: str) -> list[tuple[str, str]]:
    """配对同一 split 下的图片与 TXT 标签。"""
    pairs: list[tuple[str, str]] = []
    if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
        return pairs
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        base = os.path.splitext(fn)[0]
        txt_path = os.path.join(label_dir, base + ".txt")
        if os.path.isfile(txt_path):
            pairs.append((os.path.join(img_dir, fn), txt_path))
    return pairs


def collect_yolo_split_pairs(img_dir: str, label_dir: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    扫描 YOLO 数据集目录，自动识别 train/val/test。

    支持布局:
      - 数据集根目录（含 images/ labels/ data.yaml）
      - images/{train,val,test} + labels/{train,val,test}  （标准导出）
      - 用户直接选 images/train + labels/train
      - 扁平目录（img/label 同层无 split）→ split 名为 all

    Returns:
        [(split_name, [(img_path, txt_path), ...]), ...]
    """
    img_dir, label_dir, _ = normalize_yolo_dataset_dirs(img_dir, label_dir)
    img_dir = os.path.abspath(img_dir)
    label_dir = os.path.abspath(label_dir)
    img_base = os.path.basename(img_dir.rstrip(os.sep))
    lbl_base = os.path.basename(label_dir.rstrip(os.sep))

    # images/ + labels/ 标准结构
    if img_base == "images" and lbl_base == "labels":
        splits: list[tuple[str, list[tuple[str, str]]]] = []
        for split in YOLO_SPLIT_NAMES:
            idir = os.path.join(img_dir, split)
            ldir = os.path.join(label_dir, split)
            if os.path.isdir(idir) and os.path.isdir(ldir):
                pairs = _pair_images_and_labels(idir, ldir)
                if pairs:
                    splits.append((split, pairs))
        if splits:
            return splits

    # 用户选了某个 split 子目录，如 images/train + labels/train
    if img_base in YOLO_SPLIT_NAMES and lbl_base == img_base:
        pairs = _pair_images_and_labels(img_dir, label_dir)
        return [(img_base, pairs)] if pairs else []

    # 目录下含 train/val/test 子文件夹
    splits = []
    for split in YOLO_SPLIT_NAMES:
        idir = os.path.join(img_dir, split)
        ldir = os.path.join(label_dir, split)
        if os.path.isdir(idir) and os.path.isdir(ldir):
            pairs = _pair_images_and_labels(idir, ldir)
            if pairs:
                splits.append((split, pairs))
    if splits:
        return splits

    # 扁平 img + label 目录
    pairs = _pair_images_and_labels(img_dir, label_dir)
    if pairs:
        return [("all", pairs)]
    return []


def infer_class_ids_from_txt_paths(txt_paths: list[str]) -> set[int]:
    """从 YOLO 标签文件扫描出现的 class_id。"""
    ids: set[int] = set()
    for txt_path in txt_paths:
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        ids.add(int(float(parts[0])))
                    except ValueError:
                        continue
    return ids


def build_class_names_from_ids(class_ids: set[int]) -> list[str]:
    """根据最大 class_id 生成 class_0, class_1, ... 占位名。"""
    if not class_ids:
        return []
    max_id = max(class_ids)
    return [f"class_{i}" for i in range(max_id + 1)]


def resolve_yolo_class_names(
    img_dir: str,
    label_dir: str,
    txt_paths: list[str],
) -> tuple[list[str], str, Optional[str]]:
    """
    解析类别名：优先与 images/labels 同级的 data.yaml / classes.txt，否则从标签推断。

    Returns:
        (class_names, source, meta_file)  source 为 file | inferred | none
    """
    names, meta = load_yolo_class_names_for_dataset(img_dir, label_dir)
    if names:
        return names, "file", meta

    ids = infer_class_ids_from_txt_paths(txt_paths)
    if ids:
        return build_class_names_from_ids(ids), "inferred", None

    return [], "none", None


def iter_txt_paths_from_splits(
    splits: list[tuple[str, list[tuple[str, str]]]],
) -> list[str]:
    paths = []
    for _, pairs in splits:
        for _, txt_path in pairs:
            paths.append(txt_path)
    return paths


def _resolve_kpt_names(label: str, kpt_count: int, pose_config=None) -> list[str]:
    if pose_config is not None:
        names = pose_config.get_kpt_names(label)
        if names and len(names) >= kpt_count:
            return names[:kpt_count]
        if names:
            return names + [f"kpt_{i}" for i in range(len(names), kpt_count)]
    return [f"kpt_{i}" for i in range(kpt_count)]


def is_pose_line(parts: list[str]) -> bool:
    return len(parts) > 5 and (len(parts) - 5) % 3 == 0


def is_point_det(bw: float, bh: float, threshold: float = 0.025) -> bool:
    return bw <= threshold and bh <= threshold


def parse_yolo_line(
    line: str,
    img_w: int,
    img_h: int,
    class_names: list[str],
    *,
    keep_kpt_indices: Optional[list[int]] = None,
    pose_config=None,
) -> Optional[dict]:
    """解析单行 YOLO 标签为 JSON shape dict。"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    if class_id < 0 or class_id >= len(class_names):
        return None
    label = class_names[class_id]

    cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h

    if is_pose_line(parts):
        effective = effective_kpt_count_from_yolo_parts(parts)
        kpts = []
        for i in range(effective):
            kx = float(parts[5 + i * 3]) * img_w
            ky = float(parts[6 + i * 3]) * img_h
            kv = int(float(parts[7 + i * 3]))
            kpts.append([kx, ky, kv])

        all_names = _resolve_kpt_names(label, effective, pose_config)
        if keep_kpt_indices is not None:
            indices = [i for i in keep_kpt_indices if 0 <= i < effective]
            kpts = [kpts[i] for i in indices]
            all_names = [all_names[i] for i in indices]

        return {
            "label": label,
            "type": SHAPE_POSE,
            "points": [[x1, y1], [x2, y2]],
            "keypoints": kpts,
            "kpt_names": all_names,
        }

    if is_point_det(bw, bh):
        px = cx * img_w
        py = cy * img_h
        return {
            "label": label,
            "type": SHAPE_POINT,
            "points": [[px, py]],
        }

    return {
        "label": label,
        "type": SHAPE_RECTANGLE,
        "points": [[x1, y1], [x2, y2]],
    }


def yolo_txt_to_shapes(
    txt_path: str,
    img_w: int,
    img_h: int,
    class_names: list[str],
    *,
    keep_kpt_indices: Optional[list[int]] = None,
    pose_config=None,
) -> list[dict]:
    if not os.path.isfile(txt_path):
        return []
    shapes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            shape = parse_yolo_line(
                line, img_w, img_h, class_names,
                keep_kpt_indices=keep_kpt_indices,
                pose_config=pose_config,
            )
            if shape:
                shapes.append(shape)
    return shapes


def effective_kpt_count_from_yolo_parts(parts: list[str]) -> int:
    """统计 YOLO pose 行中有效关键点数（忽略末尾 vis=0 且坐标为 0 的 padding）。"""
    if not is_pose_line(parts):
        return 0
    slot_count = (len(parts) - 5) // 3
    last_valid = 0
    for i in range(slot_count):
        kx = float(parts[5 + i * 3])
        ky = float(parts[6 + i * 3])
        kv = int(float(parts[7 + i * 3]))
        if kv > 0 or abs(kx) > 1e-8 or abs(ky) > 1e-8:
            last_valid = i + 1
    return last_valid if last_valid > 0 else slot_count


def effective_kpt_count_from_keypoints(kpts: list) -> int:
    """统计 JSON 位姿 shape 中有效关键点数。"""
    last_valid = 0
    for i, k in enumerate(kpts):
        if len(k) >= 3:
            kv = int(k[2])
            kx, ky = float(k[0]), float(k[1])
            if kv > 0 or abs(kx) > 1e-6 or abs(ky) > 1e-6:
                last_valid = i + 1
        elif len(k) >= 2:
            if abs(float(k[0])) > 1e-6 or abs(float(k[1])) > 1e-6:
                last_valid = i + 1
    return last_valid


def scan_class_pose_stats(
    txt_paths: list[str],
    class_names: list[str],
) -> dict[str, dict]:
    """按类别统计 detect/pose 数量及最大有效关键点数。"""
    stats: dict[str, dict] = {}
    for txt_path in txt_paths:
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if cid < 0 or cid >= len(class_names):
                    continue
                name = class_names[cid]
                if name not in stats:
                    stats[name] = {"max_kpt": 0, "pose_count": 0, "det_count": 0}
                if is_pose_line(parts):
                    n = effective_kpt_count_from_yolo_parts(parts)
                    stats[name]["max_kpt"] = max(stats[name]["max_kpt"], n)
                    stats[name]["pose_count"] += 1
                else:
                    stats[name]["det_count"] += 1
    return stats


def scan_json_class_pose_stats(json_paths: list[str]) -> dict[str, dict]:
    """从 LabelMe JSON 按类别统计位姿/检测及有效关键点数。"""
    stats: dict[str, dict] = {}
    for json_path in json_paths:
        if not os.path.isfile(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("shapes", []):
            label = s.get("label", "")
            if not label:
                continue
            if label not in stats:
                stats[label] = {"max_kpt": 0, "pose_count": 0, "det_count": 0}
            st = s.get("shape_type", "")
            if st == "pose" or s.get("keypoints"):
                kpts = s.get("keypoints", [])
                n = effective_kpt_count_from_keypoints(kpts)
                stats[label]["max_kpt"] = max(stats[label]["max_kpt"], n)
                stats[label]["pose_count"] += 1
            else:
                stats[label]["det_count"] += 1
    return stats


def collect_json_paths(img_dir: str, ann_dir: str) -> list[str]:
    """收集 JSON 路径，支持 train/val/test 子目录或扁平目录。"""
    ann_dir = ann_dir or img_dir
    ann_dir = os.path.abspath(ann_dir)
    paths: list[str] = []
    for split in YOLO_SPLIT_NAMES:
        sd = os.path.join(ann_dir, split)
        if os.path.isdir(sd):
            for fn in sorted(os.listdir(sd)):
                if fn.lower().endswith(".json"):
                    paths.append(os.path.join(sd, fn))
    if paths:
        return paths
    if os.path.isdir(ann_dir):
        for fn in sorted(os.listdir(ann_dir)):
            if fn.lower().endswith(".json"):
                paths.append(os.path.join(ann_dir, fn))
    return paths


def _find_image_for_base(img_dirs: list[str], base: str) -> Optional[str]:
    """在若干目录中按 basename 查找图片。"""
    for directory in img_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for ext in IMG_EXTS:
            candidate = os.path.join(directory, base + ext)
            if os.path.isfile(candidate):
                return candidate
    return None


def collect_json_split_pairs(
    img_dir: str, ann_dir: str,
) -> list[tuple[str, list[tuple[Optional[str], str]]]]:
    """
    扫描 JSON 标注目录，配对图片路径。

    Returns:
        [(split_name, [(img_path_or_none, json_path), ...]), ...]
    """
    ann_dir = os.path.abspath(ann_dir or img_dir)
    img_dir = os.path.abspath(img_dir)
    ann_base = os.path.basename(ann_dir.rstrip(os.sep))

    splits: list[tuple[str, list[tuple[Optional[str], str]]]] = []
    for split in YOLO_SPLIT_NAMES:
        sd_ann = os.path.join(ann_dir, split)
        if not os.path.isdir(sd_ann):
            continue
        sd_img = os.path.join(img_dir, split)
        img_search = [sd_img, img_dir] if os.path.isdir(sd_img) else [img_dir]
        pairs: list[tuple[Optional[str], str]] = []
        for fn in sorted(os.listdir(sd_ann)):
            if not fn.lower().endswith(".json"):
                continue
            json_path = os.path.join(sd_ann, fn)
            base = os.path.splitext(fn)[0]
            img_path = _find_image_for_base(img_search, base)
            pairs.append((img_path, json_path))
        if pairs:
            splits.append((split, pairs))

    if splits:
        return splits

    if ann_base in YOLO_SPLIT_NAMES:
        pairs = []
        for fn in sorted(os.listdir(ann_dir)):
            if not fn.lower().endswith(".json"):
                continue
            json_path = os.path.join(ann_dir, fn)
            base = os.path.splitext(fn)[0]
            parent_img = os.path.join(os.path.dirname(ann_dir), "images", ann_base)
            img_path = _find_image_for_base([parent_img, img_dir], base)
            pairs.append((img_path, json_path))
        return [(ann_base, pairs)] if pairs else []

    pairs = []
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.lower().endswith(".json"):
            continue
        json_path = os.path.join(ann_dir, fn)
        base = os.path.splitext(fn)[0]
        img_path = _find_image_for_base([img_dir, ann_dir], base)
        pairs.append((img_path, json_path))
    return [("all", pairs)] if pairs else []


def load_yaml_kpt_shape(meta_file: Optional[str]) -> Optional[int]:
    """从 data.yaml 读取 kpt_shape[0]。"""
    if not meta_file or not meta_file.endswith((".yaml", ".yml")):
        return None
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        kpt_shape = data.get("kpt_shape")
        if isinstance(kpt_shape, (list, tuple)) and kpt_shape:
            return int(kpt_shape[0])
    except Exception:
        pass
    return None


def reconcile_pose_config(
    pose_config,
    class_stats: dict[str, dict],
    *,
    yaml_kpt_shape: Optional[int] = None,
    remove_absent: bool = True,
) -> dict:
    """
    与数据集对齐位姿类别：新增缺失、扩展关键点、删除数据集中不存在的类别。

    Returns:
        {"added": [...], "removed": [...], "updated": [...]}
    """
    dataset_pose_names = {
        name for name, st in class_stats.items() if st["pose_count"] > 0
    }

    removed: list[str] = []
    if remove_absent:
        for name in list(pose_config.get_category_names()):
            if name not in dataset_pose_names:
                pose_config.remove_category(name)
                removed.append(name)

    added: list[str] = []
    updated: list[str] = []
    for name in sorted(dataset_pose_names):
        st = class_stats[name]
        kpt_count = st["max_kpt"]
        if kpt_count <= 0 and yaml_kpt_shape:
            kpt_count = yaml_kpt_shape
        if kpt_count <= 0:
            continue

        existing = pose_config.get_category(name)
        if existing:
            old_count = pose_config.get_kpt_count(name)
            if kpt_count != old_count:
                kpt_names = pose_config.get_kpt_names(name)
                if kpt_count > old_count:
                    while len(kpt_names) < kpt_count:
                        kpt_names.append(f"kpt{len(kpt_names)}")
                else:
                    kpt_names = kpt_names[:kpt_count]
                pose_config.add_category(name, kpt_names)
                updated.append(name)
            continue

        pose_config.add_category(name, [f"kpt{i}" for i in range(kpt_count)])
        added.append(name)

    return {"added": added, "removed": removed, "updated": updated}


def sync_pose_config_from_dataset(
    pose_config,
    class_stats: dict[str, dict],
    *,
    yaml_kpt_shape: Optional[int] = None,
) -> list[str]:
    """兼容旧接口：仅新增/扩展，不删除。"""
    result = reconcile_pose_config(
        pose_config, class_stats,
        yaml_kpt_shape=yaml_kpt_shape, remove_absent=False,
    )
    return result["added"]


def build_present_class_names(
    class_names: list[str],
    class_stats: dict[str, dict],
) -> list[str]:
    """保留数据集中实际出现的类别，维持原 class_id 顺序。"""
    present = {
        n for n, st in class_stats.items()
        if st["pose_count"] + st["det_count"] > 0
    }
    ordered = [n for n in class_names if n in present]
    for n in sorted(present):
        if n not in ordered:
            ordered.append(n)
    return ordered


def organize_dataset_categories(
    pose_config,
    class_stats: dict[str, dict],
    class_names: list[str],
    target_dirs: list[str],
    *,
    yaml_kpt_shape: Optional[int] = None,
    save_global_path: Optional[str] = None,
) -> dict:
    """
    一键整理：同步 pose 类别、删除多余项、写出 classes.txt / pose_classes.json。

    Returns:
        reconcile 结果 + filtered class_names
    """
    result = reconcile_pose_config(
        pose_config, class_stats,
        yaml_kpt_shape=yaml_kpt_shape, remove_absent=True,
    )
    filtered_names = build_present_class_names(class_names, class_stats)
    output_pose = build_output_pose_config(pose_config, class_stats)

    if save_global_path:
        pose_config.save(save_global_path)

    for base in target_dirs:
        if not base:
            continue
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "classes.txt"), "w", encoding="utf-8") as f:
            for name in filtered_names:
                f.write(f"{name}\n")
        if output_pose.categories:
            payload = {"categories": dict(output_pose.categories)}
            with open(os.path.join(base, "pose_classes.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    result["class_names"] = filtered_names
    result["pose_categories"] = list(output_pose.get_category_names())
    return result


def build_output_pose_config(pose_config, class_stats: dict[str, dict]):
    """构建仅含数据集中位姿类别的 PoseConfig 副本，用于写出到输出目录。"""
    from core.pose_config import PoseConfig

    out = PoseConfig()
    for name, st in class_stats.items():
        if st["pose_count"] <= 0:
            continue
        cat = pose_config.get_category(name)
        if cat:
            out.add_category(name, list(cat["keypoints"]))
        elif st["max_kpt"] > 0:
            out.add_category(name, [f"kpt{i}" for i in range(st["max_kpt"])])
    return out


def write_dataset_meta_files(
    out_dir: str,
    split_names: list[str],
    class_names: list[str],
    pose_config,
) -> None:
    """写出 classes.txt 与 pose_classes.json（输出根目录及各 split 子目录）。"""
    pose_payload = {"categories": dict(pose_config.categories)}
    targets = {out_dir}
    for split in split_names:
        if split != "all":
            targets.add(os.path.join(out_dir, split))

    for base in targets:
        os.makedirs(base, exist_ok=True)
        classes_path = os.path.join(base, "classes.txt")
        with open(classes_path, "w", encoding="utf-8") as f:
            for name in class_names:
                f.write(f"{name}\n")
        if pose_payload["categories"]:
            pose_path = os.path.join(base, "pose_classes.json")
            with open(pose_path, "w", encoding="utf-8") as f:
                json.dump(pose_payload, f, ensure_ascii=False, indent=2)


def scan_yolo_dataset(txt_paths: list[str], class_names: list[str]) -> dict:
    """扫描 YOLO 标签，返回任务类型、关键点数量、类别使用统计。"""
    pose_lines = 0
    det_lines = 0
    kpt_counts: dict[int, int] = {}
    class_hits: dict[int, int] = {}

    for txt_path in txt_paths:
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                class_hits[cid] = class_hits.get(cid, 0) + 1
                if is_pose_line(parts):
                    pose_lines += 1
                    n = effective_kpt_count_from_yolo_parts(parts)
                    kpt_counts[n] = kpt_counts.get(n, 0) + 1
                else:
                    det_lines += 1

    if pose_lines and det_lines:
        task = "mixed"
    elif pose_lines:
        task = "pose"
    else:
        task = "detect"

    max_kpt = max(kpt_counts.keys()) if kpt_counts else 0
    used_classes = []
    for cid in sorted(class_hits.keys()):
        name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        used_classes.append({"id": cid, "name": name, "count": class_hits[cid]})

    class_stats = scan_class_pose_stats(txt_paths, class_names)
    max_kpt = max(
        (st["max_kpt"] for st in class_stats.values()),
        default=max(kpt_counts.keys()) if kpt_counts else 0,
    )

    return {
        "task": task,
        "pose_lines": pose_lines,
        "det_lines": det_lines,
        "kpt_counts": kpt_counts,
        "max_kpt": max_kpt,
        "used_classes": used_classes,
        "class_names": class_names,
        "class_stats": class_stats,
    }


def _is_pose_shape(shape: dict) -> bool:
    """internal dict（type=pose）或 LabelMe dict（shape_type=pose / 含 keypoints）。"""
    if shape.get("type") == SHAPE_POSE:
        return True
    if shape.get("shape_type") == SHAPE_POSE:
        return True
    return bool(shape.get("keypoints"))


def _kpt_is_valid(kp) -> bool:
    if len(kp) < 2:
        return False
    if len(kp) >= 3 and int(kp[2]) == 0:
        return False
    kx, ky = float(kp[0]), float(kp[1])
    return abs(kx) > 1e-6 or abs(ky) > 1e-6


def bbox_from_kpt_indices(
    keypoints: list,
    indices: list[int],
) -> Optional[tuple[float, float, float, float]]:
    """从所选关键点序号计算 axis-aligned 外接框 (x1, y1, x2, y2)。"""
    pts: list[tuple[float, float]] = []
    for i in indices:
        if not (0 <= i < len(keypoints)):
            continue
        kp = keypoints[i]
        if _kpt_is_valid(kp):
            pts.append((float(kp[0]), float(kp[1])))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def pose_shape_to_detect(shape: dict, indices: list[int]) -> Optional[dict]:
    """单个 pose 实例 → rectangle detect dict（internal 格式）。"""
    kpts = shape.get("keypoints", [])
    bbox = bbox_from_kpt_indices(kpts, indices)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "label": shape.get("label", ""),
        "type": SHAPE_RECTANGLE,
        "points": [[x1, y1], [x2, y2]],
    }


def convert_pose_shapes_to_detect_only(
    shapes: list[dict],
    indices: list[int],
) -> list[dict]:
    """仅将 pose 实例转为外接 detect 框，丢弃非位姿标注。"""
    out: list[dict] = []
    for s in shapes:
        if not _is_pose_shape(s):
            continue
        det = pose_shape_to_detect(s, indices)
        if det:
            out.append(det)
    return out


def write_classes_txt(out_dir: str, class_names: list[str]) -> None:
    """写出 classes.txt（检测数据集，不含 pose 配置）。"""
    os.makedirs(out_dir, exist_ok=True)
    classes_path = os.path.join(out_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")


def filter_pose_shapes(shapes: list[dict], keep_indices: list[int]) -> list[dict]:
    """对已有 shape dict 列表做关键点序号筛选（用于 JSON 批量处理）。"""
    out = []
    for s in shapes:
        if s.get("type") != SHAPE_POSE:
            out.append(s)
            continue
        kpts = s.get("keypoints", [])
        names = s.get("kpt_names") or [f"kpt_{i}" for i in range(len(kpts))]
        indices = [i for i in keep_indices if 0 <= i < len(kpts)]
        ns = dict(s)
        ns["keypoints"] = [kpts[i] for i in indices]
        ns["kpt_names"] = [names[i] if i < len(names) else f"kpt_{i}" for i in indices]
        out.append(ns)
    return out


def save_labelme_json(
    json_path: str,
    image_path: str,
    img_w: int,
    img_h: int,
    shapes: list[dict],
) -> None:
    Exporter.save_json(json_path, image_path, img_w, img_h, shapes)


def filter_pose_json_shapes(shapes: list[dict], keep_indices: list[int]) -> list[dict]:
    """对 LabelMe JSON shapes 列表做关键点序号筛选。"""
    out = []
    for s in shapes:
        st = s.get("shape_type", "")
        if st != "pose" and not s.get("keypoints"):
            out.append(s)
            continue
        kpts = s.get("keypoints", [])
        names = s.get("kpt_names") or [f"kpt_{i}" for i in range(len(kpts))]
        indices = [i for i in keep_indices if 0 <= i < len(kpts)]
        ns = dict(s)
        ns["keypoints"] = [kpts[i] for i in indices]
        ns["kpt_names"] = [names[i] if i < len(names) else f"kpt_{i}" for i in indices]
        out.append(ns)
    return out


def scan_json_pose_dataset(json_paths: list[str]) -> dict:
    """扫描 JSON 位姿标注，统计类别与关键点数量。"""
    kpt_counts: dict[int, int] = {}
    class_hits: dict[str, int] = {}
    pose_shapes = 0
    other_shapes = 0

    for json_path in json_paths:
        if not os.path.isfile(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("shapes", []):
            label = s.get("label", "")
            if label:
                class_hits[label] = class_hits.get(label, 0) + 1
            st = s.get("shape_type", "")
            if st == "pose" or s.get("keypoints"):
                pose_shapes += 1
                kpts = s.get("keypoints", [])
                n = effective_kpt_count_from_keypoints(kpts) if kpts else 0
                if n > 0:
                    kpt_counts[n] = kpt_counts.get(n, 0) + 1
            else:
                other_shapes += 1

    class_stats = scan_json_class_pose_stats(json_paths)
    max_kpt = max(
        (st["max_kpt"] for st in class_stats.values()),
        default=max(kpt_counts.keys()) if kpt_counts else 0,
    )
    used_classes = [
        {"id": i, "name": name, "count": class_hits[name]}
        for i, name in enumerate(sorted(class_hits.keys()))
    ]
    return {
        "task": "pose" if pose_shapes else "detect",
        "pose_lines": pose_shapes,
        "det_lines": other_shapes,
        "kpt_counts": kpt_counts,
        "max_kpt": max_kpt,
        "used_classes": used_classes,
        "class_names": sorted(class_hits.keys()),
        "class_stats": class_stats,
    }


def purge_json_annotations(
    json_paths: list[str],
    *,
    keep_labels: Optional[set[str]] = None,
    keep_pose_kpt_count: Optional[int] = None,
) -> dict:
    """原地清理 LabelMe JSON 标注。

    Args:
        json_paths: JSON 文件路径列表
        keep_labels: 若给定，删除 label 不在此集合中的所有 shape（任意 shape_type）
        keep_pose_kpt_count: 若给定，删除 pose/hierarchy_pose 中关键点数 != 该值的实例
            （非 pose 形状保留；无 keypoints 的 pose 也删除）

    Returns:
        统计: files_touched, shapes_before, shapes_after, removed_by_label,
        removed_by_kpt, removed_labels
    """
    from core.constants import SHAPE_POSE, SHAPE_HIERARCHY

    pose_types = {SHAPE_POSE, SHAPE_HIERARCHY}
    files_touched = 0
    shapes_before = 0
    shapes_after = 0
    removed_by_label = 0
    removed_by_kpt = 0
    removed_labels: dict[str, int] = {}

    for json_path in json_paths:
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            continue
        shapes_before += len(shapes)
        kept = []
        changed = False
        for s in shapes:
            label = s.get("label") or ""
            st = s.get("shape_type", "")
            # 1) 非保留类名 → 删除
            if keep_labels is not None and label not in keep_labels:
                removed_by_label += 1
                removed_labels[label or "(空)"] = removed_labels.get(label or "(空)", 0) + 1
                changed = True
                continue
            # 2) pose 关键点数不符 → 删除
            if keep_pose_kpt_count is not None and (st in pose_types or s.get("keypoints")):
                kpts = s.get("keypoints") or []
                n = len(kpts)
                if n != keep_pose_kpt_count:
                    removed_by_kpt += 1
                    changed = True
                    continue
            kept.append(s)
        shapes_after += len(kept)
        if changed:
            data["shapes"] = kept
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            files_touched += 1

    return {
        "files_touched": files_touched,
        "shapes_before": shapes_before,
        "shapes_after": shapes_after,
        "removed_by_label": removed_by_label,
        "removed_by_kpt": removed_by_kpt,
        "removed_labels": removed_labels,
    }
