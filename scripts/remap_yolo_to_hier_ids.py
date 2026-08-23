#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 YOLO 导出的「按出现类名排序」的 class_id 纠正为 RM 固定 14 类顺序。

导出工具 TO_YOLO_POSE 曾用 sorted(unique_labels) 压成 0..N-1，例如:
  B1=0, B2=1, B4=2, BG=3, ...
正确顺序 (与 cpp / hierarchical_paths 对齐):
  BG=0, B1=1, B2=2, B3=3, B4=4, BO=5, BB=6,
  RG=7, R1=8, R2=9, R3=10, R4=11, RO=12, RB=13

用法:
  python scripts/remap_yolo_to_hier_ids.py datasets/final_shenzhen
  python scripts/remap_yolo_to_hier_ids.py datasets/final_shenzhen --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 允许直接 `python scripts/...` 运行
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.constants import HIER_CLASS_NAMES, HIER_NUM_CLASSES

TARGET_NAMES = list(HIER_CLASS_NAMES)
TARGET_NAME_TO_ID = {n: i for i, n in enumerate(TARGET_NAMES)}


def _parse_data_yaml_names(yaml_path: Path) -> list[str] | None:
    if not yaml_path.is_file():
        return None
    text = yaml_path.read_text(encoding="utf-8")
    # 支持:
    #   names:
    #     0: B1
    #   names: [B1, B2, ...]
    #   names: {0: B1, 1: B2}
    names_by_id: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^names\s*:", stripped):
            in_names = True
            # inline list / dict on same line
            rest = stripped.split(":", 1)[1].strip()
            if rest.startswith("["):
                inner = rest.strip("[]")
                return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
            if rest.startswith("{"):
                for m in re.finditer(r"(\d+)\s*:\s*['\"]?([A-Za-z0-9_]+)['\"]?", rest):
                    names_by_id[int(m.group(1))] = m.group(2)
                if names_by_id:
                    return [names_by_id[i] for i in sorted(names_by_id)]
            continue
        if in_names:
            # 缩进键值  0: B1
            m = re.match(r"^(\d+)\s*:\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*$", stripped)
            if m:
                names_by_id[int(m.group(1))] = m.group(2)
                continue
            # 离开 names 块
            if not raw.startswith((" ", "\t")) and ":" in stripped:
                break
    if names_by_id:
        return [names_by_id[i] for i in sorted(names_by_id)]
    return None


def _parse_classes_txt(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names or None


def load_old_names(dataset_root: Path) -> list[str]:
    names = _parse_data_yaml_names(dataset_root / "data.yaml")
    if names:
        return names
    names = _parse_classes_txt(dataset_root / "classes.txt")
    if names:
        return names
    raise SystemExit(f"未找到 data.yaml / classes.txt 中的类别名: {dataset_root}")


def build_id_remap(old_names: list[str]) -> dict[int, int]:
    """old_class_id -> new_class_id (HIER)."""
    remap: dict[int, int] = {}
    unknown: list[str] = []
    for old_id, name in enumerate(old_names):
        if name not in TARGET_NAME_TO_ID:
            unknown.append(f"{old_id}:{name}")
            continue
        remap[old_id] = TARGET_NAME_TO_ID[name]
    if unknown:
        raise SystemExit(f"存在非 RM 标准类名，无法映射: {unknown}")
    return remap


def remap_label_file(path: Path, id_remap: dict[int, int], dry_run: bool) -> tuple[int, int]:
    """返回 (行数, 改写行数)。"""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return 0, 0
    out_lines: list[str] = []
    changed = 0
    total = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        total += 1
        parts = line.split()
        old_id = int(float(parts[0]))
        if old_id not in id_remap:
            raise SystemExit(f"{path}: 未知 class_id={old_id} (行: {line[:80]})")
        new_id = id_remap[old_id]
        if new_id != old_id:
            changed += 1
        parts[0] = str(new_id)
        out_lines.append(" ".join(parts))
    if not dry_run and changed:
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return total, changed


def write_meta(dataset_root: Path, kpt_shape: list[int] | None, dry_run: bool) -> None:
    classes_txt = "\n".join(TARGET_NAMES) + "\n"
    # 尽量保留原 kpt_shape
    if kpt_shape is None:
        old_yaml = dataset_root / "data.yaml"
        if old_yaml.is_file():
            m = re.search(r"kpt_shape:\s*\[(\d+)\s*,\s*(\d+)\]", old_yaml.read_text(encoding="utf-8"))
            if m:
                kpt_shape = [int(m.group(1)), int(m.group(2))]
    if kpt_shape is None:
        kpt_shape = [4, 3]

    abs_root = dataset_root.resolve().as_posix()
    yaml_lines = [
        "# 数据路径",
        f"path: {abs_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "# Number of classes (RM 固定 14 类, 与 cpp 对齐)",
        f"nc: {HIER_NUM_CLASSES}",
        "",
        "# Keypoints (for pose training)",
        f"kpt_shape: {kpt_shape}",
        f"flip_idx: {list(range(kpt_shape[0]))}",
        "",
        "# Class names",
        "names:",
    ]
    for i, name in enumerate(TARGET_NAMES):
        yaml_lines.append(f"  {i}: {name}")
    yaml_text = "\n".join(yaml_lines) + "\n"

    if dry_run:
        print("[dry-run] 将写入 classes.txt / data.yaml (nc=14, BG=0)")
        return
    (dataset_root / "classes.txt").write_text(classes_txt, encoding="utf-8")
    (dataset_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def collect_label_files(dataset_root: Path) -> list[Path]:
    labels = dataset_root / "labels"
    if not labels.is_dir():
        raise SystemExit(f"未找到 labels/: {labels}")
    files = sorted(labels.rglob("*.txt"))
    # 排除可能误放的 classes.txt
    return [p for p in files if p.name != "classes.txt"]


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO class_id → RM 固定 14 类 (BG=0)")
    ap.add_argument("dataset", type=Path, help="数据集根目录 (含 data.yaml / labels/)")
    ap.add_argument("--dry-run", action="store_true", help="只打印映射，不写盘")
    args = ap.parse_args()
    root = args.dataset.resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    old_names = load_old_names(root)
    id_remap = build_id_remap(old_names)

    print(f"数据集: {root}")
    print(f"旧 names ({len(old_names)}): {old_names}")
    print("id 映射 (old → new / name):")
    for old_id, new_id in sorted(id_remap.items()):
        print(f"  {old_id} → {new_id}  ({old_names[old_id]} → {TARGET_NAMES[new_id]})")
    print(f"目标 14 类: {TARGET_NAMES}")

    files = collect_label_files(root)
    total_lines = 0
    total_changed = 0
    for f in files:
        n, c = remap_label_file(f, id_remap, args.dry_run)
        total_lines += n
        total_changed += c

    write_meta(root, None, args.dry_run)
    mode = "dry-run" if args.dry_run else "已写盘"
    print(f"标签文件: {len(files)} 个, 实例行: {total_lines}, 改写: {total_changed} ({mode})")
    print("完成。BG 现在应为 class_id=0。")


if __name__ == "__main__":
    main()
