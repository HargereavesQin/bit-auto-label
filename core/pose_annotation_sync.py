# -*- coding: utf-8 -*-
"""
位姿配置变更与目录内 JSON/YOLO 标注的全局同步。
"""
from __future__ import annotations

import copy
import json
import os
from typing import Optional


def snapshot_categories(pose_config) -> dict:
    """深拷贝当前位姿类别定义。"""
    return copy.deepcopy(pose_config.categories)


def is_pose_yolo_line(parts: list[str]) -> bool:
    return len(parts) > 5 and (len(parts) - 5) % 3 == 0


def compute_pose_config_changes(before: dict, after: dict) -> dict:
    """
    对比保存前后的位姿配置。

    Returns:
        removed, renamed, trimmed, kpt_renamed
    """
    before_names = set(before.keys())
    after_names = set(after.keys())

    removed = sorted(before_names - after_names)
    trimmed: dict[str, dict] = {}
    kpt_renamed: dict[str, list] = {}

    for name in before_names & after_names:
        old_k = before[name]["kpt_count"]
        new_k = after[name]["kpt_count"]
        old_names = before[name]["keypoints"]
        new_names = after[name]["keypoints"]
        if new_k < old_k:
            trimmed[name] = {
                "old_count": old_k,
                "new_count": new_k,
                "keypoints": list(new_names),
            }
        elif new_k == old_k and old_names != new_names:
            kpt_renamed[name] = list(new_names)

    return {
        "removed": removed,
        "renamed": {},
        "trimmed": trimmed,
        "kpt_renamed": kpt_renamed,
    }


def merge_pending_renames(changes: dict, pending_renames: dict[str, str]) -> dict:
    """将对话框中记录的类别重命名合并进变更集。"""
    changes = copy.deepcopy(changes)
    renamed = dict(changes.get("renamed", {}))
    removed = list(changes.get("removed", []))

    for old_name, new_name in pending_renames.items():
        if old_name in removed:
            removed.remove(old_name)
        renamed[old_name] = new_name
        if old_name in changes.get("trimmed", {}):
            changes["trimmed"][new_name] = changes["trimmed"].pop(old_name)
        if old_name in changes.get("kpt_renamed", {}):
            changes["kpt_renamed"][new_name] = changes["kpt_renamed"].pop(old_name)

    changes["removed"] = removed
    changes["renamed"] = renamed
    return changes


def _trim_json_shape(shape: dict, new_count: int, kpt_names: list[str]) -> bool:
    st = shape.get("shape_type", "")
    if st != "pose" and not shape.get("keypoints"):
        return False
    kpts = shape.get("keypoints", [])
    if len(kpts) <= new_count and shape.get("kpt_names") == kpt_names[:new_count]:
        return False
    shape["keypoints"] = kpts[:new_count]
    shape["kpt_names"] = list(kpt_names[:new_count])
    return True


def _trim_yolo_pose_parts(parts: list[str], new_count: int) -> list[str]:
    return parts[:5] + parts[5:5 + new_count * 3]


def _iter_dataset_pairs(dir_path: str):
    """遍历目录内图片及同名的 json/txt。"""
    if not dir_path or not os.path.isdir(dir_path):
        return
    for fn in os.listdir(dir_path):
        lower = fn.lower()
        if not lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img_path = os.path.join(dir_path, fn)
        base = os.path.splitext(img_path)[0]
        yield img_path, base + ".json", base + ".txt"


def apply_pose_config_changes_to_dir(
    dir_path: str,
    class_list: list[str],
    changes: dict,
) -> dict:
    """
    将位姿配置变更应用到目录内全部 JSON / YOLO txt。

    Returns:
        removed_shapes, trimmed_shapes, renamed_shapes, affected_files
    """
    removed = set(changes.get("removed", []))
    renamed = changes.get("renamed", {})
    trimmed = changes.get("trimmed", {})
    kpt_renamed = changes.get("kpt_renamed", {})

    summary = {
        "removed_shapes": 0,
        "trimmed_shapes": 0,
        "renamed_shapes": 0,
        "affected_files": set(),
    }
    if not dir_path:
        return summary

    class_index = {name: i for i, name in enumerate(class_list)}

    for _img_path, json_path, txt_path in _iter_dataset_pairs(dir_path):
        file_changed = False

        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = None

            if data is not None:
                new_shapes = []
                for shape in data.get("shapes", []):
                    label = shape.get("label", "")
                    st = shape.get("shape_type", "")
                    is_pose = st == "pose" or bool(shape.get("keypoints"))

                    if is_pose and label in removed:
                        summary["removed_shapes"] += 1
                        file_changed = True
                        continue

                    if is_pose and label in renamed:
                        shape = dict(shape)
                        shape["label"] = renamed[label]
                        summary["renamed_shapes"] += 1
                        file_changed = True
                        label = renamed[label]

                    if is_pose and label in trimmed:
                        info = trimmed[label]
                        shape = dict(shape)
                        if _trim_json_shape(shape, info["new_count"], info["keypoints"]):
                            summary["trimmed_shapes"] += 1
                            file_changed = True
                    elif is_pose and label in kpt_renamed:
                        shape = dict(shape)
                        names = kpt_renamed[label]
                        if shape.get("kpt_names") != names:
                            shape["kpt_names"] = list(names)
                            file_changed = True

                    new_shapes.append(shape)

                if file_changed:
                    data["shapes"] = new_shapes
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    summary["affected_files"].add(json_path)

        if os.path.isfile(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue

            new_lines = []
            txt_changed = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    new_lines.append(line)
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    new_lines.append(line)
                    continue

                if cid < 0 or cid >= len(class_list):
                    new_lines.append(line)
                    continue

                label = class_list[cid]
                is_pose = is_pose_yolo_line(parts)

                if is_pose and label in removed:
                    summary["removed_shapes"] += 1
                    txt_changed = True
                    continue

                if is_pose and label in renamed:
                    summary["renamed_shapes"] += 1
                    txt_changed = True
                    label = renamed[label]

                if is_pose and label in trimmed:
                    new_count = trimmed[label]["new_count"]
                    if len(parts) > 5 + new_count * 3:
                        parts = _trim_yolo_pose_parts(parts, new_count)
                        summary["trimmed_shapes"] += 1
                        txt_changed = True

                new_lines.append(" ".join(parts) + "\n")

            if txt_changed:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                summary["affected_files"].add(txt_path)

    return summary


def describe_pose_config_changes(changes: dict) -> str:
    """生成确认对话框用的变更摘要。"""
    lines = []
    if changes.get("removed"):
        lines.append(f"删除类别: {', '.join(changes['removed'])}（移除对应位姿标注）")
    if changes.get("renamed"):
        pairs = [f"{a} → {b}" for a, b in changes["renamed"].items()]
        lines.append(f"重命名类别: {', '.join(pairs)}")
    for name, info in changes.get("trimmed", {}).items():
        lines.append(
            f"裁剪 [{name}] 关键点: {info['old_count']} → {info['new_count']}"
        )
    if changes.get("kpt_renamed"):
        lines.append(
            "更新关键点名称: "
            + ", ".join(changes["kpt_renamed"].keys())
        )
    return "\n".join(lines) if lines else "无标注同步变更"
