"""Hierarchical annotation config — robot class hierarchy (14-class cpp-aligned scheme).

Class layout (与 cpp 推理框架对齐, 颜色已编码进 class_id):
  0:BG(蓝哨兵)  1:B1  2:B2  3:B3  4:B4  5:BO(前哨+L2)  6:BB
  7:RG(红哨兵)  8:R1  9:R2  10:R3  11:R4  12:RO(前哨+L2)  13:RB

L2 只对 BO(5) / RO(12) 两个前哨兵种生效, 哨兵 BG/RG 与其它编号装甲均为 4 角点.

关键点 (5 个仅用于有 L2 的前哨; 其余兵种 4 角点):
  KP0-3 = 四角点 (TL/TR/BL/BR)
  KP4   = L2 子类 (虚拟关键点, visibility 编码: 0=none 1=up 2=mid 3=down,
          坐标复用 bbox 中心, 不单独占物理点)
  L1 颜色已并入 class_id, 不再单独占 KP。
"""
from __future__ import annotations

import json
import os
from typing import Any


# ---- KP index constants (cursorules 2.1 Virtual Keypoint Multiplexing) ----
KPT_INDEX_CORNER_TL = 0
KPT_INDEX_CORNER_TR = 1
KPT_INDEX_CORNER_BL = 2
KPT_INDEX_CORNER_BR = 3
KPT_INDEX_L2_SUBLEVEL = 4
KPT_TOTAL_HIERARCHY = 5

# ---- L2 visibility codes ----
VIS_L2_NONE = 0
VIS_L2_UP = 1
VIS_L2_MID = 2
VIS_L2_DOWN = 3

L2_LEVELS_DEFAULT = {"none": VIS_L2_NONE, "up": VIS_L2_UP, "mid": VIS_L2_MID, "down": VIS_L2_DOWN}

DEFAULT_KPT_NAMES = ["TL", "TR", "BL", "BR", "L2_sublevel"]
CORNER_KPT_NAMES = ["TL", "TR", "BL", "BR"]
CORNER_KPT_COUNT = 4

# 14 类名 (索引即 class_id, 与 cpp 框架对齐)
CLASS_NAMES_14 = [
    "BG", "B1", "B2", "B3", "B4", "BO", "BB",
    "RG", "R1", "R2", "R3", "R4", "RO", "RB",
]
NUM_CLASSES = 14
# 旧常量名保留兼容；BG = Blue Guard（蓝哨兵），不是背景
BG_CLASS_ID = 0


class HierarchicalConfig:
    """管理 14 类兵种 + L2 子类定义 (从 JSON 加载)."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or ""
        self.classes: list[dict[str, Any]] = []
        self.kpt_count: int = KPT_TOTAL_HIERARCHY
        self.kpt_names: list[str] = list(DEFAULT_KPT_NAMES)

    def load(self, config_path: str | None = None) -> bool:
        if config_path:
            self.config_path = config_path
        if not self.config_path or not os.path.exists(self.config_path):
            self._load_defaults()
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.classes = data.get("classes", [])
            self.kpt_count = data.get("kpt_count", KPT_TOTAL_HIERARCHY)
            self.kpt_names = data.get("kpt_names", list(DEFAULT_KPT_NAMES))
            if not self.classes:
                self._load_defaults()
            return True
        except Exception:
            self._load_defaults()
            return False

    def _load_defaults(self):
        self.classes = [
            {"id": i, "name": CLASS_NAMES_14[i],
             "l2_levels": (dict(L2_LEVELS_DEFAULT) if CLASS_NAMES_14[i] in ("BO", "RO") else {})}
            for i in range(NUM_CLASSES)
        ]
        self.kpt_count = KPT_TOTAL_HIERARCHY
        self.kpt_names = list(DEFAULT_KPT_NAMES)

    def save(self, config_path: str | None = None) -> bool:
        if config_path:
            self.config_path = config_path
        if not self.config_path:
            return False
        try:
            data = {
                "classes": self.classes,
                "kpt_count": self.kpt_count,
                "kpt_names": self.kpt_names,
                "l2_levels_default": dict(L2_LEVELS_DEFAULT),
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    # ---------------------------------------------------------
    def get_class_by_id(self, class_id: int) -> dict[str, Any] | None:
        for c in self.classes:
            if c.get("id") == class_id:
                return c
        return None

    def get_class_by_name(self, name: str) -> dict[str, Any] | None:
        for c in self.classes:
            if c.get("name") == name:
                return c
        return None

    def get_class_id(self, name: str) -> int | None:
        c = self.get_class_by_name(name)
        return c["id"] if c else None

    def get_class_name(self, class_id: int) -> str | None:
        c = self.get_class_by_id(class_id)
        return c["name"] if c else None

    def get_class_names(self) -> list[str]:
        return [c["name"] for c in self.classes]

    def get_annotatable_classes(self) -> list[dict[str, Any]]:
        """返回可标注的类（14 类全部可标，含 BG 蓝哨兵）。"""
        return list(self.classes)

    def has_sublevels(self, name_or_id: str | int) -> bool:
        c = self.get_class_by_name(name_or_id) if isinstance(name_or_id, str) else self.get_class_by_id(name_or_id)
        return bool(c and c.get("l2_levels"))

    def get_kpt_spec_for_class(self, name: str) -> tuple[int, list[str]]:
        """按兵种返回关键点配置：有 L2 的 BO/RO 为 5 点，其余（含 BG/RG）为 4 角点。"""
        if self.has_sublevels(name):
            return self.kpt_count, list(self.kpt_names)
        return CORNER_KPT_COUNT, list(CORNER_KPT_NAMES)

    def get_l2_levels(self, name_or_id: str | int) -> dict[str, int]:
        c = self.get_class_by_name(name_or_id) if isinstance(name_or_id, str) else self.get_class_by_id(name_or_id)
        return dict(c.get("l2_levels", {})) if c else {}

    @staticmethod
    def encode_visibility_to_label(code: int, levels: dict[str, int]) -> str:
        for label, value in levels.items():
            if value == code:
                return label
        return "none"

    @staticmethod
    def encode_label_to_visibility(label: str, levels: dict[str, int]) -> int:
        return levels.get(label, VIS_L2_NONE)
