"""
core/settings.py — 统一配置门面.

历史包袱:
  - 旧 LuoHuaLabel namespace (xml 标注)
  - 新 ArmorLabel namespace (RM/SAM/YOLO 参数)
  - core/tool_config.py (yaml, 给 standalone tool 用)

本 facade 把这三者收敛到统一 API, 旧代码也能用, 新代码必须用这个.

典型用法:
  from core.settings import Settings
  s = Settings.app()
  conf = s.get_float("Parameters/RM_VERTICAL_ANGLE_THRESHOLD", 50.0)

  t = Settings.tool("yolo_train")
  cfg = t.get_dict("yolo_train")
"""
from __future__ import annotations
import os
from typing import Any
from PySide6.QtCore import QSettings

from .constants import NS_APP, NS_LEGACY


class Settings:
    """QSettings 的薄封装 + 工厂方法.

    支持 get/set/sync/beginGroup/endGroup, 加上类型便捷方法.
    """

    def __init__(self, namespace: str = NS_APP, org: str = "settings"):
        self._q = QSettings(namespace, org)

    # ---- 通用 API ----
    def get(self, key: str, default: Any = None) -> Any:
        v = self._q.value(key, default)
        return v if v is not None else default

    def get_str(self, key: str, default: str = "") -> str:
        v = self.get(key, default)
        return str(v) if v is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        v = self.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    def get_list(self, key: str, default: list | None = None) -> list:
        v = self.get(key, default or [])
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return default or []

    def set(self, key: str, value: Any) -> None:
        self._q.setValue(key, value)

    def remove(self, key: str) -> None:
        self._q.remove(key)

    def sync(self) -> None:
        self._q.sync()

    def contains(self, key: str) -> bool:
        return self._q.contains(key)

    # ---- 上下文管理: 临时进入 group ----
    def begin_group(self, name: str) -> "_GroupCtx":
        return _GroupCtx(self, name)

    def _begin(self, name: str) -> None:
        self._q.beginGroup(name)

    def _end(self) -> None:
        self._q.endGroup()

    # ---- 工厂 ----
    @classmethod
    def app(cls) -> "Settings":
        """主 app 的 namespace (ArmorLabel)."""
        return cls(NS_APP)

    @classmethod
    def legacy(cls) -> "Settings":
        """老 namespace (LuoHuaLabel), 仅读旧数据时使用."""
        return cls(NS_LEGACY)


class _GroupCtx:
    """with settings.begin_group("Parameters") as g:"""

    def __init__(self, settings: Settings, name: str):
        self._s = settings
        self._name = name

    def __enter__(self) -> Settings:
        self._s._begin(self._name)
        return self._s

    def __exit__(self, *exc) -> None:
        self._s._end()


# ---------------------------------------------------------------------------
# 便捷读取: 直接拿 RM/SAM/YOLO/MLP 等命名参数的常用字段
# ---------------------------------------------------------------------------

def get_rm_params() -> dict:
    """RM 装甲板半自动参数."""
    with Settings.app().begin_group("Parameters") as s:
        return {
            "vertical_angle": s.get_float("RM_VERTICAL_ANGLE_THRESHOLD", 50.0),
            "parallel":       s.get_float("RM_PARALLEL_THRESHOLD", 45.0),
            "bar_length":     s.get_float("RM_BAR_LENGTH_MIN_RATIO", 0.5),
            "search_radius":  s.get_float("RM_SEARCH_RADIUS_RATIO", 1.5),
            "roi_padding":    s.get_int("RM_ROI_PADDING", 10),
            "overlay_ms":     s.get_int("RM_OVERLAY_TIMEOUT", 1000),
        }


def get_yolo_params() -> dict:
    with Settings.app().begin_group("Parameters") as s:
        return {
            "conf":   s.get_float("YOLO_CONF", 0.25),
            "iou":    s.get_float("YOLO_IOU", 0.7),
            "imgsz":  s.get_int("YOLO_IMGSZ", 640),
            "device": s.get_str("YOLO_DEVICE", "cuda"),
        }


def get_mlp_params() -> dict:
    with Settings.app().begin_group("Parameters") as s:
        return {
            "light_len":    s.get_int("MLP_ROI_LIGHT_LEN", 12),
            "persp_h":      s.get_int("MLP_ROI_PERSP_H", 28),
            "persp_w":      s.get_int("MLP_ROI_PERSP_W", 32),
            "crop_w":       s.get_int("MLP_ROI_CROP_W", 16),
            "expand_ratio": s.get_float("MLP_ROI_EXPAND_RATIO", 1.3),
            "center_ratio": s.get_float("MLP_ROI_CENTER_RATIO", 0.5),
        }


def get_editor_params() -> dict:
    with Settings.app().begin_group("Parameters") as s:
        return {
            "max_history":   s.get_int("MAX_HISTORY_STEPS", 20),
            "snap_threshold": s.get_int("SNAP_THRESHOLD", 8),
        }
