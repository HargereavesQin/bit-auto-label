"""
core/shape_registry.py — Shape 类型注册表.

历史:
  - main.py 4 处 `if shape_type == "rectangle" / "polygon" / "point" / "obb" / "pose" / ...`
  - core/exporter.py 3 处同类分发 (extract_shapes / save_yolo / save_xml)
  - 加一种新 shape 要改 7 处

本模块让每种 shape 自己负责:
  1. JSON dict ↔ QGraphicsItem 双向转换
  2. YOLO 行生成 (det / seg / pose 各 format)
  3. XML element 生成

新增 shape 只需要: 写一个 Shape 子类 + 一次 register() 调用, 不再碰 main.py / exporter.py.

注意: 由于 shape 实际类 (RectShape 等) 依赖 Qt, 这个文件只在已 import
PySide6.QtWidgets 的进程中使用. 函数本身不实例化 shape, 避免循环依赖.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Type, Callable, Optional, Any

# 延迟导入: 避免这个模块被 import 时强制拖入 Qt
def _qgraphicsitem():
    from PySide6.QtWidgets import QGraphicsItem
    return QGraphicsItem


@dataclass
class ShapeSpec:
    """一种 shape 的完整描述. 注册到全局表.

    必填:
      type_name:        JSON 中存的字符串 ("rectangle")
      label:            UI 显示名 ("矩形")
      icon:             Icon 库中的 key (Icons.RECT)
      qgraphics_class:  对应的 QGraphicsItem 子类
      to_dict:          (item) -> dict
      from_dict:        (d, parent_scene) -> QGraphicsItem

    可选 (用于导出 YOLO / VOC XML):
      to_yolo_line:     (d, img_w, img_h, class_id) -> str | None
      to_xml_node:      (d) -> dict (bndbox + 额外字段)
    """

    type_name: str
    label: str
    icon: str
    qgraphics_class: Type
    to_dict: Callable
    from_dict: Callable
    to_yolo_line: Optional[Callable] = None
    to_xml_node: Optional[Callable] = None

    def matches_item(self, item) -> bool:
        return isinstance(item, self.qgraphics_class)


# ---------------------------------------------------------------------------
# 全局表
# ---------------------------------------------------------------------------
_registry: dict[str, ShapeSpec] = {}


def register(spec: ShapeSpec) -> ShapeSpec:
    """注册一种 shape. 同名会覆盖."""
    _registry[spec.type_name] = spec
    return spec


def get(type_name: str) -> Optional[ShapeSpec]:
    return _registry.get(type_name)


def all_specs() -> list[ShapeSpec]:
    return list(_registry.values())


def spec_for_item(item) -> Optional[ShapeSpec]:
    """根据 QGraphicsItem 实例反查 spec."""
    for spec in _registry.values():
        if spec.matches_item(item):
            return spec
    return None


def unregister_all() -> None:
    """测试用. 清除全部注册."""
    _registry.clear()


# ---------------------------------------------------------------------------
# 便捷: 通用 to_dict / from_dict 工厂
# ---------------------------------------------------------------------------
def make_dict_coupler(
    *,
    to: Callable[[Any], dict],
    from_: Callable[[dict], Any],
):
    """返回 (to_dict_fn, from_dict_fn) 闭包. 给 shape 子类用.

    用法:
        class RectShape(...):
            ...

        # 在外部注册时:
        spec = ShapeSpec(
            ...,
            to_dict=lambda item: {"type": "rectangle", ...},
            from_dict=lambda d, parent: RectShape(...),
        )
    """
    return to, from_


# ---------------------------------------------------------------------------
# 默认注册 — 5 种内置 shape
# 子类实例由调用方在第一次访问时构造 (Qt 已初始化)
# ---------------------------------------------------------------------------
def ensure_default_specs():
    """注册 5 种内置 shape. 重复调用幂等."""
    if _registry:
        return
    from . import shapes as _shapes_mod
    from . import pose_shapes as _pose_mod
    from .constants import (
        SHAPE_RECTANGLE, SHAPE_POLYGON, SHAPE_POINT, SHAPE_OBB, SHAPE_POSE,
        SHAPE_HIERARCHY,
    )
    from PySide6.QtCore import QRectF, QPointF
    from PySide6.QtGui import QPolygonF, QColor, QBrush

    # ---- Rectangle ----
    def rect_to_dict(item):
        rect = item.rect()
        p1 = item.mapToScene(rect.topLeft())
        p2 = item.mapToScene(rect.bottomRight())
        return {
            "label": item.label,
            "type": SHAPE_RECTANGLE,
            "points": [[p1.x(), p1.y()], [p2.x(), p2.y()]],
        }

    def rect_from_dict(d, parent=None):
        p0, p1 = d["points"][0], d["points"][1]
        r = QRectF(QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]))
        s = _shapes_mod.RectShape(r, d["label"])
        if parent is not None:
            s.setParentItem(parent)
        return s

    def rect_to_yolo(d, w, h, cid):
        x1, y1 = d["points"][0]
        x2, y2 = d["points"][1]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    register(ShapeSpec(
        type_name=SHAPE_RECTANGLE,
        label="矩形",
        icon="rect",
        qgraphics_class=_shapes_mod.RectShape,
        to_dict=rect_to_dict,
        from_dict=rect_from_dict,
        to_yolo_line=rect_to_yolo,
    ))

    # ---- Polygon ----
    def poly_to_dict(item):
        poly = item.polygon()
        mapped = item.mapToScene(poly)
        return {
            "label": item.label,
            "type": SHAPE_POLYGON,
            "points": [[pt.x(), pt.y()] for pt in mapped],
        }

    def poly_from_dict(d, parent=None):
        pts = [QPointF(pt[0], pt[1]) for pt in d["points"]]
        poly = QPolygonF()
        for pt in pts:
            poly.append(pt)
        s = _shapes_mod.PolyShape(poly, d["label"])
        if parent is not None:
            s.setParentItem(parent)
        return s

    def poly_to_yolo(d, w, h, cid):
        flat = []
        for pt in d["points"]:
            flat.append(f"{pt[0] / w:.6f}")
            flat.append(f"{pt[1] / h:.6f}")
        return f"{cid} " + " ".join(flat)

    register(ShapeSpec(
        type_name=SHAPE_POLYGON,
        label="多边形",
        icon="polygon",
        qgraphics_class=_shapes_mod.PolyShape,
        to_dict=poly_to_dict,
        from_dict=poly_from_dict,
        to_yolo_line=poly_to_yolo,
    ))

    # ---- Point ----
    def point_to_dict(item):
        center = item.mapToScene(item.rect().center())
        return {
            "label": item.label,
            "type": SHAPE_POINT,
            "points": [[center.x(), center.y()]],
        }

    def point_from_dict(d, parent=None):
        pt = d["points"][0]
        s = _shapes_mod.PointShape(QPointF(pt[0], pt[1]), d["label"])
        if parent is not None:
            s.setParentItem(parent)
        return s

    def point_to_yolo(d, w, h, cid):
        cx = d["points"][0][0] / w
        cy = d["points"][0][1] / h
        pw = ph = 0.02
        cx = max(pw / 2, min(1.0 - pw / 2, cx))
        cy = max(ph / 2, min(1.0 - pw / 2, cy))
        return f"{cid} {cx:.6f} {cy:.6f} {pw:.6f} {ph:.6f}"

    register(ShapeSpec(
        type_name=SHAPE_POINT,
        label="点",
        icon="point",
        qgraphics_class=_shapes_mod.PointShape,
        to_dict=point_to_dict,
        from_dict=point_from_dict,
        to_yolo_line=point_to_yolo,
    ))

    # ---- OBB ----
    def obb_to_dict(item):
        poly = item.polygon()
        pts = [[poly[i].x(), poly[i].y()] for i in range(4)]
        return {
            "label": item.label,
            "type": SHAPE_OBB,
            "points": pts,
            "rect": [item.pos().x(), item.pos().y(), item.box_w, item.box_h],
            "angle": item.rotation(),
        }

    def obb_from_dict(d, parent=None):
        rect = d.get("rect", [0, 0, 0, 0])
        angle = d.get("angle", 0)
        s = _shapes_mod.RotatedRectShape(rect[0], rect[1], rect[2], rect[3], angle, label=d["label"])
        if parent is not None:
            s.setParentItem(parent)
        return s

    def obb_to_yolo(d, w, h, cid):
        flat = []
        for pt in d["points"][:4]:
            flat.append(f"{pt[0] / w:.6f}")
            flat.append(f"{pt[1] / h:.6f}")
        return f"{cid} " + " ".join(flat)

    register(ShapeSpec(
        type_name=SHAPE_OBB,
        label="旋转框",
        icon="obb",
        qgraphics_class=_shapes_mod.RotatedRectShape,
        to_dict=obb_to_dict,
        from_dict=obb_from_dict,
        to_yolo_line=obb_to_yolo,
    ))

    # ---- Pose ----
    def pose_to_dict(item):
        if item.keypoint_items:
            item.update_bbox_from_keypoints()
        rect = item.rect()
        p1 = item.mapToScene(rect.topLeft())
        p2 = item.mapToScene(rect.bottomRight())
        type_name = (SHAPE_HIERARCHY if getattr(item, 'is_hierarchical', False)
                     else SHAPE_POSE)
        return {
            "label": item.label,
            "type": type_name,
            "points": [[p1.x(), p1.y()], [p2.x(), p2.y()]],
            "keypoints": item.get_keypoints_data(),
            "kpt_names": item.get_kpt_names(),
        }

    def pose_from_dict(d, parent=None):
        p0, p1 = d["points"][0], d["points"][1]
        r = QRectF(QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]))
        s = _pose_mod.PoseShape(r, d["label"])
        if d.get("type") == SHAPE_HIERARCHY:
            s.is_hierarchical = True
        if "keypoints" in d and "kpt_names" in d:
            kpt_data = d["keypoints"]
            kpt_names = d["kpt_names"]
            filtered_kpts = []
            filtered_names = []
            for i, k in enumerate(kpt_data):
                vis = k[2] if len(k) > 2 else 0
                kx, ky = float(k[0]), float(k[1])
                if vis == 0 and abs(kx) < 1e-6 and abs(ky) < 1e-6:
                    continue
                filtered_kpts.append(k)
                filtered_names.append(
                    kpt_names[i] if i < len(kpt_names) else f"kpt_{i}"
                )
            s.kpt_names = filtered_names
            s._kpt_count = len(filtered_names)
            for k in filtered_kpts:
                s.add_keypoint(QPointF(float(k[0]), float(k[1])))
                if s.keypoint_items:
                    vis = k[2] if len(k) > 2 else 0
                    item = s.keypoint_items[-1]
                    item.visibility = vis
                    colors = {
                        0: QColor(100, 100, 100),
                        1: QColor(255, 165, 0),
                        2: QColor(0, 255, 0),
                    }
                    item.setBrush(QBrush(colors.get(vis, QColor(0, 255, 0))))
        if parent is not None:
            s.setParentItem(parent)
        return s

    def pose_to_yolo(d, w, h, cid):
        x1, y1 = d["points"][0]
        x2, y2 = d["points"][1]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for kpt in d.get("keypoints", []):
            kx = kpt[0] / w
            ky = kpt[1] / h
            kv = kpt[2] if len(kpt) > 2 else 2
            parts.append(f"{kx:.6f} {ky:.6f} {int(kv)}")
        return " ".join(parts)

    register(ShapeSpec(
        type_name=SHAPE_POSE,
        label="位姿",
        icon="pose",
        qgraphics_class=_pose_mod.PoseShape,
        to_dict=pose_to_dict,
        from_dict=pose_from_dict,
        to_yolo_line=pose_to_yolo,
    ))

    # ---- Hierarchy Pose ----
    # 复用 PoseShape 类 + pose 的 to_dict/from_dict/to_yolo_line。
    # 区别仅在于 JSON type 标记为 "hierarchy_pose"，便于 reload 保留 7-kpt 配置。
    register(ShapeSpec(
        type_name=SHAPE_HIERARCHY,
        label="层级位姿",
        icon="pose",
        qgraphics_class=_pose_mod.PoseShape,
        to_dict=pose_to_dict,
        from_dict=pose_from_dict,
        to_yolo_line=pose_to_yolo,
    ))


# 自动注册一次, 但容错: 如果 Qt 不可用 (例如纯 python 测试), 不抛
try:
    ensure_default_specs()
except Exception:
    pass
