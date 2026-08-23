# -*- coding: utf-8 -*-
"""
位姿标注图形 - 包含 bbox + 关键点
"""
from typing import Optional

from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem
from PySide6.QtGui import QPen, QBrush, QColor, QFont, QPainter
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from core.shapes import BaseShape, clamp_item_position


FIXED_KPT_INDEX = 0
ADJUST_KPT_INDICES = (1, 2, 3, 4)
KPT_RADIUS = 7


class KeypointItem(QGraphicsEllipseItem):
    """单个关键点图形"""

    def __init__(self, index, name, parent=None, radius=KPT_RADIUS):
        r = radius
        super().__init__(-r, -r, r * 2, r * 2, parent)
        self.kpt_index = index  # 关键点序号（从0开始）
        self.kpt_name = name
        self.visibility = 2  # 0=未标注, 1=遮挡, 2=可见

        self.setBrush(QBrush(QColor(0, 255, 0)))
        self.setPen(QPen(QColor(255, 255, 255), 1.5))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(1002)
        self.setCursor(Qt.CrossCursor)

        # 序号标签
        self.label_item = QGraphicsTextItem(str(index + 1), self)
        self.label_item.setDefaultTextColor(QColor(255, 255, 0))
        self.label_item.setFont(QFont("Arial", 9, QFont.Bold))
        self.label_item.setPos(r + 2, -r - 2)
        self.label_item.setZValue(1003)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor(255, 255, 0)))
        super().hoverEnterEvent(event)

    def _visibility_colors(self):
        """普通关键点 0/1/2；分层 KP4(L2) 用 0..3。"""
        parent = self.parentItem()
        if (getattr(parent, "is_hierarchical", False)
                and getattr(self, "kpt_index", -1) == 4):
            return {
                0: QColor(100, 100, 100),
                1: QColor(255, 165, 0),
                2: QColor(0, 255, 0),
                3: QColor(100, 149, 237),
            }
        return {0: QColor(100, 100, 100), 1: QColor(255, 165, 0), 2: QColor(0, 255, 0)}

    def hoverLeaveEvent(self, event):
        colors = self._visibility_colors()
        self.setBrush(QBrush(colors.get(self.visibility, QColor(0, 255, 0))))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            # 右键切换可见性（不传递给父级，避免被删除）
            # 分层 KP4 = L2 子类码：0→1→2→3→0；角点仍为 0→1→2→0
            parent = self.parentItem()
            if (getattr(parent, "is_hierarchical", False)
                    and getattr(self, "kpt_index", -1) == 4):
                self.visibility = (self.visibility + 1) % 4
            else:
                self.visibility = (self.visibility + 1) % 3
            colors = self._visibility_colors()
            self.setBrush(QBrush(colors.get(self.visibility, QColor(0, 255, 0))))
            event.accept()
            # 通知父 PoseShape + 触发画布 state_changed, 让 main.py 落盘
            if parent is not None and hasattr(parent, "keypoint_visibility_changed"):
                try:
                    parent.keypoint_visibility_changed(self)
                except Exception:
                    pass
            scene = self.scene()
            if scene is not None and hasattr(scene, "state_changed"):
                try:
                    scene.state_changed.emit()
                except Exception:
                    pass
            return
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            scene_rect = self.scene().sceneRect()
            new_pos = value
            x = max(scene_rect.left(), min(new_pos.x(), scene_rect.right()))
            y = max(scene_rect.top(), min(new_pos.y(), scene_rect.bottom()))
            return QPointF(x, y)
        if change == QGraphicsItem.ItemPositionHasChanged:
            parent = self.parentItem()
            if parent is not None and getattr(parent, "bbox_hidden", False):
                parent.update_bbox_from_keypoints()
                parent.set_bbox_visible(False)
            scene = self.scene()
            if scene is not None and hasattr(scene, "state_changed"):
                try:
                    scene.state_changed.emit()
                except Exception:
                    pass
        return super().itemChange(change, value)


class PoseShape(QGraphicsRectItem, BaseShape):
    """
    位姿标注图形
    包含一个 bbox 矩形框 + 一组有序关键点
    """

    def __init__(self, bbox_rect, label="", kpt_names=None, keypoints=None):
        """
        bbox_rect: QRectF bbox 范围
        label: 类别名
        kpt_names: 关键点名称列表
        keypoints: 已有的关键点坐标列表 [(x,y,visibility), ...]
        """
        super().__init__(bbox_rect)
        self.label = label
        self.kpt_names = kpt_names or []
        self._kpt_count = len(self.kpt_names)  # 内部存储，通过属性访问
        self.is_temp = False
        self.bbox_hidden = False

        # 样式
        self.normal_pen = QPen(QColor(255, 165, 0), 2)
        self.normal_brush = QBrush(QColor(255, 165, 0, 30))
        self.hover_brush = QBrush(QColor(255, 165, 0, 80))

        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # 标签
        self.setup_label(self)
        if label:
            self.update_label_text(label)
            self.update_label_position(self)

        # 关键点列表
        self.keypoint_items = []
        if keypoints:
            for i, (kx, ky, kv) in enumerate(keypoints):
                name = self.kpt_names[i] if i < len(self.kpt_names) else f"kpt_{i}"
                kpt = KeypointItem(i, name, self)
                kpt.setPos(kx, ky)
                kpt.visibility = kv
                colors = {0: QColor(100, 100, 100), 1: QColor(255, 165, 0), 2: QColor(0, 255, 0)}
                kpt.setBrush(QBrush(colors.get(kv, QColor(0, 255, 0))))
                self.keypoint_items.append(kpt)

    @property
    def kpt_count(self):
        return self._kpt_count

    @kpt_count.setter
    def kpt_count(self, value):
        self._kpt_count = value

    def get_keypoints_data(self):
        """获取关键点数据 [(x, y, visibility), ...]"""
        data = []
        for kpt in self.keypoint_items:
            scene_pos = self.mapToScene(kpt.pos())
            data.append([scene_pos.x(), scene_pos.y(), kpt.visibility])
        return data

    def get_kpt_names(self):
        """获取关键点名称列表"""
        return self.kpt_names

    def trim_keypoints_to(self, count: int, kpt_names: Optional[list] = None):
        """裁剪多余关键点并更新名称（配置变更后同步画布）。"""
        count = max(0, int(count))
        while len(self.keypoint_items) > count:
            item = self.keypoint_items.pop()
            if item.scene():
                item.scene().removeItem(item)
            item.setParentItem(None)
        if kpt_names is not None:
            self.kpt_names = list(kpt_names[:count])
        self._kpt_count = count
        for i, kpt in enumerate(self.keypoint_items):
            name = self.kpt_names[i] if i < len(self.kpt_names) else f"kpt_{i}"
            kpt.kpt_index = i
            kpt.kpt_name = name
            if kpt.label_item:
                kpt.label_item.setPlainText(str(i + 1))

    def set_kpt_names(self, names: list):
        """仅更新关键点名称（数量不变）。"""
        self.kpt_names = list(names)
        self._kpt_count = len(self.kpt_names)
        for i, kpt in enumerate(self.keypoint_items):
            name = self.kpt_names[i] if i < len(self.kpt_names) else f"kpt_{i}"
            kpt.kpt_name = name

    def add_keypoint(self, scene_pos):
        """添加一个关键点（在场景坐标中）"""
        idx = len(self.keypoint_items)
        if idx >= self.kpt_count:
            return False

        name = self.kpt_names[idx] if idx < len(self.kpt_names) else f"kpt_{idx}"
        local_pos = self.mapFromScene(scene_pos)
        kpt = KeypointItem(idx, name, self)
        kpt.setPos(local_pos)
        self.keypoint_items.append(kpt)
        return True

    def get_next_kpt_index(self):
        """获取下一个要标注的关键点序号"""
        return len(self.keypoint_items)

    def is_kpt_complete(self):
        """关键点是否全部标注完成"""
        return len(self.keypoint_items) >= self.kpt_count

    def set_bbox_visible(self, visible: bool):
        """隐藏/显示 bbox 矩形（仅标点模式下隐藏）。"""
        self.bbox_hidden = not visible
        if visible:
            self.setPen(self.normal_pen)
            self.setBrush(self.normal_brush)
        else:
            self.setPen(QPen(Qt.NoPen))
            self.setBrush(QBrush(Qt.NoBrush))
        if hasattr(self, "label_item") and self.label_item is not None:
            self.label_item.setVisible(visible)

    def update_bbox_from_keypoints(self, padding: float = 10.0):
        """根据已放置关键点自动更新 bbox（导出 YOLO 仍需要 bbox）。"""
        if not self.keypoint_items:
            return
        xs, ys = [], []
        for kpt in self.keypoint_items:
            if kpt.visibility == 0:
                continue
            xs.append(kpt.pos().x())
            ys.append(kpt.pos().y())
        if not xs:
            for kpt in self.keypoint_items:
                xs.append(kpt.pos().x())
                ys.append(kpt.pos().y())
        if not xs:
            return
        x1 = min(xs) - padding
        y1 = min(ys) - padding
        x2 = max(xs) + padding
        y2 = max(ys) + padding
        self.setRect(x1, y1, max(x2 - x1, 4.0), max(y2 - y1, 4.0))
        if hasattr(self, "update_label_position"):
            self.update_label_position(self)

    def hoverEnterEvent(self, event):
        self.setBrush(self.hover_brush)
        self.setCursor(Qt.PointingHandCursor)
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(self.normal_brush)
        self.setCursor(Qt.ArrowCursor)
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=False)
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False):
            return super().itemChange(change, clamp_item_position(self, value))
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.update_label_position(self)
        return super().itemChange(change, value)
