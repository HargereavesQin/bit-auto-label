# -*- coding: utf-8 -*-
from PySide6.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem
from PySide6.QtGui import QPixmap, QPolygonF, QPen, QColor, QBrush
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from core.shapes import RectShape, PolyShape, PointShape, RotatedRectShape, HandleItem
from core.pose_shapes import PoseShape, KeypointItem


class CanvasMode:
    EDIT = 0
    RECT = 1
    POLY = 2
    POINT = 3
    RBOX = 4
    POSE = 5
    MLP = 6

    @staticmethod
    def get_mode_name(mode):
        names = {1: "矩形", 2: "多边形", 3: "点", 4: "旋转框", 5: "位姿", 6: "MLP分类"}
        return names.get(mode, "未知")


class Canvas(QGraphicsScene):
    mouse_moved = Signal(int, int)
    shape_drawn = Signal(object)
    shape_double_clicked = Signal(object)
    state_changed = Signal()
    shape_deleted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Qt owns the C++ items, but signal/QVariant handoffs can leave their
        # Python subclasses without a strong owner. Keep accepted annotations
        # alive until explicitly removed, including between mouse events.
        self._annotation_items = {}
        self.mode = CanvasMode.RECT
        self.img_item = None
        self.sam_client = None
        self.sam_enabled = False

        self.drawing = False
        self.start_pt = None
        self.temp_item = None
        self.poly_pts = []

        # 位姿标注状态
        self.pose_active = True  # 是否已激活（允许绘制 bbox）
        self.pose_kpt_only = False  # 仅标点：跳过手动 bbox，点击直接放置关键点
        self.rm_auto_mode = False  # RM装甲板半自动模式：禁用SAM辅助bbox，走手动bbox+自动关键点
        self.hierarchical_active = False  # 层级标注：KP4 L2 元数据自动放置
        self.hier_meta_l2_code = 0

        # 右键框选删除相关
        self.right_dragging = False
        self.right_drag_rect = None

        # 智能悬停提示图层
        self.sam_hover_item = None

        # RM装甲板全自动可视化覆盖层
        self.rm_overlay_items = []

        self.h_line = QGraphicsLineItem()
        self.v_line = QGraphicsLineItem()
        crosshair_pen = QPen(QColor(255, 255, 255, 200), 1, Qt.DashLine)
        self.h_line.setPen(crosshair_pen)
        self.v_line.setPen(crosshair_pen)
        self.h_line.setZValue(9999)
        self.v_line.setZValue(9999)
        self.h_line.hide()
        self.v_line.hide()
        self.addItem(self.h_line)
        self.addItem(self.v_line)

    def addItem(self, item):
        super().addItem(item)
        if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
            self._annotation_items[id(item)] = item

    def removeItem(self, item):
        super().removeItem(item)
        self._annotation_items.pop(id(item), None)

    def clear(self):
        super().clear()
        self._annotation_items.clear()

    def load_image(self, path):
        self.clear_shapes()
        self.clear_rm_overlays()
        pixmap = QPixmap(path)
        if self.img_item:
            self.removeItem(self.img_item)
        self.img_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.img_item)
        self.setSceneRect(pixmap.rect())
        self.h_line.show()
        self.v_line.show()

    def clear_shapes(self):
        for item in self.items():
            if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                self.removeItem(item)

    def set_mode(self, mode):
        self.mode = mode
        self.cancel_drawing()
        for item in self.selectedItems():
            item.setSelected(False)

    def set_sam_enabled(self, enabled):
        self.sam_enabled = enabled

    def set_hierarchical_active(self, active: bool, l2_code: int = 0):
        self.hierarchical_active = active
        self.hier_meta_l2_code = l2_code
        if not active and self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None

    def clear_rm_overlays(self):
        for item in self.rm_overlay_items:
            self.removeItem(item)
        self.rm_overlay_items.clear()

    def is_inside_image(self, pt):
        if not self.img_item: return False
        return self.sceneRect().contains(pt)

    def clamp_point(self, pt):
        rect = self.sceneRect()
        x = max(rect.left(), min(pt.x(), rect.right()))
        y = max(rect.top(), min(pt.y(), rect.bottom()))
        return QPointF(x, y)

    def update_crosshair(self, pt):
        if self.img_item:
            rect = self.sceneRect()
            x = max(rect.left(), min(pt.x(), rect.right()))
            y = max(rect.top(), min(pt.y(), rect.bottom()))
            self.h_line.setLine(rect.left(), y, rect.right(), y)
            self.v_line.setLine(x, rect.top(), x, rect.bottom())
            self.mouse_moved.emit(int(x), int(y))

    def mouseMoveEvent(self, event):
        pt = event.scenePos()
        self.update_crosshair(pt)
        super().mouseMoveEvent(event)
        clamped_pt = self.clamp_point(pt)

        # ---------------- 右键框选删除拖拽 ----------------
        if self.right_dragging and self.start_pt:
            rect = QRectF(min(self.start_pt.x(), clamped_pt.x()), min(self.start_pt.y(), clamped_pt.y()),
                          abs(clamped_pt.x() - self.start_pt.x()), abs(clamped_pt.y() - self.start_pt.y()))
            if self.right_drag_rect:
                self.removeItem(self.right_drag_rect)
            self.right_drag_rect = QGraphicsRectItem(rect)
            self.right_drag_rect.setPen(QPen(QColor(181, 117, 106), 2, Qt.DashLine))
            self.right_drag_rect.setBrush(QBrush(QColor(181, 117, 106, 40)))
            self.right_drag_rect.setZValue(9998)
            self.addItem(self.right_drag_rect)
            return

        # ---------------- SAM 智能辅助悬停 ----------------
        sam_modes = [CanvasMode.RECT, CanvasMode.POLY, CanvasMode.RBOX]
        if self.mode == CanvasMode.POSE:
            sam_modes.append(CanvasMode.POSE)
        if (self.sam_enabled and self.is_inside_image(pt) and self.mode in sam_modes
                and not (self.mode == CanvasMode.POSE and self._pose_keypoint_target() is not None)):
            if self.sam_client:
                self.sam_client.request_inference(clamped_pt.x(), clamped_pt.y(), is_click=False)
            return
        elif self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None

        # ---------------- 常规绘图 ----------------
        if self.drawing and self.start_pt:
            rect = QRectF(min(self.start_pt.x(), clamped_pt.x()), min(self.start_pt.y(), clamped_pt.y()),
                          abs(clamped_pt.x() - self.start_pt.x()), abs(clamped_pt.y() - self.start_pt.y()))
            if self.temp_item: self.removeItem(self.temp_item)

            if self.mode in (CanvasMode.RECT, CanvasMode.POSE):
                self.temp_item = QGraphicsRectItem(rect)
                self.temp_item.is_temp = True
                if self.mode == CanvasMode.POSE:
                    self.temp_item.setPen(QPen(QColor(220, 186, 128), 2, Qt.DashLine))
                else:
                    self.temp_item.setPen(QPen(QColor(201, 163, 107), 2, Qt.DashLine))

            # 手动拉框时，调用全新的 RotatedRectShape 参数格式
            elif self.mode == CanvasMode.RBOX:
                cx, cy = rect.center().x(), rect.center().y()
                w, h = max(1, rect.width()), max(1, rect.height())
                self.temp_item = RotatedRectShape(cx, cy, w, h, 0, is_temp=True)

            self.addItem(self.temp_item)

        elif self.mode == CanvasMode.POLY and not self.sam_enabled and len(self.poly_pts) > 0:
            self.update_temp_poly(mouse_pos=clamped_pt)

    def handle_sam_result(self, poly_pts, rect_xywh, rect_obb, score, is_click):
        """处理来自 SAM 后台的推理结果，正确区分矩形、多边形和旋转框"""
        sam_modes = [CanvasMode.RECT, CanvasMode.POLY, CanvasMode.RBOX]
        if self.mode == CanvasMode.POSE:
            sam_modes.append(CanvasMode.POSE)
        if not self.sam_enabled or self.mode not in sam_modes:
            return

        if self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None

        # Ignore in-flight hover/click results while filling an existing pose.
        if self.mode == CanvasMode.POSE and self._pose_keypoint_target() is not None:
            return

        if not poly_pts or not rect_xywh:
            return

        # ---- 模式判断：矩形智能框 / 多边形点选 / 旋转框 ----
        if self.mode == CanvasMode.RECT:
            x, y, w, h = rect_xywh
            rect = QRectF(x, y, w, h)

            if is_click:
                shape = RectShape(rect)
                self.shape_drawn.emit(shape)
            else:
                self.sam_hover_item = QGraphicsRectItem(rect)
                self.sam_hover_item.setPen(QPen(QColor(143, 166, 143), 2, Qt.DashLine))
                self.sam_hover_item.setBrush(QBrush(QColor(143, 166, 143, 50)))
                self.addItem(self.sam_hover_item)

        elif self.mode == CanvasMode.POLY:
            qpts = [QPointF(p[0], p[1]) for p in poly_pts]
            if is_click:
                shape = PolyShape(QPolygonF(qpts))
                self.shape_drawn.emit(shape)
            else:
                self.sam_hover_item = PolyShape(QPolygonF(qpts), is_temp=True)
                self.sam_hover_item.setPen(QPen(QColor(143, 166, 143), 2, Qt.DashLine))
                self.sam_hover_item.setBrush(QBrush(QColor(143, 166, 143, 50)))
                self.addItem(self.sam_hover_item)

        # SAM 的 OBB 旋转框处理分支
        elif self.mode == CanvasMode.RBOX:
            if not rect_obb or len(rect_obb) < 5: return
            cx, cy, w, h, angle = rect_obb

            if is_click:
                shape = RotatedRectShape(cx, cy, w, h, angle)
                self.shape_drawn.emit(shape)
            else:
                self.sam_hover_item = RotatedRectShape(cx, cy, w, h, angle, is_temp=True)
                self.addItem(self.sam_hover_item)

        # SAM 位姿模式：用 SAM 检测矩形框作为 PoseShape 的 bbox
        elif self.mode == CanvasMode.POSE:
            x, y, w, h = rect_xywh
            rect = QRectF(x, y, w, h)

            if is_click:
                shape = PoseShape(rect, label="")
                self.shape_drawn.emit(shape)
                if shape.scene() is self:
                    self.clearSelection()
                    shape.setSelected(True)
            else:
                self.sam_hover_item = QGraphicsRectItem(rect)
                self.sam_hover_item.setPen(QPen(QColor(220, 186, 128), 2, Qt.DashLine))
                self.sam_hover_item.setBrush(QBrush(QColor(220, 186, 128, 50)))
                self.addItem(self.sam_hover_item)

    def mousePressEvent(self, event):
        pt = event.scenePos()
        clamped_pt = self.clamp_point(pt)

        # ---------------- 右键删除逻辑（优先于所有其他操作） ----------------
        if event.button() == Qt.RightButton:
            # A keypoint/its number is part of the pose, so merely skipping it
            # would hit the parent box next and delete the whole annotation.
            for item in self.items(clamped_pt):
                keypoint = item if isinstance(item, KeypointItem) else item.parentItem()
                if isinstance(keypoint, KeypointItem):
                    keypoint.mousePressEvent(event)
                    return
            # 检查是否点击到了标注形状
            items = self.items(clamped_pt)
            clicked_shape = None
            for item in items:
                # KeypointItem 右键由它自己处理（切换可见性），跳过不删除
                if isinstance(item, KeypointItem):
                    continue
                if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                    if not getattr(item, 'is_temp', False):
                        clicked_shape = item
                        break
                elif isinstance(item, HandleItem):
                    parent = item.parentItem()
                    if isinstance(parent, (RectShape, PolyShape, RotatedRectShape)) and not getattr(parent, 'is_temp', False):
                        clicked_shape = parent
                        break

            if clicked_shape:
                # 右键点击到标注 -> 直接删除
                self.removeItem(clicked_shape)
                self.shape_deleted.emit()
                return
            else:
                # 右键点击空白区域 -> 开始框选删除
                self.right_dragging = True
                self.start_pt = clamped_pt
                return

        # Once a pose exists, clicks fill its point slots before invoking SAM.
        # Corners just outside the predicted bbox still belong to this pose.
        if (self.mode == CanvasMode.POSE and event.button() == Qt.LeftButton
                and self.is_inside_image(pt)):
            for item in self.items(clamped_pt):
                if isinstance(item, KeypointItem) or isinstance(item.parentItem(), KeypointItem):
                    super().mousePressEvent(event)
                    return
            clicked_pose = self._find_clicked_pose(clamped_pt)
            target = (clicked_pose if clicked_pose is not None and not clicked_pose.is_kpt_complete()
                      else self._pose_keypoint_target())
            if target is not None:
                self.clearSelection()
                target.setSelected(True)
                if self.sam_hover_item:
                    self.removeItem(self.sam_hover_item)
                    self.sam_hover_item = None
                self._add_pose_keypoint(target, clamped_pt)
                event.accept()
                return

# ---------------- SAM 确认生成 ----------------
        sam_click_modes = [CanvasMode.RECT, CanvasMode.POLY, CanvasMode.RBOX]
        if self.mode == CanvasMode.POSE:
            sam_click_modes.append(CanvasMode.POSE)
        if self.sam_enabled and event.button() == Qt.LeftButton and self.mode in sam_click_modes:
            if self.mode == CanvasMode.POSE:
                clicked_pose = self._find_clicked_pose(clamped_pt)
                if clicked_pose:
                    pass
                elif not self.pose_kpt_only:
                    if self.is_inside_image(pt) and self.sam_client:
                        self.sam_client.request_inference(clamped_pt.x(), clamped_pt.y(), is_click=True)
                    return
            else:
                if self.is_inside_image(pt) and self.sam_client:
                    self.sam_client.request_inference(clamped_pt.x(), clamped_pt.y(), is_click=True)
                return

        items = self.items(clamped_pt)
        clicked_item = None
        for item in items:
            if isinstance(item, HandleItem) and item.isVisible():
                if not getattr(item.parentItem(), 'is_temp', False):
                    clicked_item = item
                    break
        if not clicked_item:
            for item in items:
                if isinstance(item, (PolyShape, RectShape, PointShape, RotatedRectShape, PoseShape)):
                    if not getattr(item, 'is_temp', False):
                        # POSE模式下，PoseShape的点击由下方pose代码专门处理（关键点添加）
                        if self.mode == CanvasMode.POSE and isinstance(item, PoseShape):
                            continue
                        clicked_item = item
                        break

        if clicked_item and not self.sam_enabled:
            super().mousePressEvent(event)
            if event.button() == Qt.LeftButton:
                if not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                    for item in self.selectedItems():
                        if item != clicked_item and item != clicked_item.parentItem():
                            item.setSelected(False)
                    if isinstance(clicked_item, HandleItem):
                        clicked_item.parentItem().setSelected(True)
                    else:
                        clicked_item.setSelected(True)
            return

        # ---------------- 位姿标注模式 ----------------
        if self.mode == CanvasMode.POSE:
            if event.button() == Qt.LeftButton:
                clicked_pose = self._find_clicked_pose(clamped_pt)
                print(f"[POSE] press: sam={self.sam_enabled} kpt_only={self.pose_kpt_only} "
                      f"hier={self.hierarchical_active} clicked_pose={'Y' if clicked_pose else 'N'}")

                if clicked_pose:
                    if not clicked_pose.is_kpt_complete():
                        n_before = len(clicked_pose.keypoint_items)
                        ok = self._add_pose_keypoint(clicked_pose, clamped_pt)
                        print(f"[POSE]   → add kpt to existing: {n_before}->{len(clicked_pose.keypoint_items)} ok={ok}")
                    else:
                        print(f"[POSE]   → clicked pose already complete (ignored)")
                    super().mousePressEvent(event)
                    return
                elif self.pose_kpt_only:
                    incomplete = self._find_incomplete_pose()
                    if incomplete is not None:
                        n_before = len(incomplete.keypoint_items)
                        self._add_pose_keypoint(incomplete, clamped_pt)
                        print(f"[POSE]   → kpt_only: add to incomplete {n_before}->{len(incomplete.keypoint_items)}")
                        return
                    self._start_kpt_only_pose(clamped_pt)
                    print(f"[POSE]   → kpt_only: started new pose instance")
                    return
                else:
                    # 直接绘制 bbox（无需按空格激活）— 需要【拖拽】，单纯点击不会产生 bbox
                    self.drawing = True
                    self.start_pt = clamped_pt
                    print(f"[POSE]   → start drag for bbox (必须拖拽 >5px 才会创建实例)")
            return

        # ---------------- 常规绘图起点 ----------------
        if not self.is_inside_image(pt) and not self.drawing: return
        if event.button() == Qt.LeftButton:
            if self.mode in [CanvasMode.RECT, CanvasMode.RBOX]:
                self.drawing = True
                self.start_pt = clamped_pt
            elif self.mode == CanvasMode.POLY:
                if len(self.poly_pts) > 2:
                    dist = ((clamped_pt.x() - self.poly_pts[0].x()) ** 2 + (
                            clamped_pt.y() - self.poly_pts[0].y()) ** 2) ** 0.5
                    if dist < 10:
                        self.finish_poly_shape()
                        return
                self.poly_pts.append(clamped_pt)
                self.update_temp_poly()
            elif self.mode == CanvasMode.POINT:
                shape = PointShape(clamped_pt)
                self.shape_drawn.emit(shape)
        elif event.button() == Qt.RightButton:
            if self.mode == CanvasMode.POLY and len(self.poly_pts) > 2:
                self.finish_poly_shape()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        # ---------------- 右键框选删除松手 ----------------
        if event.button() == Qt.RightButton and self.right_dragging:
            self.right_dragging = False
            if self.right_drag_rect:
                # 获取框选矩形
                drag_rect = self.right_drag_rect.rect()
                self.removeItem(self.right_drag_rect)
                self.right_drag_rect = None

                # 删除矩形区域内所有标注形状
                if drag_rect.width() > 5 and drag_rect.height() > 5:
                    removed = False
                    for item in self.items(drag_rect):
                        if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                            if not getattr(item, 'is_temp', False):
                                self.removeItem(item)
                                removed = True
                    if removed:
                        self.shape_deleted.emit()
            return

        if self.sam_enabled: return

        # ---------------- 位姿模式 bbox 完成 ----------------
        if self.mode == CanvasMode.POSE and event.button() == Qt.LeftButton and self.drawing:
            if self.pose_kpt_only:
                self.drawing = False
                print(f"[POSE] release: kpt_only mode, no bbox drawn")
                return
            self.drawing = False
            self.pose_active = True
            if self.temp_item:
                pt = self.clamp_point(event.scenePos())
                rect = QRectF(min(self.start_pt.x(), pt.x()), min(self.start_pt.y(), pt.y()),
                              abs(pt.x() - self.start_pt.x()), abs(pt.y() - self.start_pt.y()))
                self.removeItem(self.temp_item)
                self.temp_item = None

                if rect.width() > 5 and rect.height() > 5:
                    # 创建 PoseShape（不带关键点，后续点击添加）
                    shape = PoseShape(rect, label="")
                    self.shape_drawn.emit(shape)
                    print(f"[POSE] release: bbox {rect.width():.0f}x{rect.height():.0f} → shape_drawn emitted")
                else:
                    print(f"[POSE] release: bbox too small ({rect.width():.0f}x{rect.height():.0f} ≤5px), dropped")
            else:
                print(f"[POSE] release: no temp_item (pure click, no drag) → nothing created")
            self.state_changed.emit()
            return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.temp_item:
                pt = self.clamp_point(event.scenePos())
                rect = QRectF(min(self.start_pt.x(), pt.x()), min(self.start_pt.y(), pt.y()),
                              abs(pt.x() - self.start_pt.x()), abs(pt.y() - self.start_pt.y()))
                self.removeItem(self.temp_item)
                self.temp_item = None

                if rect.width() > 5 and rect.height() > 5:
                    if self.mode == CanvasMode.RECT:
                        self.shape_drawn.emit(RectShape(rect))

                    # 手动松开鼠标完成绘制时，实例化新的 RotatedRectShape
                    elif self.mode == CanvasMode.RBOX:
                        cx, cy = rect.center().x(), rect.center().y()
                        w, h = rect.width(), rect.height()
                        self.shape_drawn.emit(RotatedRectShape(cx, cy, w, h, 0))

        self.state_changed.emit()

    def mouseDoubleClickEvent(self, event):
        pt = event.scenePos()
        if not self.is_inside_image(pt): return

        for item in self.items(pt):
            if isinstance(item, (HandleItem, PolyShape, RectShape, PointShape, RotatedRectShape)) and not getattr(item,
                                                                                                                  'is_temp',
                                                                                                                  False):
                shape_item = item.parentItem() if isinstance(item, HandleItem) else item
                self.shape_double_clicked.emit(shape_item)
                return

        if event.button() == Qt.LeftButton and self.mode == CanvasMode.POLY and not self.sam_enabled and len(
                self.poly_pts) > 2:
            self.finish_poly_shape()
        else:
            super().mouseDoubleClickEvent(event)

    def update_temp_poly(self, mouse_pos=None):
        display_pts = self.poly_pts.copy()
        if mouse_pos is not None: display_pts.append(mouse_pos)
        if len(display_pts) < 2:
            if self.temp_item: self.removeItem(self.temp_item); self.temp_item = None
            return
        if self.temp_item and isinstance(self.temp_item, PolyShape):
            self.temp_item.setPolygon(QPolygonF(display_pts))
        else:
            if self.temp_item: self.removeItem(self.temp_item)
            self.temp_item = PolyShape(QPolygonF(display_pts), is_temp=True)
            self.addItem(self.temp_item)

    def finish_poly_shape(self):
        shape = PolyShape(QPolygonF(self.poly_pts))
        self.poly_pts.clear()
        if self.temp_item:
            self.removeItem(self.temp_item)
            self.temp_item = None
        self.shape_drawn.emit(shape)

    def cancel_drawing(self):
        self.drawing = False
        self.poly_pts.clear()
        if self.temp_item:
            self.removeItem(self.temp_item)
            self.temp_item = None
        if self.sam_hover_item:
            self.removeItem(self.sam_hover_item)
            self.sam_hover_item = None

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key_Backspace or key == Qt.Key_Delete:
            for item in self.selectedItems():
                self.removeItem(item)
            self.state_changed.emit()
            return
        elif key == Qt.Key_Z and modifiers == Qt.ControlModifier:
            if self.mode == CanvasMode.POLY and not self.sam_enabled and len(self.poly_pts) > 0:
                self.poly_pts.pop()
                self.update_temp_poly()
                return
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            if self.mode == CanvasMode.POLY and len(self.poly_pts) > 2:
                self.finish_poly_shape()
                return
        elif key == Qt.Key_Escape:
            self.cancel_drawing()
            return
        elif key in [Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V]:
            items = self.selectedItems()
            if items and isinstance(items[0], RotatedRectShape):
                delta = 0
                if key == Qt.Key_Z:
                    delta = -5
                elif key == Qt.Key_X:
                    delta = -1
                elif key == Qt.Key_C:
                    delta = 1
                elif key == Qt.Key_V:
                    delta = 5
                if delta != 0:
                    item = items[0]
                    item.setRotation(item.rotation() + delta)
                    self.state_changed.emit()
                return

        super().keyPressEvent(event)

    def _resolve_pose_shape(self, item):
        if isinstance(item, PoseShape) and not getattr(item, "is_temp", False):
            return item
        if isinstance(item, KeypointItem):
            parent = item.parentItem()
            if isinstance(parent, PoseShape) and not getattr(parent, "is_temp", False):
                return parent
        return None

    def _find_clicked_pose(self, scene_pt):
        for item in self.items(scene_pt):
            pose = self._resolve_pose_shape(item)
            if pose is not None:
                return pose
        return None

    def _find_incomplete_pose(self):
        for item in self.items():
            if isinstance(item, PoseShape) and not getattr(item, "is_temp", False):
                if not item.is_kpt_complete():
                    return item
        return None

    def _pose_keypoint_target(self):
        pending = [item for item in self.items()
                   if isinstance(item, PoseShape) and not getattr(item, "is_temp", False)
                   and item.kpt_count > 0 and not item.is_kpt_complete()]
        selected = [item for item in pending if item.isSelected()]
        if len(selected) == 1:
            return selected[0]
        return pending[0] if len(pending) == 1 else None

    def _add_pose_keypoint(self, pose_shape, scene_pt):
        if pose_shape.is_kpt_complete():
            return False
        pose_shape.add_keypoint(scene_pt)
        if pose_shape.bbox_from_keypoints:
            pose_shape.update_bbox_from_keypoints()
            pose_shape.set_bbox_visible(not self.pose_kpt_only)
        # 层级标注：用户放完 4 个角点 (KP0-3) 后自动放置 KP4 (L2 子类, 虚拟点)
        if (self.hierarchical_active
                and pose_shape.kpt_count >= 5
                and len(pose_shape.keypoint_items) == 4):
            self._auto_place_hier_metadata(pose_shape)
        self.state_changed.emit()
        return True

    def ensure_hier_l2_keypoint(self, pose_shape, l2_code, kpt_names=None, kpt_count=5):
        """将实例升级/同步为分层：保证有 KP4，visibility = L2 码。

        需已有至少 4 个角点。返回是否成功写回 KP4。
        """
        if pose_shape is None or getattr(pose_shape, "is_temp", False):
            return False
        if len(pose_shape.keypoint_items) < 4:
            return False
        pose_shape.is_hierarchical = True
        if kpt_names is not None:
            pose_shape.kpt_names = list(kpt_names)
        if kpt_count is not None:
            pose_shape.kpt_count = int(kpt_count)
        if len(pose_shape.keypoint_items) == 4:
            xs = [pose_shape.mapToScene(k.pos()).x() for k in pose_shape.keypoint_items[:4]]
            ys = [pose_shape.mapToScene(k.pos()).y() for k in pose_shape.keypoint_items[:4]]
            center_scene = QPointF(sum(xs) / 4.0, sum(ys) / 4.0)
            pose_shape.add_keypoint(center_scene)
        if len(pose_shape.keypoint_items) < 5:
            return False
        kpt = pose_shape.keypoint_items[4]
        kpt.visibility = int(l2_code)
        colors = {
            0: QColor(100, 100, 100),
            1: QColor(255, 165, 0),
            2: QColor(0, 255, 0),
            3: QColor(100, 149, 237),
        }
        kpt.setBrush(QBrush(colors.get(kpt.visibility, QColor(0, 255, 0))))
        return True

    def _auto_place_hier_metadata(self, pose_shape):
        """层级标注：在 bbox 中心自动放置 KP4 (L2 子类, 虚拟关键点)。

        坐标复用 bbox 中心 (由 4 角推出), visibility 编码 L2 子类码。
        L2 仅对 BO/RO 有意义; 非 BO/RO 兵种 L2 code 恒为 0 (none)。
        """
        self.ensure_hier_l2_keypoint(
            pose_shape, self.hier_meta_l2_code,
            kpt_names=getattr(pose_shape, "kpt_names", None),
            kpt_count=getattr(pose_shape, "kpt_count", 5),
        )

    def _start_kpt_only_pose(self, scene_pt):
        """仅标点模式：点击空白处创建实例并放置第一个关键点。"""
        rect = QRectF(scene_pt.x() - 1, scene_pt.y() - 1, 2, 2)
        shape = PoseShape(rect, label="")
        shape.bbox_from_keypoints = True
        self.shape_drawn.emit(shape)
        if shape.kpt_count > 0:
            shape.set_bbox_visible(False)
            shape.add_keypoint(scene_pt)
            shape.update_bbox_from_keypoints()
            shape.set_bbox_visible(False)
        self.state_changed.emit()
