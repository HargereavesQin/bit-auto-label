# -*- coding: utf-8 -*-
"""
LoRA 装甲板端到端自动标注 — 与 core/rm_armor_auto.py 完全解耦。

手动: 输入类别名 → SAM-LoRA 文本检测 → minAreaRect 4 角点 → PoseShape
自动: 串行遍历 14 类 → 同上
"""
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal


class LoraArmorAuto(QObject):
    """端到端 LoRA-SAM 装甲板标注状态机。"""

    # category, bbox [x,y,w,h], keypoints [(x,y)*4], score
    pose_shape_placed = Signal(str, list, list, float)
    auto_finished = Signal()
    status_message = Signal(str, str)  # msg, level
    # done, total, current_cls, placed_count
    progress_changed = Signal(int, int, str, int)

    CLASS_NAMES = (
        "BG", "B1", "B2", "B3", "B4", "BO", "BB",
        "RG", "R1", "R2", "R3", "R4", "RO", "RB",
    )

    def __init__(self, lora_sam_client, pose_config=None, parent=None):
        super().__init__(parent)
        self.client = lora_sam_client
        self.pose_config = pose_config
        self.client.text_result_ready.connect(self._on_text_result)
        self._mode = "idle"  # idle | manual | auto
        self._current_category = None
        self._auto_queue = []
        self._auto_total = 0
        self._auto_done = 0
        self._all_results = {}
        self._placed_count = 0
        self.last_result_msg = ""
        self.last_result_level = "info"

    @property
    def is_busy(self):
        return self._mode != "idle"

    def request_manual(self, category: str):
        """手动模式：检测单个类别。"""
        category = (category or "").strip()
        if not category:
            self.status_message.emit("请输入类别名（如 B1）", "warning")
            return
        if not self.client.model:
            self.status_message.emit("LoRA-SAM 模型未就绪", "error")
            return
        if self._mode != "idle":
            self.status_message.emit(f"LoRA 标注进行中 ({self._mode})，请稍候", "warning")
            return
        self._mode = "manual"
        self._current_category = category
        self._placed_count = 0
        self._auto_total = 1
        self._auto_done = 0
        self.status_message.emit(f"LoRA 手动检测: {category}…", "info")
        self.progress_changed.emit(0, 1, category, 0)
        self.client.request_text_inference(category)

    def request_auto(self, class_names=None):
        """自动模式：串行遍历类别队列。"""
        if not self.client.model:
            self.status_message.emit("LoRA-SAM 模型未就绪", "error")
            return
        if self._mode != "idle":
            self.status_message.emit(f"LoRA 标注进行中 ({self._mode})，请稍候", "warning")
            return
        names = list(class_names) if class_names else list(self.CLASS_NAMES)
        if not names:
            self.status_message.emit("无可用类别", "warning")
            return
        self._mode = "auto"
        self._auto_queue = names
        self._auto_total = len(names)
        self._auto_done = 0
        self._all_results = {}
        self._placed_count = 0
        self.status_message.emit(f"LoRA 自动标注: 共 {len(names)} 类…", "info")
        self.progress_changed.emit(0, self._auto_total, "", 0)
        self._process_next_class()

    def cancel(self):
        """中断当前流程。"""
        was_busy = self._mode != "idle"
        self._auto_queue.clear()
        self._mode = "idle"
        self._current_category = None
        if was_busy:
            self.last_result_msg = "LoRA 标注已取消"
            self.last_result_level = "warning"
            self.status_message.emit(self.last_result_msg, "warning")
            self.auto_finished.emit()

    def _process_next_class(self):
        if not self._auto_queue:
            self._finalize_auto()
            return
        cls = self._auto_queue.pop(0)
        self._current_category = cls
        remaining = len(self._auto_queue)
        self.status_message.emit(
            f"LoRA 检测 [{cls}]（剩余 {remaining} 类）…", "info"
        )
        self.progress_changed.emit(
            self._auto_done, self._auto_total, cls, self._placed_count
        )
        self.client.request_text_inference(cls)

    def _on_text_result(self, results, prompt_text):
        if self._mode == "idle":
            return
        # 忽略过期/不匹配的回调
        if self._current_category and prompt_text != self._current_category:
            return

        if results:
            self._all_results[prompt_text] = results
            for res in results:
                self._place_pose_shape(prompt_text, res)
        else:
            self._all_results.setdefault(prompt_text, [])

        if self._mode == "manual":
            n = len(results) if results else 0
            self._auto_done = 1
            level = "success" if n else "warning"
            msg = f"LoRA 手动完成 [{prompt_text}]: {n} 个目标"
            self.last_result_msg = msg
            self.last_result_level = level
            self.status_message.emit(msg, level)
            self.progress_changed.emit(1, 1, prompt_text, self._placed_count)
            self._mode = "idle"
            self._current_category = None
            self.auto_finished.emit()
        elif self._mode == "auto":
            self._auto_done += 1
            self._process_next_class()

    def _place_pose_shape(self, category, result):
        """从 SAM mask 轮廓用 minAreaRect 提取 bbox + 4 角点关键点。"""
        poly_pts = result.get("poly_pts") or []
        if len(poly_pts) < 3:
            return

        contour = np.array(poly_pts, dtype=np.float32)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # 4×2

        # 角点顺序规范化：按角度排序，尽量稳定 TL→TR→BR→BL 近似
        keypoints = self._order_box_points(box)

        bbox = result.get("rect")
        if not bbox or len(bbox) < 4:
            xs = [p[0] for p in keypoints]
            ys = [p[1] for p in keypoints]
            x, y = min(xs), min(ys)
            bbox = [x, y, max(xs) - x, max(ys) - y]
        else:
            bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

        score = float(result.get("score", 0.0))
        self._placed_count += 1
        self.pose_shape_placed.emit(category, bbox, keypoints, score)

    @staticmethod
    def _order_box_points(box):
        """将 minAreaRect 四角点排成近似 TL, TR, BR, BL。"""
        pts = np.asarray(box, dtype=np.float32)
        # 按 y 再按 x 粗分上下
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        # OpenCV 常用: tl=最小和, br=最大和, tr=最小差, bl=最大差
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        ordered = [tl, tr, br, bl]
        return [(float(p[0]), float(p[1])) for p in ordered]

    def _finalize_auto(self):
        total = self._placed_count
        classes_hit = sum(1 for v in self._all_results.values() if v)
        level = "success" if total > 0 else "warning"
        if total > 0:
            msg = f"LoRA 自动标注完成: {total} 个目标 / {classes_hit} 类命中"
        else:
            msg = "LoRA 自动标注完成: 未检测到目标"
        self.last_result_msg = msg
        self.last_result_level = level
        self.status_message.emit(msg, level)
        self.progress_changed.emit(
            self._auto_total, self._auto_total, "", self._placed_count
        )
        self._mode = "idle"
        self._current_category = None
        self.auto_finished.emit()
