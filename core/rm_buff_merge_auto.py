# -*- coding: utf-8 -*-
"""
能量机关 (buff-merge) SAM + YOLO 半自动标注模块

工作流:
1. YOLO pose 权重粗略标注 bbox + 5 关键点
2. SAM 文字提示 (red_target / blue_target) 在 bbox 内分割轮廓
3. 多轮廓合并后计算极轴极值点，精修第 1/3/4/5 关键点
"""
import os
import glob
from PySide6.QtCore import QObject, Signal, QThread, QRectF, QSettings

from core.buff_merge_geometry import compute_circle_keypoints
from core.pose_shapes import FIXED_KPT_INDEX, ADJUST_KPT_INDICES


CLASS_TO_PROMPT = {
    "RT": "red_target",
    "BT": "blue_target",
    "0": "red_target",
    "1": "blue_target",
    "red_target": "red_target",
    "blue_target": "blue_target",
}

YOLO_ID_TO_LABEL = {0: "RT", 1: "BT"}


class YoloRoughWorker(QThread):
    finished = Signal(bool, list, str)

    def __init__(self, model, image_path, conf, iou, imgsz):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def run(self):
        try:
            from core.yolo_client import YoloClient
            client = YoloClient()
            client.model = self.model
            client.task = self.model.task
            client.names = self.model.names
            shapes = client.predict_image(
                self.image_path, conf=self.conf, iou=self.iou, imgsz=self.imgsz
            )
            detections = [s for s in shapes if s.get("type") == "pose"]
            self.finished.emit(True, detections, f"检测到 {len(detections)} 个目标")
        except Exception as e:
            self.finished.emit(False, [], str(e))


class RMBuffMergeAuto(QObject):
    """能量机关 buff-merge 自动标注控制器"""

    yolo_rough_done = Signal(str, list)  # image_path, detections
    keypoints_refined = Signal(object, list)
    rough_step_placed = Signal(object, list)
    error_message = Signal(str)
    status_message = Signal(str)

    ADJUST_KPT_INDICES = (1, 2, 3, 4)
    FIXED_KPT_INDEX = 0
    BBOX_EXPAND_RATIO = 0.25

    def __init__(self, sam_client, parent=None):
        super().__init__(parent)
        self.sam_client = sam_client
        self.enabled = False
        self._phase = "idle"
        self._pending_shapes = []
        self._pending_prompt = ""
        self._pending_sam_results = []
        self._point_fallback_index = 0
        self._yolo_model = None
        self._yolo_worker = None
        self._load_worker = None
        self.sam_client.text_result_ready.connect(self._on_text_result)
        self.sam_client.inference_result.connect(self._on_point_result)

    @property
    def phase(self):
        return self._phase

    def set_enabled(self, enabled):
        self.enabled = enabled
        if not enabled:
            self._reset()

    @staticmethod
    def default_weight_path():
        settings = QSettings("ArmorLabel", "settings")
        settings.beginGroup("Parameters")
        saved = str(settings.value("BUFF_YOLO_WEIGHT", ""))
        settings.endGroup()
        if saved and os.path.isfile(saved):
            return saved

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        patterns = [
            os.path.join(base, "models", "buff*.pt"),
            os.path.join(base, "models", "buff*.onnx"),
            os.path.join(base, "models", "buff*.engine"),
            os.path.join(base, "models", "*buff*pose*.pt"),
            os.path.join(base, "models", "*energy*.pt"),
            os.path.join(base, "stored_weights", "model", "buff", "*.pt"),
            os.path.join(base, "stored_weights", "model", "buff", "*.onnx"),
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[0]
        return ""

    def load_yolo_async(self, weight_path=None):
        path = weight_path or self.default_weight_path()
        if not path or not os.path.isfile(path):
            self.error_message.emit("未找到能量机关 YOLO 权重，请在面板中选择")
            return

        if self._load_worker and self._load_worker.isRunning():
            return

        from core.yolo_client import YoloLoadWorker
        self._load_worker = YoloLoadWorker(path)
        self._load_worker.loaded.connect(self._on_yolo_loaded)
        self._load_worker.start()
        self.status_message.emit("正在加载 YOLO 模型...")

    def _on_yolo_loaded(self, model, success, msg):
        if success:
            if model.task != "pose":
                self.error_message.emit(f"权重任务类型应为 pose，当前为 {model.task}")
                return
            self._yolo_model = model
            self.status_message.emit(msg)
        else:
            self.error_message.emit(msg)

    def _sam_ready(self):
        if not self.sam_client.model:
            self.error_message.emit("SAM 模型未就绪")
            return False
        parent = self.parent()
        if parent is not None and getattr(parent, "sam_loading", False):
            self.error_message.emit("SAM 正在提取图片特征，请稍候再试")
            return False
        if not self.sam_client.inference_worker.inference_state:
            self.error_message.emit("SAM 图像特征未就绪，请等待状态栏显示「分析完成」")
            return False
        return True

    def request_yolo_rough(self, image_path):
        if not self.enabled:
            return
        if not self._yolo_model:
            self.load_yolo_async()
            self.error_message.emit("YOLO 模型未加载，正在后台加载...")
            return
        if self._yolo_worker and self._yolo_worker.isRunning():
            return

        settings = QSettings("ArmorLabel", "settings")
        settings.beginGroup("Parameters")
        conf = float(settings.value("YOLO_CONF", 0.25))
        iou = float(settings.value("YOLO_IOU", 0.7))
        imgsz = int(settings.value("YOLO_IMGSZ", 640))
        settings.endGroup()

        self._phase = "yolo_rough"
        self._yolo_image_path = image_path
        self.status_message.emit("YOLO 粗略标注中...")
        self._yolo_worker = YoloRoughWorker(self._yolo_model, image_path, conf, iou, imgsz)
        self._yolo_worker.finished.connect(self._on_yolo_rough_finished)
        self._yolo_worker.start()

    def _on_yolo_rough_finished(self, success, detections, msg):
        bound_path = getattr(self, "_yolo_image_path", None) or ""
        self._phase = "idle"
        if not success:
            self.error_message.emit(msg)
            return
        if not detections:
            self.error_message.emit("YOLO 未检测到能量机关目标")
            return
        self.yolo_rough_done.emit(bound_path, detections)
        self.status_message.emit(msg)

    def resolve_pose_label(self, yolo_label):
        label = str(yolo_label).strip()
        if label in ("RT", "BT"):
            return label
        if label.isdigit():
            return YOLO_ID_TO_LABEL.get(int(label), label)
        if self._yolo_model and self._yolo_model.names:
            for cls_id, name in self._yolo_model.names.items():
                if str(name) == label:
                    return YOLO_ID_TO_LABEL.get(int(cls_id), str(name))
        lower = label.lower()
        if "red" in lower or label == "0":
            return "RT"
        if "blue" in lower or label == "1":
            return "BT"
        return label

    def prompt_for_label(self, label):
        return CLASS_TO_PROMPT.get(label, CLASS_TO_PROMPT.get(str(label), label))

    def request_sam_refine(self, pose_shapes, prompt_text=None):
        if not self.enabled:
            return
        if self._phase != "idle":
            self.status_message.emit("SAM 精修进行中...")
            return
        if not self._sam_ready():
            return

        shapes = [s for s in pose_shapes if s is not None]
        if not shapes:
            self.error_message.emit("请先选择或创建位姿实例")
            return

        self._pending_shapes = shapes
        self._pending_prompt = self._resolve_prompt(shapes, prompt_text)
        if self._pending_prompt not in ("red_target", "blue_target"):
            self.error_message.emit("无法推断提示词，请输入 red_target 或 blue_target")
            return

        self._phase = "sam_refine"
        self._point_fallback_index = 0
        self.status_message.emit(f"SAM 文字精修: {self._pending_prompt}")
        print(f"  [buff] request_sam_refine prompt={self._pending_prompt} shapes={len(shapes)}")
        self.sam_client.request_text_inference(self._pending_prompt)

    def _resolve_prompt(self, shapes, prompt_text=None):
        if prompt_text and str(prompt_text).strip():
            return str(prompt_text).strip()
        for shape in shapes:
            label = getattr(shape, "label", "") or ""
            candidate = self.prompt_for_label(label)
            if candidate in ("red_target", "blue_target"):
                return candidate
        label = getattr(shapes[0], "label", "") or ""
        return self.prompt_for_label(label)

    def _on_text_result(self, results, prompt_text):
        if self._phase != "sam_refine" or prompt_text != self._pending_prompt:
            return
        if not self.enabled:
            self._reset()
            return

        print(f"  [buff] text_result prompt={prompt_text} global_results={len(results)}")
        self._pending_sam_results = results or []

        if not self._pending_sam_results:
            self.status_message.emit("文字提示无结果，尝试 bbox 中心点分割...")
            self._start_point_fallback()
            return

        self._apply_refine_from_results(self._pending_sam_results)

    def _apply_refine_from_results(self, sam_results):
        ok_count = 0
        fail_count = 0
        try:
            for shape in self._pending_shapes:
                refined = self._refine_shape_keypoints(shape, sam_results)
                if refined is None:
                    fail_count += 1
                    continue
                self.keypoints_refined.emit(shape, refined)
                ok_count += 1
        except Exception as e:
            self.error_message.emit(f"SAM 精修失败: {e}")
            self._reset()
            return

        if ok_count == 0:
            if sam_results:
                self.status_message.emit("bbox 内无匹配轮廓，尝试点提示回退...")
                self._start_point_fallback()
                return
            self.error_message.emit(
                f"SAM 未检测到 '{self._pending_prompt}'，请确认提示词并等待图片特征分析完成"
            )
        else:
            msg = f"SAM 精修完成: {ok_count}/{len(self._pending_shapes)}"
            if fail_count:
                msg += f" ({fail_count} 个 bbox 内无匹配，可重试点提示)"
            self.status_message.emit(msg)
        self._reset()

    def _start_point_fallback(self):
        if not self._pending_shapes:
            self._reset()
            return
        self._phase = "sam_point_refine"
        self._point_fallback_index = 0
        parent = self.parent()
        if parent is not None and hasattr(parent, "scene"):
            self._saved_sam_enabled = parent.scene.sam_enabled
            parent.scene.sam_enabled = False
        self._request_next_point_fallback()

    def _request_next_point_fallback(self):
        while self._point_fallback_index < len(self._pending_shapes):
            shape = self._pending_shapes[self._point_fallback_index]
            pt = self._sam_seed_point(shape)
            if pt is None:
                self._point_fallback_index += 1
                continue
            x, y = pt
            print(f"  [buff] point_fallback #{self._point_fallback_index} at ({x:.1f},{y:.1f})")
            self.sam_client.request_inference(x, y, is_click=True)
            return
        self.error_message.emit("SAM 精修失败：文字与点提示均未在 bbox 内找到轮廓")
        self._reset()

    def _on_point_result(self, poly_pts, rect_xywh, rect_obb, score, is_click):
        if self._phase != "sam_point_refine" or not self.enabled:
            return
        if self._point_fallback_index >= len(self._pending_shapes):
            return

        shape = self._pending_shapes[self._point_fallback_index]
        if poly_pts:
            sam_result = [{"poly_pts": poly_pts, "rect": rect_xywh}]
            refined = self._refine_shape_keypoints(shape, sam_result, strict_bbox=False)
            if refined is not None:
                self.keypoints_refined.emit(shape, refined)
                print(f"  [buff] point_fallback ok shape #{self._point_fallback_index}")

        self._point_fallback_index += 1
        if self._point_fallback_index >= len(self._pending_shapes):
            self.status_message.emit("SAM 点提示精修完成")
            self._reset()
        else:
            self._request_next_point_fallback()

    @staticmethod
    def _sam_seed_point(pose_shape):
        kpts = pose_shape.get_keypoints_data()
        if len(kpts) > RMBuffMergeAuto.FIXED_KPT_INDEX:
            k = kpts[RMBuffMergeAuto.FIXED_KPT_INDEX]
            if k[2] > 0:
                return float(k[0]), float(k[1])
        bbox = pose_shape.rect()
        if bbox.isEmpty():
            return None
        return bbox.center().x(), bbox.center().y()

    def _expanded_bbox(self, bbox):
        pad_x = bbox.width() * self.BBOX_EXPAND_RATIO
        pad_y = bbox.height() * self.BBOX_EXPAND_RATIO
        return bbox.adjusted(-pad_x, -pad_y, pad_x, pad_y)

    def _filter_sam_results_for_bbox(self, bbox, sam_results, strict_bbox=True):
        if not sam_results:
            return []

        search = bbox if strict_bbox else self._expanded_bbox(bbox)
        matched = []
        for res in sam_results:
            rx, ry, rw, rh = res["rect"]
            bar_rect = QRectF(rx, ry, rw, rh)
            if not search.contains(bar_rect):
                continue
            poly_pts = res.get("poly_pts", [])
            if not poly_pts:
                continue
            all_inside = True
            for px, py in poly_pts:
                if not search.contains(QRectF(px, py, 1, 1)):
                    all_inside = False
                    break
            if all_inside:
                matched.append(res)

        if matched:
            return matched

        if strict_bbox:
            return self._filter_sam_results_for_bbox(bbox, sam_results, strict_bbox=False)

        bcx, bcy = bbox.center().x(), bbox.center().y()
        best = None
        best_dist = float("inf")
        max_dist = max(bbox.width(), bbox.height()) * 1.5
        for res in sam_results:
            rx, ry, rw, rh = res["rect"]
            cx, cy = rx + rw / 2, ry + rh / 2
            dist = (cx - bcx) ** 2 + (cy - bcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = res
        if best is not None and best_dist <= max_dist ** 2:
            return [best]
        return []

    def _refine_shape_keypoints(self, pose_shape, sam_results, strict_bbox=True):
        bbox = pose_shape.rect()
        if bbox.isEmpty():
            return None

        filtered = self._filter_sam_results_for_bbox(bbox, sam_results, strict_bbox=strict_bbox)
        if not filtered:
            return None

        polys = [r["poly_pts"] for r in filtered if r.get("poly_pts")]
        if not polys:
            return None

        yolo_kpts = pose_shape.get_keypoints_data()
        if len(yolo_kpts) < 5:
            return None

        extremes = compute_circle_keypoints(polys, yolo_kpts, bbox)
        if not extremes:
            return None

        new_kpts = [list(k) for k in yolo_kpts]
        for i, pt in enumerate(extremes):
            if i < len(new_kpts):
                new_kpts[i][0] = float(pt[0])
                new_kpts[i][1] = float(pt[1])

        return new_kpts

    @staticmethod
    def detection_to_pose_data(detection, pose_label):
        pts = detection.get("points", [])
        if len(pts) < 2:
            return None
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        kpts = detection.get("keypoints") or []
        if len(kpts) < 5:
            return None
        return rect, pose_label, kpts

    def _reset(self):
        parent = self.parent()
        if parent is not None and hasattr(parent, "scene") and hasattr(self, "_saved_sam_enabled"):
            parent.scene.sam_enabled = self._saved_sam_enabled
            del self._saved_sam_enabled
        self._phase = "idle"
        self._pending_shapes = []
        self._pending_prompt = ""
        self._pending_sam_results = []
        self._point_fallback_index = 0
        self._yolo_image_path = None
        if self._yolo_worker and self._yolo_worker.isRunning():
            try:
                self._yolo_worker.finished.disconnect(self._on_yolo_rough_finished)
            except Exception:
                pass
