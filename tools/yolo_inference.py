# -*- coding: utf-8 -*-
"""
YOLO 推理查看器 - 支持图片和视频的 YOLO 推理显示
支持实时阈值调整、视频暂停/继续、结果叠加显示、独立全屏显示窗口
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QGridLayout,
    QProgressBar, QGroupBox, QCheckBox,
    QRadioButton, QButtonGroup, QSlider, QFrame, QDialog, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMutex, QWaitCondition
from PySide6.QtGui import QFont, QImage, QPixmap


# =========================================================
# 加载编辑风样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()


CATEGORY_COLORS = [
    (255, 107, 107),
    (78, 205, 196),
    (69, 183, 209),
    (150, 206, 180),
    (255, 234, 167),
    (255, 159, 243),
    (0, 210, 211),
    (255, 127, 80),
    (147, 112, 219),
    (60, 179, 113),
]


class MLPLoadWorker(QThread):
    loaded = Signal(object, bool, str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        self._stop_flag = False
        try:
            from core.mlp_classifier import MLPClassifier
            classifier = MLPClassifier(self.model_path)
            if self._stop_flag:
                return
            labels_preview = ', '.join(classifier.labels[:5])
            if len(classifier.labels) > 5:
                labels_preview += '...'
            msg = (f"MLP模型加载成功\n"
                   f"格式: {classifier.model_type}\n"
                   f"输入尺寸: {classifier.input_size}\n"
                   f"类别数: {len(classifier.labels)}\n"
                   f"类别: {labels_preview}")
            self.loaded.emit(classifier, True, msg)
        except ImportError as e:
            self.loaded.emit(None, False, f"缺少依赖库: {str(e)}\n请运行: pip install onnxruntime")
        except Exception as e:
            self.loaded.emit(None, False, f"MLP模型加载失败: {e}")


def draw_inference_results(image, results, task, names, fps=0.0, ms=0.0,
                           mlp_classifier=None, mlp_conf_threshold=0.5,
                           mlp_roi_mode="perspective", collect_mlp_rois=False,
                           hide_negative=False):
    from core.mlp_classifier import MLPClassifier

    annotated = image.copy()
    h, w = annotated.shape[:2]
    total_count = 0
    class_counts = {}
    total_mlp_ms = 0.0
    mlp_class_counts = {}
    mlp_rois = []

    all_corrected_kpts = [None] * len(results)
    all_filtered_indices = [set() for _ in range(len(results))]
    for ri, result in enumerate(results):
        boxes = result.boxes
        keypoints = result.keypoints if hasattr(result, 'keypoints') else None
        obb = result.obb if hasattr(result, 'obb') else None

        kp_xy = None
        kp_conf = None
        if task == "pose" and keypoints is not None:
            kp_xy = keypoints.xy.cpu().numpy() if keypoints.xy is not None else None
            kp_conf = keypoints.conf.cpu().numpy() if hasattr(keypoints, 'conf') and keypoints.conf is not None else None

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for j in range(len(xyxy)):
                mlp_label = "?"
                mlp_conf_str = "?"
                corrected_kpts = None
                if task == "pose" and mlp_classifier is not None and kp_xy is not None and j < len(kp_xy):
                    kpts_j = kp_xy[j]
                    has_valid_kpts = False
                    if kp_conf is not None and j < len(kp_conf):
                        has_valid_kpts = bool(np.all(kp_conf[j] > 0.3))
                    elif kp_xy is not None:
                        has_valid_kpts = True
                    if has_valid_kpts:
                        try:
                            t0 = time.time()
                            roi = None
                            is_1ch = (mlp_classifier.input_channels == 1)
                            if mlp_roi_mode == "bbox":
                                roi = MLPClassifier.extract_armor_roi_bbox(
                                    image, xyxy[j],
                                    output_size=mlp_classifier.input_size,
                                    binarize=is_1ch
                                )
                            else:
                                refined_kpts, _ = MLPClassifier.refine_keypoints_and_clean(
                                    image, kpts_j, xyxy[j]
                                )
                                if refined_kpts is not None:
                                    corrected_kpts = refined_kpts
                                    all_corrected_kpts[ri] = all_corrected_kpts[ri] or {}
                                    all_corrected_kpts[ri][j] = corrected_kpts
                                roi = MLPClassifier.extract_armor_roi(
                                    image, xyxy[j], kpts_j,
                                    output_size=mlp_classifier.input_size,
                                    corrected_kpts=corrected_kpts,
                                    binarize=is_1ch
                                )
                            if roi is not None:
                                mlp_name, mlp_conf_val, _ = mlp_classifier.predict(roi)
                                total_mlp_ms += (time.time() - t0) * 1000
                                if mlp_conf_val >= mlp_conf_threshold:
                                    mlp_label = mlp_name
                                    mlp_conf_str = f"{mlp_conf_val:.2f}"
                                    mlp_class_counts[mlp_name] = mlp_class_counts.get(mlp_name, 0) + 1
                                else:
                                    mlp_label = "?"
                                    mlp_conf_str = f"{mlp_conf_val:.2f}"
                                if collect_mlp_rois:
                                    debug_stages = None
                                    if mlp_roi_mode == "perspective" and corrected_kpts is not None:
                                        debug_stages = MLPClassifier.extract_armor_roi_debug(
                                            image, xyxy[j], kpts_j,
                                            output_size=mlp_classifier.input_size,
                                            corrected_kpts=corrected_kpts,
                                            binarize=True
                                        )
                                    mlp_rois.append((roi.copy(), mlp_label, mlp_conf_val, debug_stages))
                        except Exception:
                            pass

                if hide_negative and mlp_classifier is not None and task == "pose" and mlp_label in ("negative", "?"):
                    all_filtered_indices[ri].add(j)
                    continue

                x1, y1, x2, y2 = xyxy[j]
                conf = float(confs[j])
                cls_id = int(clss[j])
                label = names.get(cls_id, f"class_{cls_id}")
                class_counts[label] = class_counts.get(label, 0) + 1
                color = CATEGORY_COLORS[cls_id % len(CATEGORY_COLORS)]
                total_count += 1

                if mlp_classifier is not None and task == "pose":
                    text = f"{label} {conf:.2f} | {mlp_label} {mlp_conf_str}"
                else:
                    text = f"{label} {conf:.2f}"

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_bg_h = th + 6
                cv2.rectangle(annotated, (int(x1), int(y1) - text_bg_h), (int(x1) + tw + 4, int(y1)), color, -1)
                cv2.putText(annotated, text, (int(x1) + 2, int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if task == "pose" and kp_xy is not None:
            corr_dict = all_corrected_kpts[ri] if ri < len(all_corrected_kpts) and all_corrected_kpts[ri] is not None else {}
            filtered = all_filtered_indices[ri]
            for j, kp_arr in enumerate(kp_xy):
                if j in filtered:
                    continue
                use_kpts = corr_dict.get(j, kp_arr)
                use_kpts = MLPClassifier.sort_keypoints(use_kpts)
                for k_idx, (kx, ky) in enumerate(use_kpts):
                    kc = CATEGORY_COLORS[k_idx % len(CATEGORY_COLORS)]
                    is_corrected = (j in corr_dict)
                    radius = 6 if is_corrected else 4
                    cv2.circle(annotated, (int(kx), int(ky)), radius, kc, -1)
                    cv2.putText(annotated, str(k_idx + 1), (int(kx) + 6, int(ky) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, kc, 1)

        if task == "obb" and obb is not None and len(obb) > 0:
            if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None:
                corners = obb.xyxyxyxy.cpu().numpy()
                for corner_pts in corners:
                    pts = corner_pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

    bar_h = 30
    cv2.rectangle(annotated, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.rectangle(annotated, (0, 0), (w, bar_h), (40, 40, 40), 1)

    info_line1 = f"FPS: {fps:.1f} | YOLO: {ms:.1f}ms"
    if mlp_classifier is not None:
        info_line1 += f" | MLP: {total_mlp_ms:.1f}ms"
    info_line1 += f" | target: {total_count} | class: {len(class_counts)}"
    cv2.putText(annotated, info_line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if mlp_classifier is not None and mlp_class_counts:
        bar_h2 = 24
        cv2.rectangle(annotated, (0, bar_h), (w, bar_h + bar_h2), (0, 0, 0), -1)
        mlp_text = "MLP: " + ", ".join([f"{k}:{v}" for k, v in mlp_class_counts.items()])
        cv2.putText(annotated, mlp_text, (8, bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return annotated, total_count, class_counts, total_mlp_ms, mlp_rois


class YoloLoadWorker(QThread):
    loaded = Signal(object, bool, str)

    def __init__(self, weight_path):
        super().__init__()
        self.weight_path = weight_path

    def run(self):
        try:
            from core.yolo_runtime import load_yolo_model
            model = load_yolo_model(self.weight_path)
            task = model.task
            names = model.names
            self.loaded.emit(model, True, f"模型加载成功，任务类型: {task}，类别数: {len(names)}")
        except Exception as e:
            self.loaded.emit(None, False, f"模型加载失败: {e}")


class YoloInferenceWorker(QThread):
    frame_ready = Signal(object, object)
    finished_signal = Signal(bool, str)
    progress_signal = Signal(int, int, str)

    def __init__(self, model, video_path, conf, iou, imgsz, device,
                 save_video, save_images, output_path, task, mlp_classifier=None,
                 mlp_conf_threshold=0.5, mlp_roi_mode="perspective", hide_negative=False, half=False,
                 rotate_angle=0, flip_horizontal=False):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.save_video = save_video
        self.save_images = save_images
        self.output_path = output_path
        self.task = task
        self.mlp_classifier = mlp_classifier
        self.mlp_conf_threshold = mlp_conf_threshold
        self.mlp_roi_mode = mlp_roi_mode
        self.hide_negative = hide_negative
        self.half = half
        self.rotate_angle = rotate_angle
        self.flip_horizontal = flip_horizontal
        self._stop = False
        self._pause = False
        self._pause_mutex = QMutex()
        self._pause_condition = QWaitCondition()
        self._total_frames = 0

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False
        self._pause_condition.wakeAll()

    def stop(self):
        self._stop = True
        self.resume()

    def _apply_rotate(self, frame):
        if self.rotate_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished_signal.emit(False, "无法打开视频文件")
                return

            fps_val = cap.get(cv2.CAP_PROP_FPS)
            if fps_val <= 0:
                fps_val = 30.0
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ret, frame = cap.read()
            if not ret:
                cap.release()
                self.finished_signal.emit(False, "视频为空")
                return

            if self.rotate_angle or self.flip_horizontal:
                frame = self._apply_rotate(frame)

            max_w = 1280
            orig_h, orig_w = frame.shape[:2]
            if orig_w > max_w:
                scale = max_w / orig_w
                frame = cv2.resize(frame, (max_w, int(orig_h * scale)))

            writer = None
            if self.save_video and self.output_path:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(self.output_path, fourcc, fps_val, (w, h))

            frame_idx = 0
            while not self._stop:
                if self._pause:
                    self._pause_mutex.lock()
                    self._pause_condition.wait(self._pause_mutex)
                    self._pause_mutex.unlock()
                    if self._stop:
                        break

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                if self.rotate_angle or self.flip_horizontal:
                    frame = self._apply_rotate(frame)

                orig_h, orig_w = frame.shape[:2]
                if orig_w > max_w:
                    scale = max_w / orig_w
                    frame = cv2.resize(frame, (max_w, int(orig_h * scale)))

                results = self.model.predict(
                    frame, conf=self.conf, iou=self.iou,
                    imgsz=self.imgsz, device=self.device,
                    verbose=False, task=self.task, half=self.half
                )
                inference_ms = (time.time() - t0) * 1000

                actual_fps = 1000.0 / inference_ms if inference_ms > 0 else 0

                annotated, count, class_counts, mlp_ms, _ = draw_inference_results(
                    frame.copy(), results, self.task,
                    self.model.names, actual_fps, inference_ms,
                    mlp_classifier=self.mlp_classifier,
                    mlp_conf_threshold=self.mlp_conf_threshold,
                    mlp_roi_mode=self.mlp_roi_mode,
                    hide_negative=self.hide_negative
                )

                if writer:
                    writer.write(annotated)

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
                self.frame_ready.emit(qimg.copy(), {
                    'fps': actual_fps, 'ms': inference_ms,
                    'count': count, 'classes': class_counts,
                    'frame': frame_idx, 'total': self._total_frames,
                    'mlp_ms': mlp_ms
                })
                self.progress_signal.emit(frame_idx + 1, self._total_frames, f"帧 {frame_idx + 1}/{self._total_frames}")
                frame_idx += 1

            cap.release()
            if writer:
                writer.release()

            if self._stop:
                self.finished_signal.emit(False, "推理已停止")
            else:
                self.finished_signal.emit(True, f"推理完成，共处理 {frame_idx} 帧")

        except Exception as e:
            import traceback
            self.finished_signal.emit(False, f"推理错误: {e}\n{traceback.format_exc()}")


class VideoPlayerWorker(QThread):
    frame_ready = Signal(object, int, int)
    finished_signal = Signal()
    progress_signal = Signal(int, int)
    video_ended = Signal()

    def __init__(self, video_path, speed=1.0):
        super().__init__()
        self.video_path = video_path
        self.speed = speed
        self._stop = False
        self._pause = False
        self._pause_mutex = QMutex()
        self._pause_condition = QWaitCondition()
        self._seek_to = -1
        self._seek_mutex = QMutex()
        self._total_frames = 0
        self._cap = None
        self._frame_idx = 0
        self._video_ended = False

    def seek(self, frame_num):
        with self._seek_mutex:
            self._seek_to = frame_num
        self._pause_condition.wakeAll()

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False
        self._pause_condition.wakeAll()

    def stop(self):
        self._stop = True
        self._pause = False
        self._pause_condition.wakeAll()

    def _do_seek(self, frame_num):
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self._frame_idx = frame_num

    def run(self):
        try:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                self.finished_signal.emit()
                return

            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            base_delay = 1000.0 / fps

            self._frame_idx = 0
            self._video_ended = False

            while not self._stop:
                if self._pause and not self._was_seek_pending():
                    self._pause_mutex.lock()
                    self._pause_condition.wait(self._pause_mutex)
                    self._pause_mutex.unlock()
                    if self._stop:
                        break
                    if self._seek_to < 0:
                        continue

                with self._seek_mutex:
                    seek_frame = self._seek_to
                    self._seek_to = -1

                if seek_frame >= 0:
                    self._do_seek(seek_frame)

                ret, frame = self._cap.read()
                if not ret:
                    if self._video_ended:
                        self._pause = True
                        continue
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_idx = 0
                    self._video_ended = True
                    self.video_ended.emit()
                    self._pause = True
                    continue

                self._video_ended = False
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
                self.frame_ready.emit(qimg.copy(), self._frame_idx, self._total_frames)
                self.progress_signal.emit(self._frame_idx + 1, self._total_frames)
                self._frame_idx += 1

                delay = max(1, int(base_delay / self.speed))
                self.msleep(delay)

            if self._cap:
                self._cap.release()
            self.finished_signal.emit()

        except Exception:
            pass

    def _was_seek_pending(self):
        with self._seek_mutex:
            return self._seek_to >= 0


class YoloDisplayWindow(QMainWindow):
    inference_stopped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO 推理结果")
        self.setMinimumSize(800, 600)
        self.resize(1280, 800)
        # yolo_inference.py 顶层加载 EDITORIAL_QSS 后所有窗口都继承
        self._is_fullscreen = False

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setProperty("role", "display_mono")
        self.display_label.setText("等待推理...")
        main_layout.addWidget(self.display_label, 1)

        control_widget = QFrame()
        control_widget.setObjectName("footerBar")
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(12, 8, 12, 8)
        control_layout.setSpacing(8)

        progress_hbox = QHBoxLayout()
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.setValue(0)
        self.lbl_progress = QLabel("0/0")
        self.lbl_progress.setProperty("role", "mono")
        self.lbl_progress.style().unpolish(self.lbl_progress)
        self.lbl_progress.style().polish(self.lbl_progress)
        progress_hbox.addWidget(self.progress_slider)
        progress_hbox.addWidget(self.lbl_progress)
        control_layout.addLayout(progress_hbox)

        btn_hbox = QHBoxLayout()

        self.btn_prev = QPushButton("◀◀")
        self.btn_prev.setFixedWidth(50)
        self.btn_play_pause = QPushButton("⏸")
        self.btn_play_pause.setFixedWidth(50)
        self.btn_next = QPushButton("▶▶")
        self.btn_next.setFixedWidth(50)

        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.cmb_speed.setCurrentText("1x")
        self.cmb_speed.setFixedWidth(60)

        lbl_thresh = QLabel("阈值:")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setFixedWidth(70)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.7)
        self.spin_iou.setFixedWidth(70)

        self.btn_close = QPushButton("关闭显示")

        btn_hbox.addWidget(self.btn_prev)
        btn_hbox.addWidget(self.btn_play_pause)
        btn_hbox.addWidget(self.btn_next)
        btn_hbox.addWidget(self.cmb_speed)
        btn_hbox.addStretch()
        btn_hbox.addWidget(lbl_thresh)
        btn_hbox.addWidget(self.spin_conf)
        btn_hbox.addWidget(QLabel(" IOU"))
        btn_hbox.addWidget(self.spin_iou)
        btn_hbox.addWidget(self.btn_close)
        control_layout.addLayout(btn_hbox)

        main_layout.addWidget(control_widget)

        self._player_worker = None
        self._inference_worker = None
        self._is_playing = False
        self._is_player_mode = False
        self._is_dragging_slider = False

        self.progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self._on_slider_released)

    def update_display(self, qimg, stats=None):
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled)
        self.display_label.setText("")

        if stats:
            fps = stats.get('fps', 0)
            ms = stats.get('ms', 0)
            mlp_ms = stats.get('mlp_ms', 0)
            count = stats.get('count', 0)
            frame = stats.get('frame', 0)
            total = stats.get('total', 1)
            if total > 0 and not self._is_dragging_slider:
                self.progress_slider.setValue(int(frame / total * 100))
                if mlp_ms > 0:
                    self.lbl_progress.setText(f"{frame + 1}/{total}  FPS:{fps:.1f}  YOLO:{ms:.1f}ms  MLP:{mlp_ms:.1f}ms  目标:{count}")
                else:
                    self.lbl_progress.setText(f"{frame + 1}/{total}  FPS:{fps:.1f}  {ms:.1f}ms  目标:{count}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.display_label.pixmap():
            pixmap = self.display_label.pixmap()
            if pixmap:
                scaled = pixmap.scaled(
                    self.display_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.display_label.setPixmap(scaled)

    def set_inference_worker(self, worker):
        self._inference_worker = worker

    def set_player_worker(self, worker):
        self._player_worker = worker

    def closeEvent(self, event):
        self._cleanup_workers()
        self.inference_stopped.emit()
        event.accept()

    def _cleanup_workers(self):
        if self._player_worker:
            self._player_worker.stop()
            self._player_worker.wait(2000)
            self._player_worker = None
        if self._inference_worker:
            self._inference_worker.stop()
            self._inference_worker.wait(2000)
            self._inference_worker = None

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F11:
            if self._is_fullscreen:
                self.showNormal()
                self._is_fullscreen = False
            else:
                self.showFullScreen()
                self._is_fullscreen = True
        elif key == Qt.Key_Space:
            self._toggle_play_pause()
        elif key == Qt.Key_Left:
            self._seek_relative(-5)
        elif key == Qt.Key_Right:
            self._seek_relative(5)
        else:
            super().keyPressEvent(event)

    def _toggle_play_pause(self):
        worker = self._inference_worker or self._player_worker
        if not worker or not worker.isRunning():
            return
        if self._is_playing:
            worker.pause()
            self.btn_play_pause.setText("▶")
            self._is_playing = False
        else:
            worker.resume()
            self.btn_play_pause.setText("⏸")
            self._is_playing = True

    def _seek_relative(self, delta):
        worker = self._player_worker
        if not worker:
            return
        total = worker._total_frames
        if total <= 0:
            return
        curr = int(self.progress_slider.value() / 100.0 * total)
        worker.seek(max(0, min(total - 1, curr + delta)))

    def _on_slider_pressed(self):
        self._is_dragging_slider = True

    def _on_slider_released(self):
        self._is_dragging_slider = False
        if self._player_worker:
            total = self._player_worker._total_frames
            if total > 0:
                frame_num = int(self.progress_slider.value() / 100.0 * total)
                self._player_worker.seek(frame_num)

    def _on_video_ended(self):
        self._is_playing = False
        self.btn_play_pause.setText("▶")

    def update_player_frame(self, qimg, frame_idx, total):
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled)
        self.display_label.setText("")
        if not self._is_dragging_slider:
            if total > 0:
                self.progress_slider.setValue(int(frame_idx / total * 100))
            self.lbl_progress.setText(f"{frame_idx + 1}/{total}  播放结束")


class YoloInferenceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 推理查看器")
        self.setMinimumSize(700, 500)
        self.resize(720, 560)
        self.setStyleSheet(EDITORIAL_QSS)

        self.model = None
        self.task = None
        self.names = None
        self._load_worker = None
        self._mlp_load_worker = None
        self._mlp_classifier = None
        self._inference_worker = None
        self._is_video_mode = False
        self._source_images = []
        self._image_index = 0
        self._display_window = None
        self._mlp_roi_dialog = None
        self._onnx_half = False

        self._setup_ui()
        self._connect_signals()
        self._load_config()

    def _setup_ui(self):
        font_bold = QFont("Microsoft YaHei", 10, QFont.Bold)
        font_normal = QFont("Microsoft YaHei", 9)


        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        from ui.yolo_runtime_switch import YoloRuntimeSwitch
        self.runtime_switch = YoloRuntimeSwitch(self)
        self.runtime_switch.runtime_changed.connect(self._on_runtime_changed)
        main_layout.addWidget(self.runtime_switch)

        settings_group = QGroupBox("推理设置")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(10)

        lbl_model = QLabel("模型权重:")
        lbl_model.setFont(font_bold)
        self.edt_model = QLineEdit()
        self.edt_model.setFont(font_normal)
        self.edt_model.setReadOnly(True)
        self.btn_model = QPushButton("选择模型")
        self.btn_model.setFont(font_normal)
        settings_layout.addWidget(lbl_model, 0, 0)
        settings_layout.addWidget(self.edt_model, 0, 1, 1, 3)
        settings_layout.addWidget(self.btn_model, 0, 4)

        lbl_source_type = QLabel("推理源:")
        lbl_source_type.setFont(font_bold)
        self.radio_image = QRadioButton("图片")
        self.radio_video = QRadioButton("视频")
        self.radio_image.setFont(font_normal)
        self.radio_video.setFont(font_normal)
        self.radio_image.setChecked(True)
        self.source_group = QButtonGroup()
        self.source_group.addButton(self.radio_image, 0)
        self.source_group.addButton(self.radio_video, 1)
        src_layout = QHBoxLayout()
        src_layout.addWidget(self.radio_image)
        src_layout.addWidget(self.radio_video)
        src_layout.addStretch()
        settings_layout.addWidget(lbl_source_type, 1, 0)
        settings_layout.addLayout(src_layout, 1, 1, 1, 3)

        self.edt_source = QLineEdit()
        self.edt_source.setFont(font_normal)
        self.edt_source.setReadOnly(True)
        self.btn_source = QPushButton("选择")
        self.btn_source.setFont(font_normal)
        settings_layout.addWidget(self.edt_source, 2, 1, 1, 3)
        settings_layout.addWidget(self.btn_source, 2, 4)

        lbl_conf = QLabel("置信度:")
        lbl_conf.setFont(font_bold)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setFont(font_normal)
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)

        lbl_iou = QLabel("IOU:")
        lbl_iou.setFont(font_bold)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setFont(font_normal)
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.7)

        lbl_imgsz = QLabel("图片大小:")
        lbl_imgsz.setFont(font_bold)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setFont(font_normal)
        self.spin_imgsz.setRange(320, 1280)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)

        lbl_device = QLabel("设备:")
        lbl_device.setFont(font_bold)
        self.cmb_device = QComboBox()
        self.cmb_device.setFont(font_normal)
        self.cmb_device.addItems(["0", "1", "2", "3", "cuda", "cpu"])
        self.cmb_device.setCurrentText("0")

        settings_layout.addWidget(lbl_conf, 3, 0)
        settings_layout.addWidget(self.spin_conf, 3, 1)
        settings_layout.addWidget(lbl_iou, 3, 2)
        settings_layout.addWidget(self.spin_iou, 3, 3)
        settings_layout.addWidget(lbl_imgsz, 4, 0)
        settings_layout.addWidget(self.spin_imgsz, 4, 1)
        settings_layout.addWidget(lbl_device, 4, 2)
        settings_layout.addWidget(self.cmb_device, 4, 3)

        lbl_save = QLabel("保存:")
        lbl_save.setFont(font_bold)
        self.chk_save_image = QCheckBox("保存标注图片")
        self.chk_save_video = QCheckBox("保存标注视频")
        self.chk_save_image.setFont(font_normal)
        self.chk_save_video.setFont(font_normal)
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.chk_save_image)
        save_layout.addWidget(self.chk_save_video)
        save_layout.addStretch()
        settings_layout.addWidget(lbl_save, 5, 0)
        settings_layout.addLayout(save_layout, 5, 1, 1, 3)

        lbl_out = QLabel("输出目录:")
        lbl_out.setFont(font_bold)
        self.edt_out = QLineEdit()
        self.edt_out.setFont(font_normal)
        self.edt_out.setReadOnly(True)
        self.btn_out = QPushButton("选择")
        self.btn_out.setFont(font_normal)
        settings_layout.addWidget(lbl_out, 6, 0)
        settings_layout.addWidget(self.edt_out, 6, 1, 1, 3)
        settings_layout.addWidget(self.btn_out, 6, 4)

        lbl_rotate = QLabel("旋转:")
        lbl_rotate.setFont(font_bold)
        self.cmb_rotate = QComboBox()
        self.cmb_rotate.setFont(font_normal)
        self.cmb_rotate.addItems(["0°", "90°", "180°", "270°"])
        self.cmb_rotate.setCurrentIndex(0)
        self.cmb_rotate.setFixedWidth(80)
        self.chk_flip = QCheckBox("水平翻转")
        self.chk_flip.setFont(font_normal)
        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(self.cmb_rotate)
        rotate_layout.addWidget(self.chk_flip)
        rotate_layout.addStretch()
        settings_layout.addWidget(lbl_rotate, 7, 0)
        settings_layout.addLayout(rotate_layout, 7, 1, 1, 3)

        main_layout.addWidget(settings_group)

        self.mlp_group = QGroupBox("MLP 分类器 (仅 Pose 模式)")
        self.mlp_group.setFont(font_bold)
        mlp_layout = QGridLayout(self.mlp_group)
        mlp_layout.setSpacing(10)

        lbl_mlp = QLabel("MLP模型:")
        lbl_mlp.setFont(font_bold)
        self.edt_mlp = QLineEdit()
        self.edt_mlp.setFont(font_normal)
        self.edt_mlp.setReadOnly(True)
        self.btn_mlp = QPushButton("选择MLP模型")
        self.btn_mlp.setFont(font_normal)
        self.lbl_mlp_status = QLabel("未加载")
        self.lbl_mlp_status.setFont(font_normal)
        self.lbl_mlp_status.setProperty("role", "muted")
        self.lbl_mlp_status.style().unpolish(self.lbl_mlp_status)
        self.lbl_mlp_status.style().polish(self.lbl_mlp_status)

        mlp_layout.addWidget(lbl_mlp, 0, 0)
        mlp_layout.addWidget(self.edt_mlp, 0, 1, 1, 3)
        mlp_layout.addWidget(self.btn_mlp, 0, 4)
        mlp_layout.addWidget(self.lbl_mlp_status, 1, 0, 1, 5)

        lbl_mlp_conf = QLabel("MLP阈值:")
        lbl_mlp_conf.setFont(font_bold)
        self.spin_mlp_conf = QDoubleSpinBox()
        self.spin_mlp_conf.setFont(font_normal)
        self.spin_mlp_conf.setRange(0.0, 1.0)
        self.spin_mlp_conf.setDecimals(2)
        self.spin_mlp_conf.setSingleStep(0.05)
        self.spin_mlp_conf.setValue(0.5)
        mlp_layout.addWidget(lbl_mlp_conf, 2, 0)
        mlp_layout.addWidget(self.spin_mlp_conf, 2, 1)

        lbl_mlp_roi = QLabel("ROI模式:")
        lbl_mlp_roi.setFont(font_bold)
        self.radio_perspective = QRadioButton("透视变换")
        self.radio_perspective.setFont(font_normal)
        self.radio_perspective.setChecked(True)
        self.radio_bbox = QRadioButton("矩形裁剪")
        self.radio_bbox.setFont(font_normal)
        self._mlp_roi_group = QButtonGroup()
        self._mlp_roi_group.addButton(self.radio_perspective, 0)
        self._mlp_roi_group.addButton(self.radio_bbox, 1)
        roi_mode_layout = QHBoxLayout()
        roi_mode_layout.addWidget(self.radio_perspective)
        roi_mode_layout.addWidget(self.radio_bbox)
        roi_mode_layout.addStretch()
        mlp_layout.addWidget(lbl_mlp_roi, 3, 0)
        mlp_layout.addLayout(roi_mode_layout, 3, 1, 1, 4)

        lbl_mlp_opts = QLabel("显示/保存:")
        lbl_mlp_opts.setFont(font_bold)
        self.chk_show_mlp_roi = QCheckBox("显示MLP ROI")
        self.chk_show_mlp_roi.setFont(font_normal)
        self.chk_save_mlp_roi = QCheckBox("保存MLP ROI")
        self.chk_save_mlp_roi.setFont(font_normal)
        self.chk_hide_negative = QCheckBox("仅显示有效目标(隐藏negative)")
        self.chk_hide_negative.setFont(font_normal)
        self.chk_hide_negative.setChecked(True)
        opts_layout = QHBoxLayout()
        opts_layout.addWidget(self.chk_show_mlp_roi)
        opts_layout.addWidget(self.chk_save_mlp_roi)
        opts_layout.addWidget(self.chk_hide_negative)
        opts_layout.addStretch()
        mlp_layout.addWidget(lbl_mlp_opts, 4, 0)
        mlp_layout.addLayout(opts_layout, 4, 1, 1, 4)

        main_layout.addWidget(self.mlp_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_start = QPushButton("开始推理")
        self.btn_start.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        self.btn_stop = QPushButton("停止推理")
        self.btn_stop.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.btn_stop.setEnabled(False)

        self.btn_open_saved = QPushButton("打开已保存视频")
        self.btn_open_saved.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        btn_layout.addWidget(self.btn_open_saved)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        main_layout.addLayout(btn_layout)

    def _connect_signals(self):
        self.btn_model.clicked.connect(self._on_select_model)
        self.btn_mlp.clicked.connect(self._on_select_mlp)
        self.btn_source.clicked.connect(self._on_select_source)
        self.btn_out.clicked.connect(self._on_select_output)
        self.radio_image.toggled.connect(self._on_source_type_changed)
        self.radio_video.toggled.connect(self._on_source_type_changed)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_open_saved.clicked.connect(self._on_open_saved)

    def _on_runtime_changed(self, runtime, previous):
        from core.yolo_runtime import runtime_label, set_runtime
        if self._inference_worker and self._inference_worker.isRunning():
            QMessageBox.warning(self, "提示", "推理进行中，无法切换运行时")
            set_runtime(previous)
            self.runtime_switch.sync_from_settings()
            return
        self.model = None
        self.task = None
        self.names = None
        QMessageBox.information(
            self, "运行时已切换",
            f"已切换至 {runtime_label(runtime)}。\n请重新选择并加载模型权重。",
        )

    def _load_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig().get_tool_config("yolo_inference")

        try:
            self.radio_image.toggled.disconnect(self._on_source_type_changed)
            self.radio_video.toggled.disconnect(self._on_source_type_changed)
        except Exception:
            pass

        self.edt_model.setText(cfg.get("model_path", ""))
        model_path = cfg.get("model_path", "")
        if model_path and os.path.exists(model_path):
            if self._load_worker and self._load_worker.isRunning():
                self._load_worker.quit()
                self._load_worker.wait(2000)
            self._load_worker = YoloLoadWorker(model_path)
            self._load_worker.loaded.connect(self._on_model_loaded)
            self._load_worker.start()

        source_type = cfg.get("source_type", "image")
        if source_type == "video":
            self.radio_video.setChecked(True)
            self.edt_source.setText(cfg.get("source_path", ""))
            self._source_images = []
        else:
            self.radio_image.setChecked(True)
            self._source_images = cfg.get("source_images", [])
            valid_images = [p for p in self._source_images if os.path.exists(p)]
            if valid_images:
                self._source_images = valid_images
                self.edt_source.setText(f"{len(self._source_images)} 张图片已选择")
            elif self._source_images:
                self.edt_source.setText(f"{len(self._source_images)} 张图片已选择（部分文件已失效）")
            else:
                self.edt_source.setText(cfg.get("source_path", ""))

        self.spin_conf.setValue(float(cfg.get("conf", 0.25)))
        self.spin_iou.setValue(float(cfg.get("iou", 0.7)))
        self.spin_imgsz.setValue(int(cfg.get("imgsz", 640)))
        self.cmb_device.setCurrentText(cfg.get("device", "0"))
        self.chk_save_image.setChecked(bool(cfg.get("save_image", False)))
        self.chk_save_video.setChecked(bool(cfg.get("save_video", False)))
        self.edt_out.setText(cfg.get("output_dir", ""))
        self.edt_mlp.setText(cfg.get("mlp_path", ""))
        mlp_path = cfg.get("mlp_path", "")
        if mlp_path and os.path.exists(mlp_path):
            if self._mlp_load_worker and self._mlp_load_worker.isRunning():
                self._mlp_load_worker.quit()
                self._mlp_load_worker.wait(2000)
            self._mlp_load_worker = MLPLoadWorker(mlp_path)
            self._mlp_load_worker.loaded.connect(self._on_mlp_loaded)
            self._mlp_load_worker.start()
        self.spin_mlp_conf.setValue(float(cfg.get("mlp_conf", 0.5)))
        if cfg.get("mlp_roi_mode") == "bbox":
            self.radio_bbox.setChecked(True)
        else:
            self.radio_perspective.setChecked(True)
        self.chk_show_mlp_roi.setChecked(bool(cfg.get("show_mlp_roi", False)))
        self.chk_save_mlp_roi.setChecked(bool(cfg.get("save_mlp_roi", False)))
        self.chk_hide_negative.setChecked(bool(cfg.get("hide_negative", True)))
        self.cmb_rotate.setCurrentIndex(int(cfg.get("rotate_angle", 0)))
        self.chk_flip.setChecked(bool(cfg.get("flip_horizontal", False)))

        self.radio_image.toggled.connect(self._on_source_type_changed)
        self.radio_video.toggled.connect(self._on_source_type_changed)

    def _save_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig()
        cfg.set("yolo_inference", "model_path", self.edt_model.text())
        cfg.set("yolo_inference", "source_type", "video" if self.radio_video.isChecked() else "image")
        if self.radio_video.isChecked():
            cfg.set("yolo_inference", "source_path", self.edt_source.text())
            cfg.set("yolo_inference", "source_images", [])
        else:
            cfg.set("yolo_inference", "source_path", self.edt_source.text())
            cfg.set("yolo_inference", "source_images", self._source_images)
        cfg.set("yolo_inference", "conf", self.spin_conf.value())
        cfg.set("yolo_inference", "iou", self.spin_iou.value())
        cfg.set("yolo_inference", "imgsz", self.spin_imgsz.value())
        cfg.set("yolo_inference", "device", self.cmb_device.currentText())
        cfg.set("yolo_inference", "save_image", self.chk_save_image.isChecked())
        cfg.set("yolo_inference", "save_video", self.chk_save_video.isChecked())
        cfg.set("yolo_inference", "output_dir", self.edt_out.text())
        cfg.set("yolo_inference", "mlp_path", self.edt_mlp.text())
        cfg.set("yolo_inference", "mlp_conf", self.spin_mlp_conf.value())
        cfg.set("yolo_inference", "mlp_roi_mode", "bbox" if self.radio_bbox.isChecked() else "perspective")
        cfg.set("yolo_inference", "show_mlp_roi", self.chk_show_mlp_roi.isChecked())
        cfg.set("yolo_inference", "save_mlp_roi", self.chk_save_mlp_roi.isChecked())
        cfg.set("yolo_inference", "hide_negative", self.chk_hide_negative.isChecked())
        cfg.set("yolo_inference", "rotate_angle", self.cmb_rotate.currentIndex())
        cfg.set("yolo_inference", "flip_horizontal", self.chk_flip.isChecked())
        cfg.save()

    def _get_default_dir(self, category):
        base = _ROOT
        if category == "origin_data":
            d = os.path.join(base, "origin_data")
            return d if os.path.exists(d) else base
        elif category == "yolo":
            d = os.path.join(base, "runs")
            return d if os.path.exists(d) else base
        elif category == "mlp":
            d = os.path.join(base, "stored_weights", "model", "armor_tradition")
            return d if os.path.exists(d) else base
        return base

    def _on_select_model(self):
        default_dir = self._get_default_dir("yolo")
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型", default_dir, "模型文件 (*.pt *.onnx *.yaml);;所有文件 (*)")
        if not path:
            return
        self.edt_model.setText(path)
        if self._load_worker and self._load_worker.isRunning():
            return
        self._load_worker = YoloLoadWorker(path)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.start()

    def _on_model_loaded(self, model, success, msg):
        if success:
            self.model = model
            self.task = model.task
            self.names = model.names
            self._onnx_half = False
            if hasattr(model, 'ckpt_path') and model.ckpt_path and model.ckpt_path.endswith('.onnx'):
                self._onnx_half = self._check_onnx_fp16(model.ckpt_path)
            QMessageBox.information(self, "模型加载", msg)
        else:
            QMessageBox.warning(self, "模型加载失败", msg)

    def _check_onnx_fp16(self, onnx_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_type = session.get_inputs()[0].type
            is_fp16 = input_type == 'tensor(float16)'
            del session
            return is_fp16
        except Exception:
            return False

    def _on_select_mlp(self):
        default_dir = self._get_default_dir("mlp")
        path, _ = QFileDialog.getOpenFileName(self, "选择 MLP 模型", default_dir, "模型文件 (*.onnx *.pt *.pth);;所有文件 (*)")
        if not path:
            return
        self.edt_mlp.setText(path)
        if self._mlp_load_worker and self._mlp_load_worker.isRunning():
            return
        self._mlp_load_worker = MLPLoadWorker(path)
        self._mlp_load_worker.loaded.connect(self._on_mlp_loaded)
        self._mlp_load_worker.start()

    def _mlp_roi_mode(self):
        if self._mlp_roi_group.checkedId() == 1:
            return "bbox"
        return "perspective"

    def _show_mlp_roi_dialog(self, basename, mlp_rois):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QWidget, QPushButton, QGridLayout
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap

        if self._mlp_roi_dialog is not None and self._mlp_roi_dialog.isVisible():
            self._mlp_roi_dialog.close()
            self._mlp_roi_dialog.deleteLater()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"MLP ROI 调试 - {basename}")
        dialog.resize(1200, 500)
        dialog.setStyleSheet(EDITORIAL_QSS)

        layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(450)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(8)

        thumb_w, thumb_h = 140, 100
        stage_thumb_w, stage_thumb_h = 100, 72
        stage_labels = ["原图+修正kpts", "透视后", "裁剪后", "二值化", "最终ROI"]
        stage_keys = ["before_persp", "after_persp", "after_crop", "after_binary", "final"]

        for idx, roi_data in enumerate(mlp_rois):
            if len(roi_data) >= 4:
                roi_bgr, mlp_label, mlp_conf_val, debug_stages = roi_data
            else:
                roi_bgr, mlp_label, mlp_conf_val = roi_data
                debug_stages = None

            lbl_title = QLabel(f"检测 #{idx+1}: {mlp_label} ({mlp_conf_val:.2f})")
            lbl_title.setProperty("role", "status_bold_success" if mlp_label != "?" else "status_warn")
            lbl_title.setAlignment(Qt.AlignLeft)
            grid.addWidget(lbl_title, idx * 6, 0, 1, 5)

            if debug_stages is not None:
                for si, (stage_name, stage_key) in enumerate(zip(stage_labels, stage_keys)):
                    stage_img = debug_stages.get(stage_key)
                    if stage_img is not None:
                        stage_rgb = cv2.cvtColor(stage_img, cv2.COLOR_BGR2RGB)
                        h_s, w_s = stage_rgb.shape[:2]
                        qimg_s = QImage(stage_rgb.data, w_s, h_s, stage_rgb.strides[0], QImage.Format_RGB888)
                        pixmap_s = QPixmap.fromImage(qimg_s).scaled(stage_thumb_w, stage_thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                        lbl_stage = QLabel()
                        lbl_stage.setPixmap(pixmap_s)
                        lbl_stage.setAlignment(Qt.AlignCenter)
                        lbl_stage.setProperty("role", "display")

                        lbl_stage_name = QLabel(stage_name)
                        lbl_stage_name.setAlignment(Qt.AlignCenter)
                        lbl_stage_name.setProperty("role", "hint")

                        grid.addWidget(lbl_stage, idx * 6 + 1, si)
                        grid.addWidget(lbl_stage_name, idx * 6 + 2, si)
                    else:
                        lbl_stage = QLabel("N/A")
                        lbl_stage.setAlignment(Qt.AlignCenter)
                        lbl_stage.setProperty("role", "display_sm_muted")
                        lbl_stage_name = QLabel(stage_name)
                        lbl_stage_name.setAlignment(Qt.AlignCenter)
                        lbl_stage_name.setProperty("role", "hint")
                        grid.addWidget(lbl_stage, idx * 6 + 1, si)
                        grid.addWidget(lbl_stage_name, idx * 6 + 2, si)
            else:
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                h_roi, w_roi = roi_rgb.shape[:2]
                qimg = QImage(roi_rgb.data, w_roi, h_roi, roi_rgb.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                lbl_img = QLabel()
                lbl_img.setPixmap(pixmap)
                lbl_img.setAlignment(Qt.AlignCenter)
                lbl_img.setProperty("role", "display_sm")

                lbl_text = QLabel(f"{mlp_label}\n{mlp_conf_val:.2f}")
                lbl_text.setAlignment(Qt.AlignCenter)
                lbl_text.setProperty("role", "status_bold_success" if mlp_label != "?" else "status_warn")

                grid.addWidget(lbl_img, idx * 6 + 1, 0, 2, 3)
                grid.addWidget(lbl_text, idx * 6 + 3, 0, 1, 3)

            sep = QLabel()
            sep.setFixedHeight(1)
            sep.setProperty("role", "hairline")
            grid.addWidget(sep, idx * 6 + 4, 0, 1, 5)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dialog.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self._mlp_roi_dialog = dialog
        dialog.show()

    def _save_mlp_rois(self, output_dir, basename, mlp_rois):
        roi_dir = os.path.join(output_dir, "mlp_roi")
        os.makedirs(roi_dir, exist_ok=True)
        for idx, (roi_bgr, mlp_label, mlp_conf, _debug_stages) in enumerate(mlp_rois):
            safe_label = str(mlp_label).replace("/", "_").replace("\\", "_")
            out_name = f"{basename}_roi_{idx}_{safe_label}.jpg"
            out_path = os.path.join(roi_dir, out_name)
            cv2.imwrite(out_path, roi_bgr)

    def _on_mlp_loaded(self, classifier, success, msg):
        if success:
            self._mlp_classifier = classifier
            self.lbl_mlp_status.setText(f"已加载: {classifier.model_type} | 输入: {classifier.input_size} | 类别: {len(classifier.labels)}")
            self.lbl_mlp_status.setProperty("role", "status_success")
            QMessageBox.information(self, "MLP模型加载", msg)
        else:
            self._mlp_classifier = None
            self.lbl_mlp_status.setText("加载失败")
            self.lbl_mlp_status.setProperty("role", "status_warn")
            QMessageBox.warning(self, "MLP模型加载失败", msg)

    def _on_select_source(self):
        default_dir = self._get_default_dir("origin_data")
        if self.radio_image.isChecked():
            paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", default_dir, "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)")
            if paths:
                self._source_images = sorted(paths)
                self.edt_source.setText(f"{len(self._source_images)} 张图片已选择")
                self._is_video_mode = False
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", default_dir, "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v *.ts *.3gp);;所有文件 (*)")
            if path:
                self.edt_source.setText(path)
                self._is_video_mode = True

    def _on_select_output(self):
        default_dir = self._get_default_dir("origin_data")
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", default_dir)
        if path:
            self.edt_out.setText(path)

    def _on_source_type_changed(self):
        self.edt_source.clear()
        self._source_images = []
        self._is_video_mode = self.radio_video.isChecked()

    def _on_start(self):
        if not self.model:
            QMessageBox.warning(self, "提示", "请先选择并加载模型")
            return
        if not self.edt_source.text():
            QMessageBox.warning(self, "提示", "请先选择推理源")
            return

        conf = self.spin_conf.value()
        iou = self.spin_iou.value()
        imgsz = self.spin_imgsz.value()
        device = self.cmb_device.currentText()
        save_image = self.chk_save_image.isChecked()
        save_video = self.chk_save_video.isChecked()
        output_dir = self.edt_out.text()
        rotate_angle = self.cmb_rotate.currentIndex() * 90
        flip_horizontal = self.chk_flip.isChecked()

        if save_image or save_video:
            if not output_dir:
                QMessageBox.warning(self, "提示", "请选择输出目录")
                return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._is_video_mode = self.radio_video.isChecked()

        if self._is_video_mode:
            self._start_video_inference(conf, iou, imgsz, device, save_video, save_image, output_dir,
                                         rotate_angle=rotate_angle, flip_horizontal=flip_horizontal)
        else:
            self._start_image_inference(conf, iou, imgsz, device, save_image, output_dir)

    def _start_video_inference(self, conf, iou, imgsz, device, save_video, save_image, output_dir,
                                rotate_angle=0, flip_horizontal=False):
        video_path = self.edt_source.text()
        output_path = None
        if save_video:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{basename}_annotated.mp4")

        self._display_window = YoloDisplayWindow()
        self._display_window.inference_stopped.connect(self._on_display_closed)

        self._inference_worker = YoloInferenceWorker(
            self.model, video_path, conf, iou, imgsz, device,
            save_video, save_image, output_path, self.task,
            mlp_classifier=self._mlp_classifier,
            mlp_conf_threshold=self.spin_mlp_conf.value(),
            mlp_roi_mode=self._mlp_roi_mode(),
            hide_negative=self.chk_hide_negative.isChecked(),
            half=self._onnx_half,
            rotate_angle=rotate_angle,
            flip_horizontal=flip_horizontal
        )
        self._display_window.set_inference_worker(self._inference_worker)

        self._inference_worker.frame_ready.connect(self._display_window.update_display)
        self._inference_worker.finished_signal.connect(self._on_finished)
        self._inference_worker.progress_signal.connect(self._on_progress)

        self._display_window.spin_conf.setValue(conf)
        self._display_window.spin_iou.setValue(iou)
        self._display_window.spin_conf.valueChanged.connect(self._on_threshold_changed)
        self._display_window.spin_iou.valueChanged.connect(self._on_threshold_changed_iou)
        self._display_window.cmb_speed.currentTextChanged.connect(self._on_speed_changed)
        self._display_window.progress_slider.sliderMoved.connect(self._on_seek)
        self._display_window.btn_play_pause.clicked.connect(self._on_play_pause)
        self._display_window.btn_prev.clicked.connect(self._on_prev_frame)
        self._display_window.btn_next.clicked.connect(self._on_next_frame)
        self._display_window.btn_close.clicked.connect(self._on_display_close)

        self._display_window.show()
        self._inference_worker.start()

    def _start_image_inference(self, conf, iou, imgsz, device, save_image, output_dir):
        self._display_window = YoloDisplayWindow()
        self._display_window.inference_stopped.connect(self._on_display_closed)
        self._display_window.show()
        self._image_index = 0
        self._infer_next_image(conf, iou, imgsz, device, save_image, output_dir)

    def _infer_next_image(self, conf, iou, imgsz, device, save_image, output_dir):
        if self._image_index >= len(self._source_images):
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        img_path = self._source_images[self._image_index]
        t0 = time.time()
        results = self.model.predict(img_path, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, task=self.task, half=self._onnx_half)
        inference_ms = (time.time() - t0) * 1000
        actual_fps = 1000.0 / inference_ms if inference_ms > 0 else 0

        img = cv2.imread(img_path)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        annotated, count, class_counts, mlp_ms, mlp_rois = draw_inference_results(
            img, results, self.task, self.names, actual_fps, inference_ms,
            mlp_classifier=self._mlp_classifier,
            mlp_conf_threshold=self.spin_mlp_conf.value(),
            mlp_roi_mode=self._mlp_roi_mode(),
            collect_mlp_rois=True,
            hide_negative=self.chk_hide_negative.isChecked()
        )

        if save_image:
            out_path = os.path.join(output_dir, f"{basename}_annotated.jpg")
            cv2.imwrite(out_path, annotated)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        stats = {'fps': actual_fps, 'ms': inference_ms, 'count': count, 'frame': self._image_index,
                 'total': len(self._source_images), 'mlp_ms': mlp_ms}
        if self._display_window:
            self._display_window.update_display(qimg, stats)
            self._display_window.lbl_progress.setText(f"{self._image_index + 1}/{len(self._source_images)}")

        if mlp_rois:
            if self.chk_show_mlp_roi.isChecked():
                self._show_mlp_roi_dialog(basename, mlp_rois)
            if self.chk_save_mlp_roi.isChecked() and output_dir:
                self._save_mlp_rois(output_dir, basename, mlp_rois)

        self._image_index += 1
        if self._image_index < len(self._source_images):
            QTimer.singleShot(1, lambda: self._infer_next_image(conf, iou, imgsz, device, save_image, output_dir))
        else:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def _on_frame_ready(self, qimg, stats):
        if self._display_window:
            self._display_window.update_display(qimg, stats)

    def _on_progress(self, current, total, text):
        pass

    def _on_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            QMessageBox.information(self, "推理完成", msg)
        else:
            QMessageBox.warning(self, "推理结束", msg)

    def _on_stop(self):
        if self._inference_worker and self._inference_worker.isRunning():
            self._inference_worker.stop()
            self._inference_worker.wait(3000)
        self._inference_worker = None
        if self._display_window:
            self._display_window._cleanup_workers()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_display_close(self):
        self._on_stop()
        if self._display_window:
            self._display_window.close()
            self._display_window = None

    def _on_display_closed(self):
        self._on_stop()
        self._display_window = None

    def _on_open_saved(self):
        default_dir = self._get_default_dir("origin_data")
        path, _ = QFileDialog.getOpenFileName(self, "选择已保存的标注视频", default_dir, "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v *.ts *.3gp);;所有文件 (*)")
        if not path:
            return

        if self._display_window is None:
            self._display_window = YoloDisplayWindow()
            self._display_window.inference_stopped.connect(self._on_display_closed)

        self._display_window.spin_conf.setVisible(False)
        self._display_window.spin_iou.setVisible(False)

        player = VideoPlayerWorker(path, 1.0)
        self._display_window.set_player_worker(player)
        self._display_window._is_player_mode = True
        self._display_window._is_playing = True
        self._display_window.btn_play_pause.setText("⏸")

        player.frame_ready.connect(self._display_window.update_player_frame)
        player.finished_signal.connect(self._on_player_finished)
        player.progress_signal.connect(self._on_player_progress)
        player.video_ended.connect(self._display_window._on_video_ended)
        player.start()

        self._display_window.btn_play_pause.clicked.connect(self._on_play_pause)
        self._display_window.btn_prev.clicked.connect(self._on_prev_frame)
        self._display_window.btn_next.clicked.connect(self._on_next_frame)
        self._display_window.cmb_speed.currentTextChanged.connect(self._on_speed_changed)
        self._display_window.btn_close.clicked.connect(self._on_display_close)

        self._display_window.show()

    def _on_player_finished(self):
        if self._display_window:
            self._display_window.btn_play_pause.setText("▶")

    def _on_player_progress(self, current, total):
        if self._display_window:
            self._display_window.lbl_progress.setText(f"{current}/{total}")
            if total > 0:
                self._display_window.progress_slider.setValue(int(current / total * 100))

    def _on_play_pause(self):
        if not self._display_window:
            return
        worker = self._display_window._player_worker
        if not worker:
            return

        if self._display_window._is_playing:
            worker.pause()
            self._display_window.btn_play_pause.setText("▶")
            self._display_window._is_playing = False
        else:
            worker.resume()
            self._display_window.btn_play_pause.setText("⏸")
            self._display_window._is_playing = True

    def _on_prev_frame(self):
        if self._display_window:
            worker = self._display_window._player_worker
            if worker:
                total = worker._total_frames
                if total <= 0:
                    return
                curr = int(self._display_window.progress_slider.value() / 100.0 * total)
                worker.seek(max(0, curr - 5))

    def _on_next_frame(self):
        if self._display_window:
            worker = self._display_window._player_worker
            if worker:
                total = worker._total_frames
                if total <= 0:
                    return
                curr = int(self._display_window.progress_slider.value() / 100.0 * total)
                worker.seek(min(total - 1, curr + 5))

    def _on_seek(self, value):
        if self._display_window:
            worker = self._display_window._player_worker
            if worker:
                total = worker._total_frames
                worker.seek(int(value / 100.0 * total))

    def _on_speed_changed(self, text):
        if self._display_window:
            worker = self._display_window._player_worker
            if worker:
                speed = float(text.replace("x", ""))
                worker.speed = speed

    def _on_threshold_changed(self, val):
        if self._inference_worker and self._inference_worker.isRunning():
            self._inference_worker.conf = val

    def _on_threshold_changed_iou(self, val):
        if self._inference_worker and self._inference_worker.isRunning():
            self._inference_worker.iou = val

    def closeEvent(self, event):
        self._save_config()
        self._on_stop()
        if self._mlp_load_worker and self._mlp_load_worker.isRunning():
            self._mlp_load_worker.stop()
            self._mlp_load_worker.wait(2000)
        if self._mlp_roi_dialog is not None:
            if self._mlp_roi_dialog.isVisible():
                self._mlp_roi_dialog.close()
            self._mlp_roi_dialog.deleteLater()
            self._mlp_roi_dialog = None
        if self._display_window:
            self._display_window.close()
            self._display_window = None
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = YoloInferenceWindow()
    window.show()
    sys.exit(app.exec())
