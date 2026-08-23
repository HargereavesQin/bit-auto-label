# -*- coding: utf-8 -*-
"""
装甲板数据提取工具 - 从视频或图片文件夹中提取装甲板目标数据
支持YOLO检测 + MLP分类，按类别保存裁剪后的装甲板图片
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import glob
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QGridLayout,
    QProgressBar, QGroupBox, QCheckBox,
    QRadioButton, QButtonGroup, QTextEdit, QStatusBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


# =========================================================
# 加载编辑风样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()


def letterbox_resize(image, target_w, target_h, fill_color=(114, 114, 114)):
    """
    YOLO风格的letterbox resize
    等比缩放 + 居中灰色填充
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)

    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2

    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    return canvas


class ArmorExtractWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int, int, int)
    finish_signal = Signal(bool, str)
    saved_count_signal = Signal(int)

    def __init__(self, yolo_model_path, mlp_model_path,
                 target_category, yolo_conf, mlp_threshold,
                 crop_mode, output_w, output_h, output_dir,
                 source_type="video", video_path="", image_dir="",
                 frame_interval=1, source_prefix="",
                 exposure_enabled=False, exposure_factor=1.0,
                 save_mode="perspective"):
        super().__init__()
        self.yolo_model_path = yolo_model_path
        self.mlp_model_path = mlp_model_path
        self.target_category = target_category
        self.yolo_conf = yolo_conf
        self.mlp_threshold = mlp_threshold
        self.crop_mode = crop_mode
        self.output_w = output_w
        self.output_h = output_h
        self.output_dir = output_dir
        self.source_type = source_type
        self.video_path = video_path
        self.image_dir = image_dir
        self.frame_interval = frame_interval
        self.source_prefix = source_prefix
        self.exposure_enabled = exposure_enabled
        self.exposure_factor = exposure_factor
        self.save_mode = save_mode
        self._stop = False

    def stop(self):
        self._stop = True

    def log(self, msg):
        self.log_signal.emit(msg)

    def _get_image_files(self):
        """Get sorted list of image files from directory."""
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP')
        all_images = []
        for ext in exts:
            all_images.extend(glob.glob(os.path.join(self.image_dir, ext)))
        return sorted(all_images)

    def run(self):
        try:
            from core.mlp_classifier import MLPClassifier

            self.log(f"加载YOLO模型: {self.yolo_model_path}")
            from ultralytics import YOLO
            yolo_model = YOLO(self.yolo_model_path)
            self.log("YOLO模型加载成功")

            self.log(f"加载MLP模型: {self.mlp_model_path}")
            mlp_classifier = MLPClassifier(self.mlp_model_path)
            self.log(f"MLP模型加载成功: {len(mlp_classifier.labels)} 类")

            os.makedirs(self.output_dir, exist_ok=True)

            saved_count = 0
            instance_counts = {}

            if self.source_type == "image":
                image_files = self._get_image_files()
                total_images = len(image_files)
                if total_images == 0:
                    self.finish_signal.emit(False, f"图片目录为空: {self.image_dir}")
                    return
                self.log(f"图片目录: {self.image_dir}, 共 {total_images} 张图片")

                for proc_idx, img_path in enumerate(image_files):
                    if self._stop:
                        self.log("用户取消")
                        self.finish_signal.emit(False, "用户取消")
                        return

                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue

                    frame_idx = proc_idx

                    results = yolo_model(frame, conf=self.yolo_conf, verbose=False)
                    result = results[0] if results else None

                    detected, matched = self._process_frame(
                        frame, result, frame_idx, mlp_classifier,
                        instance_counts, saved_count
                    )
                    if matched > 0:
                        saved_count = instance_counts.get('_total', 0)

                    self.progress_signal.emit(proc_idx + 1, total_images, frame_idx, detected)

                    if proc_idx % 10 == 0 or matched > 0:
                        self.log(f"图片 {frame_idx+1}/{total_images}: 检测 {detected}, 匹配 {matched}, 已保存 {saved_count} 张")

                self.log(f"提取完成! 共保存 {saved_count} 张图片 → {self.output_dir}")
                self.finish_signal.emit(True, f"成功提取 {saved_count} 张图片")

            else:
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.finish_signal.emit(False, f"无法打开视频: {self.video_path}")
                    return

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.log(f"视频信息: {total_frames} 帧, {fps:.1f} FPS")

                target_frame_indices = list(range(0, total_frames, max(1, self.frame_interval)))
                self.log(f"预计处理: {len(target_frame_indices)} 帧")

                for proc_idx, fidx in enumerate(target_frame_indices):
                    if self._stop:
                        self.log("用户取消")
                        self.finish_signal.emit(False, "用户取消")
                        return

                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame_idx = fidx

                    results = yolo_model(frame, conf=self.yolo_conf, verbose=False)
                    result = results[0] if results else None

                    detected, matched = self._process_frame(
                        frame, result, frame_idx, mlp_classifier,
                        instance_counts, saved_count
                    )
                    if matched > 0:
                        saved_count = instance_counts.get('_total', 0)

                    self.progress_signal.emit(proc_idx + 1, len(target_frame_indices), frame_idx, detected)

                    if proc_idx % 10 == 0 or matched > 0:
                        self.log(f"帧 {frame_idx}: 检测 {detected}, 匹配 {matched}, 已保存 {saved_count} 张")

                cap.release()
                saved_count = instance_counts.get('_total', 0)
                self.log(f"提取完成! 共保存 {saved_count} 张图片 → {self.output_dir}")
                self.finish_signal.emit(True, f"成功提取 {saved_count} 张图片")

        except ImportError as e:
            self.finish_signal.emit(False, f"缺少依赖: {str(e)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finish_signal.emit(False, str(e))

    def _process_frame(self, frame, result, frame_idx, mlp_classifier, instance_counts, saved_count):
        """Process a single frame and return (detected_count, matched_count)."""
        from core.mlp_classifier import MLPClassifier

        if self.exposure_enabled and self.exposure_factor < 1.0:
            frame = np.clip(frame * self.exposure_factor, 0, 255).astype(np.uint8)

        boxes = result.boxes if result else None
        keypoints = result.keypoints if result and hasattr(result, 'keypoints') else None

        kp_xy = None
        kp_conf = None
        if keypoints is not None:
            kp_xy = keypoints.xy.cpu().numpy() if keypoints.xy is not None else None
            kp_conf = keypoints.conf.cpu().numpy() if hasattr(keypoints, 'conf') and keypoints.conf is not None else None

        detected = 0
        matched = 0
        is_1ch = (mlp_classifier.input_channels == 1)

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for j in range(len(xyxy)):
                detected += 1

                color_roi = None
                mlp_input = None
                corrected_kpts = None

                try:
                    if self.crop_mode == "bbox":
                        roi = MLPClassifier.extract_armor_roi_bbox(
                            frame, xyxy[j],
                            output_size=mlp_classifier.input_size,
                            binarize=is_1ch
                        )
                        if roi is None:
                            continue
                        mlp_input = roi
                    else:
                        if kp_xy is not None and j < len(kp_xy):
                            kpts_j = kp_xy[j]
                            has_valid_kpts = False
                            if kp_conf is not None and j < len(kp_conf):
                                has_valid_kpts = bool(np.all(kp_conf[j] > 0.3))
                            elif kp_xy is not None:
                                has_valid_kpts = True

                            if has_valid_kpts:
                                refined_kpts, _ = MLPClassifier.refine_keypoints_and_clean(
                                    frame, kpts_j, xyxy[j]
                                )
                                if refined_kpts is not None:
                                    corrected_kpts = refined_kpts

                                color_roi = MLPClassifier.extract_armor_roi(
                                    frame, xyxy[j], kpts_j,
                                    output_size=mlp_classifier.input_size,
                                    corrected_kpts=corrected_kpts,
                                    binarize=False
                                )

                                if color_roi is None:
                                    continue

                                if is_1ch:
                                    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
                                    _, mlp_input = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                                else:
                                    mlp_input = color_roi
                            else:
                                continue
                        else:
                            continue

                    if mlp_input is None:
                        continue

                    mlp_name, mlp_confidence, _ = mlp_classifier.predict(mlp_input)

                    if mlp_confidence < self.mlp_threshold:
                        continue

                    if self.target_category:
                        try:
                            target_id = int(self.target_category)
                            if mlp_classifier.labels is not None and mlp_name in mlp_classifier.labels:
                                mlp_class_id = mlp_classifier.labels.index(mlp_name)
                            else:
                                mlp_class_id = -1
                            if mlp_class_id != target_id:
                                continue
                        except ValueError:
                            if mlp_name != self.target_category:
                                continue
                except Exception:
                    continue

                if self.save_mode == "perspective" and color_roi is not None:
                    save_img = cv2.resize(color_roi, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)
                else:
                    cropped = frame[int(xyxy[j][1]):int(xyxy[j][3]), int(xyxy[j][0]):int(xyxy[j][2])]
                    if cropped.size == 0:
                        continue
                    save_img = letterbox_resize(cropped, self.output_w, self.output_h)

                instance_counts[mlp_name] = instance_counts.get(mlp_name, 0) + 1
                instance_idx = instance_counts[mlp_name]

                if self.source_type == "image":
                    filename = f"{self.source_prefix}_{mlp_name}_{frame_idx+1:04d}_{instance_idx:02d}.jpg"
                else:
                    filename = f"{self.source_prefix}_{mlp_name}_{frame_idx:06d}_{instance_idx:02d}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, save_img)

                saved_count += 1
                instance_counts['_total'] = saved_count
                matched += 1

        return detected, matched


class ArmorExtractorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from ui.icons import Icon, Icons, C
        self.setWindowTitle("装甲板数据提取工具")
        self.setWindowIcon(Icon.icon(Icons.ARMOR, size=24, color=C.AMBER))
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        self.setStyleSheet(EDITORIAL_QSS)

        self.worker = None

        self._setup_ui()
        self._connect_signals()
        self._load_config()
        self._on_source_changed()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        font_bold = QFont("Microsoft YaHei", 10, QFont.Bold)
        font_normal = QFont("Microsoft YaHei", 9)

        input_group = QGroupBox("输入设置")
        input_layout = QGridLayout(input_group)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)

        lbl_src = QLabel("来源类型:")
        lbl_src.setFont(font_bold)
        self.radio_video = QRadioButton("视频")
        self.radio_image = QRadioButton("图片文件夹")
        self.radio_video.setFont(font_normal)
        self.radio_image.setFont(font_normal)
        self.radio_video.setChecked(True)
        src_layout = QHBoxLayout()
        src_layout.addWidget(lbl_src)
        src_layout.addWidget(self.radio_video)
        src_layout.addWidget(self.radio_image)
        src_layout.addStretch()
        input_layout.addLayout(src_layout, 0, 0, 1, 6)

        self.lbl_video = QLabel("视频文件:")
        self.lbl_video.setFont(font_bold)
        self.edt_video = QLineEdit()
        self.edt_video.setFont(font_normal)
        self.edt_video.setPlaceholderText("选择视频文件路径...")
        self.btn_video = QPushButton("选择文件")
        self.btn_video.setFont(font_normal)
        input_layout.addWidget(self.lbl_video, 1, 0, 1, 1)
        input_layout.addWidget(self.edt_video, 1, 1, 1, 4)
        input_layout.addWidget(self.btn_video, 1, 5, 1, 1)

        self.lbl_img_dir = QLabel("图片目录:")
        self.lbl_img_dir.setFont(font_bold)
        self.edt_img_dir = QLineEdit()
        self.edt_img_dir.setFont(font_normal)
        self.edt_img_dir.setPlaceholderText("选择包含图片的文件夹...")
        self.btn_img_dir = QPushButton("选择目录")
        self.btn_img_dir.setFont(font_normal)
        self.lbl_img_count = QLabel("0 张")
        self.lbl_img_count.setFont(font_normal)
        self.lbl_img_count.setProperty("role", "accent")
        input_layout.addWidget(self.lbl_img_dir, 2, 0, 1, 1)
        input_layout.addWidget(self.edt_img_dir, 2, 1, 1, 4)
        input_layout.addWidget(self.btn_img_dir, 2, 5, 1, 1)
        input_layout.addWidget(self.lbl_img_count, 2, 6, 1, 1)

        lbl_yolo = QLabel("YOLO模型:")
        lbl_yolo.setFont(font_bold)
        self.edt_yolo = QLineEdit()
        self.edt_yolo.setFont(font_normal)
        self.edt_yolo.setPlaceholderText("选择YOLO模型路径...")
        self.btn_yolo = QPushButton("选择模型")
        self.btn_yolo.setFont(font_normal)
        self.lbl_yolo_status = QLabel("未加载")
        self.lbl_yolo_status.setFont(font_normal)
        self.lbl_yolo_status.setProperty("role", "muted")
        input_layout.addWidget(lbl_yolo, 3, 0, 1, 1)
        input_layout.addWidget(self.edt_yolo, 3, 1, 1, 4)
        input_layout.addWidget(self.btn_yolo, 3, 5, 1, 1)
        input_layout.addWidget(self.lbl_yolo_status, 4, 1, 1, 1)

        lbl_mlp = QLabel("MLP模型:")
        lbl_mlp.setFont(font_bold)
        self.edt_mlp = QLineEdit()
        self.edt_mlp.setFont(font_normal)
        self.edt_mlp.setPlaceholderText("选择MLP模型路径...")
        self.btn_mlp = QPushButton("选择模型")
        self.btn_mlp.setFont(font_normal)
        self.lbl_mlp_status = QLabel("未加载")
        self.lbl_mlp_status.setFont(font_normal)
        self.lbl_mlp_status.setProperty("role", "muted")
        input_layout.addWidget(lbl_mlp, 5, 0, 1, 1)
        input_layout.addWidget(self.edt_mlp, 5, 1, 1, 4)
        input_layout.addWidget(self.btn_mlp, 5, 5, 1, 1)
        input_layout.addWidget(self.lbl_mlp_status, 6, 1, 1, 1)

        main_layout.addWidget(input_group)

        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout(param_group)
        param_layout.setContentsMargins(8, 8, 8, 8)
        param_layout.setSpacing(8)

        lbl_target = QLabel("目标类别:")
        lbl_target.setFont(font_bold)
        self.edt_target = QLineEdit()
        self.edt_target.setFont(font_normal)
        self.edt_target.setPlaceholderText("留空表示所有类别")
        param_layout.addWidget(lbl_target, 0, 0, 1, 1)
        param_layout.addWidget(self.edt_target, 0, 1, 1, 1)

        lbl_yolo_conf = QLabel("YOLO阈值:")
        lbl_yolo_conf.setFont(font_bold)
        self.spin_yolo_conf = QDoubleSpinBox()
        self.spin_yolo_conf.setFont(font_normal)
        self.spin_yolo_conf.setRange(0.01, 1.0)
        self.spin_yolo_conf.setDecimals(2)
        self.spin_yolo_conf.setValue(0.25)
        param_layout.addWidget(lbl_yolo_conf, 0, 2, 1, 1)
        param_layout.addWidget(self.spin_yolo_conf, 0, 3, 1, 1)

        lbl_mlp_thresh = QLabel("MLP阈值:")
        lbl_mlp_thresh.setFont(font_bold)
        self.spin_mlp_thresh = QDoubleSpinBox()
        self.spin_mlp_thresh.setFont(font_normal)
        self.spin_mlp_thresh.setRange(0.01, 1.0)
        self.spin_mlp_thresh.setDecimals(2)
        self.spin_mlp_thresh.setValue(0.5)
        param_layout.addWidget(lbl_mlp_thresh, 0, 4, 1, 1)
        param_layout.addWidget(self.spin_mlp_thresh, 0, 5, 1, 1)

        self.lbl_interval = QLabel("抽帧间隔:")
        self.lbl_interval.setFont(font_bold)
        self.spin_interval = QSpinBox()
        self.spin_interval.setFont(font_normal)
        self.spin_interval.setRange(1, 1000)
        self.spin_interval.setValue(5)
        param_layout.addWidget(self.lbl_interval, 1, 0, 1, 1)
        param_layout.addWidget(self.spin_interval, 1, 1, 1, 1)

        lbl_crop = QLabel("裁剪模式:")
        lbl_crop.setFont(font_bold)
        self.radio_bbox = QRadioButton("Bbox裁剪")
        self.radio_persp = QRadioButton("透视变换")
        self.radio_bbox.setFont(font_normal)
        self.radio_persp.setFont(font_normal)
        self.radio_bbox.setChecked(True)
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(self.radio_bbox)
        crop_layout.addWidget(self.radio_persp)
        crop_layout.addStretch()
        param_layout.addWidget(lbl_crop, 1, 2, 1, 1)
        param_layout.addLayout(crop_layout, 1, 3, 1, 3)

        lbl_size = QLabel("输出尺寸:")
        lbl_size.setFont(font_bold)
        self.spin_out_w = QSpinBox()
        self.spin_out_w.setFont(font_normal)
        self.spin_out_w.setRange(1, 1000)
        self.spin_out_w.setValue(28)
        self.lbl_x = QLabel("×")
        self.lbl_x.setFont(font_normal)
        self.spin_out_h = QSpinBox()
        self.spin_out_h.setFont(font_normal)
        self.spin_out_h.setRange(1, 1000)
        self.spin_out_h.setValue(20)
        param_layout.addWidget(lbl_size, 2, 0, 1, 1)
        param_layout.addWidget(self.spin_out_w, 2, 1, 1, 1)
        param_layout.addWidget(self.lbl_x, 2, 2, 1, 1)
        param_layout.addWidget(self.spin_out_h, 2, 3, 1, 1)

        lbl_output = QLabel("输出目录:")
        lbl_output.setFont(font_bold)
        self.edt_output = QLineEdit()
        self.edt_output.setFont(font_normal)
        self.edt_output.setPlaceholderText("选择输出目录...")
        self.btn_output = QPushButton("选择目录")
        self.btn_output.setFont(font_normal)
        param_layout.addWidget(lbl_output, 3, 0, 1, 1)
        param_layout.addWidget(self.edt_output, 3, 1, 1, 4)
        param_layout.addWidget(self.btn_output, 3, 5, 1, 1)

        lbl_exposure = QLabel("降低曝光:")
        lbl_exposure.setFont(font_bold)
        self.chk_exposure = QCheckBox("启用")
        self.chk_exposure.setFont(font_normal)
        lbl_exp_factor = QLabel("系数:")
        lbl_exp_factor.setFont(font_normal)
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setFont(font_normal)
        self.spin_exposure.setRange(0.1, 1.0)
        self.spin_exposure.setDecimals(2)
        self.spin_exposure.setSingleStep(0.05)
        self.spin_exposure.setValue(1.0)
        self.spin_exposure.setPrefix("")
        self.spin_exposure.setSuffix("")
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(self.chk_exposure)
        exp_layout.addWidget(lbl_exp_factor)
        exp_layout.addWidget(self.spin_exposure)
        exp_layout.addStretch()
        param_layout.addWidget(lbl_exposure, 4, 0, 1, 1)
        param_layout.addLayout(exp_layout, 4, 1, 1, 4)

        lbl_save = QLabel("保存模式:")
        lbl_save.setFont(font_bold)
        self.radio_save_persp = QRadioButton("透视变换区域")
        self.radio_save_bbox = QRadioButton("Bbox原始裁剪")
        self.radio_save_persp.setFont(font_normal)
        self.radio_save_bbox.setFont(font_normal)
        self.radio_save_persp.setChecked(True)
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.radio_save_persp)
        save_layout.addWidget(self.radio_save_bbox)
        save_layout.addStretch()
        param_layout.addWidget(lbl_save, 5, 0, 1, 1)
        param_layout.addLayout(save_layout, 5, 1, 1, 4)

        main_layout.addWidget(param_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_start = QPushButton("开始提取")
        self.btn_start.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFont(QFont("Microsoft YaHei", 10))
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        main_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(font_normal)
        self.progress_bar.setMinimumHeight(20)
        self.lbl_progress = QLabel("就绪")
        self.lbl_progress.setFont(font_normal)
        self.lbl_progress.setAlignment(Qt.AlignCenter)

        self.lbl_saved = QLabel("已保存: 0 张")
        self.lbl_saved.setFont(font_normal)
        self.lbl_saved.setProperty("role", "accent")

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.lbl_progress)
        main_layout.addWidget(self.lbl_saved)

        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group, 1)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

    def _connect_signals(self):
        self.btn_video.clicked.connect(self._on_select_video)
        self.btn_img_dir.clicked.connect(self._on_select_image_dir)
        self.btn_yolo.clicked.connect(self._on_select_yolo)
        self.btn_mlp.clicked.connect(self._on_select_mlp)
        self.btn_output.clicked.connect(self._on_select_output)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.radio_video.toggled.connect(self._on_source_changed)

    def _on_source_changed(self):
        is_video = self.radio_video.isChecked()
        self.lbl_video.setVisible(is_video)
        self.edt_video.setVisible(is_video)
        self.btn_video.setVisible(is_video)
        self.lbl_img_dir.setVisible(not is_video)
        self.edt_img_dir.setVisible(not is_video)
        self.btn_img_dir.setVisible(not is_video)
        self.lbl_img_count.setVisible(not is_video)
        self.spin_interval.setVisible(is_video)
        self.lbl_interval.setVisible(is_video)

    def _get_default_dir(self, category):
        """Get default directory based on category."""
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

    def _count_images_in_dir(self, img_dir):
        if not img_dir or not os.path.isdir(img_dir):
            return 0
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP')
        count = 0
        for ext in exts:
            count += len(glob.glob(os.path.join(img_dir, ext)))
        return count

    def _load_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig().get_tool_config("armor_extractor")
        self.edt_video.setText(cfg.get("video_path", ""))
        self.edt_img_dir.setText(cfg.get("image_dir", ""))
        img_count = self._count_images_in_dir(cfg.get("image_dir", ""))
        self.lbl_img_count.setText(f"{img_count} 张")
        self.edt_yolo.setText(cfg.get("yolo_path", ""))
        self.edt_mlp.setText(cfg.get("mlp_path", ""))
        self.edt_output.setText(cfg.get("output_dir", ""))
        self.edt_target.setText(cfg.get("target_category", ""))
        self.spin_yolo_conf.setValue(float(cfg.get("yolo_conf", 0.25)))
        self.spin_mlp_thresh.setValue(float(cfg.get("mlp_thresh", 0.5)))
        self.spin_interval.setValue(int(cfg.get("frame_interval", 5)))
        self.spin_out_w.setValue(int(cfg.get("output_w", 28)))
        self.spin_out_h.setValue(int(cfg.get("output_h", 20)))

        source_type = cfg.get("source_type", "video")
        if source_type == "image":
            self.radio_image.setChecked(True)
        else:
            self.radio_video.setChecked(True)

        crop_mode = cfg.get("crop_mode", "bbox")
        if crop_mode == "perspective":
            self.radio_persp.setChecked(True)
        else:
            self.radio_bbox.setChecked(True)

        self.chk_exposure.setChecked(bool(cfg.get("exposure_enabled", False)))
        self.spin_exposure.setValue(float(cfg.get("exposure_factor", 1.0)))

        save_mode = cfg.get("save_mode", "perspective")
        if save_mode == "bbox":
            self.radio_save_bbox.setChecked(True)
        else:
            self.radio_save_persp.setChecked(True)

    def _save_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig()
        crop_mode = "perspective" if self.radio_persp.isChecked() else "bbox"
        source_type = "image" if self.radio_image.isChecked() else "video"
        save_mode = "bbox" if self.radio_save_bbox.isChecked() else "perspective"
        cfg.set("armor_extractor", "source_type", source_type)
        cfg.set("armor_extractor", "video_path", self.edt_video.text())
        cfg.set("armor_extractor", "image_dir", self.edt_img_dir.text())
        cfg.set("armor_extractor", "yolo_path", self.edt_yolo.text())
        cfg.set("armor_extractor", "mlp_path", self.edt_mlp.text())
        cfg.set("armor_extractor", "output_dir", self.edt_output.text())
        cfg.set("armor_extractor", "target_category", self.edt_target.text())
        cfg.set("armor_extractor", "yolo_conf", self.spin_yolo_conf.value())
        cfg.set("armor_extractor", "mlp_thresh", self.spin_mlp_thresh.value())
        cfg.set("armor_extractor", "frame_interval", self.spin_interval.value())
        cfg.set("armor_extractor", "output_w", self.spin_out_w.value())
        cfg.set("armor_extractor", "output_h", self.spin_out_h.value())
        cfg.set("armor_extractor", "crop_mode", crop_mode)
        cfg.set("armor_extractor", "save_mode", save_mode)
        cfg.set("armor_extractor", "exposure_enabled", self.chk_exposure.isChecked())
        cfg.set("armor_extractor", "exposure_factor", self.spin_exposure.value())
        cfg.save()

    def _on_select_video(self):
        default_dir = self._get_default_dir("origin_data")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", default_dir,
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v *.ts *.3gp);;所有文件 (*)"
        )
        if path:
            self.edt_video.setText(path)

    def _on_select_image_dir(self):
        default_dir = self._get_default_dir("origin_data")
        path = QFileDialog.getExistingDirectory(
            self, "选择图片目录", default_dir
        )
        if path:
            self.edt_img_dir.setText(path)
            img_count = self._count_images_in_dir(path)
            self.lbl_img_count.setText(f"{img_count} 张")

    def _on_select_yolo(self):
        default_dir = self._get_default_dir("yolo")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型", default_dir,
            "模型文件 (*.pt *.onnx);;所有文件 (*)"
        )
        if path:
            self.edt_yolo.setText(path)
            self.lbl_yolo_status.setText("已选择")
            self.lbl_yolo_status.setProperty("role", "status_success")

    def _on_select_mlp(self):
        default_dir = self._get_default_dir("mlp")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择MLP模型", default_dir,
            "模型文件 (*.pt *.pth *.onnx);;所有文件 (*)"
        )
        if path:
            self.edt_mlp.setText(path)
            self.lbl_mlp_status.setText("已选择")
            self.lbl_mlp_status.setProperty("role", "status_success")

    def _on_select_output(self):
        default_dir = self._get_default_dir("origin_data")
        path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", default_dir
        )
        if path:
            self.edt_output.setText(path)

    def _on_start(self):
        is_video = self.radio_video.isChecked()
        video_path = self.edt_video.text().strip()
        image_dir = self.edt_img_dir.text().strip()
        yolo_path = self.edt_yolo.text().strip()
        mlp_path = self.edt_mlp.text().strip()
        output_dir = self.edt_output.text().strip()

        if is_video:
            if not video_path or not os.path.exists(video_path):
                QMessageBox.warning(self, "错误", "请选择有效的视频文件")
                return
        else:
            if not image_dir or not os.path.isdir(image_dir):
                QMessageBox.warning(self, "错误", "请选择有效的图片目录")
                return
            img_count = self._count_images_in_dir(image_dir)
            if img_count == 0:
                QMessageBox.warning(self, "错误", "图片目录为空")
                return

        if not yolo_path or not os.path.exists(yolo_path):
            QMessageBox.warning(self, "错误", "请选择有效的YOLO模型")
            return
        if not mlp_path or not os.path.exists(mlp_path):
            QMessageBox.warning(self, "错误", "请选择有效的MLP模型")
            return
        if not output_dir:
            QMessageBox.warning(self, "错误", "请选择输出目录")
            return

        crop_mode = "perspective" if self.radio_persp.isChecked() else "bbox"
        save_mode = "bbox" if self.radio_save_bbox.isChecked() else "perspective"

        if is_video:
            source_prefix = os.path.splitext(os.path.basename(video_path))[0]
        else:
            source_prefix = os.path.basename(os.path.normpath(image_dir))

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_progress.setText("正在提取...")
        self.lbl_saved.setText("已保存: 0 张")
        self.log_text.clear()
        self.statusBar.showMessage("处理中...")

        self.worker = ArmorExtractWorker(
            yolo_model_path=yolo_path,
            mlp_model_path=mlp_path,
            target_category=self.edt_target.text().strip(),
            yolo_conf=self.spin_yolo_conf.value(),
            mlp_threshold=self.spin_mlp_thresh.value(),
            crop_mode=crop_mode,
            output_w=self.spin_out_w.value(),
            output_h=self.spin_out_h.value(),
            output_dir=output_dir,
            source_type="video" if is_video else "image",
            video_path=video_path,
            image_dir=image_dir,
            frame_interval=self.spin_interval.value(),
            source_prefix=source_prefix,
            exposure_enabled=self.chk_exposure.isChecked(),
            exposure_factor=self.spin_exposure.value(),
            save_mode=save_mode
        )
        self.worker.log_signal.connect(self._on_log)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finish_signal.connect(self._on_finished)
        self.worker.saved_count_signal.connect(self._on_saved_count)
        self.worker.start()

    def _on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress.setText("已停止")
        self.statusBar.showMessage("已停止")

    def _on_log(self, msg):
        self.log_text.append(msg)
        self.log_text.ensureCursorVisible()

    def _on_progress(self, current, total, frame_idx, detected):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.lbl_progress.setText(f"处理中: {current}/{total} (帧 {frame_idx}, 检测 {detected})")

    def _on_saved_count(self, count):
        self.lbl_saved.setText(f"已保存: {count} 张")

    def _on_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self.lbl_progress.setText(msg)
            self.statusBar.showMessage("提取完成")
            QMessageBox.information(self, "完成", msg)
        else:
            self.lbl_progress.setText(f"失败: {msg}")
            self.statusBar.showMessage("失败")
            QMessageBox.warning(self, "错误", msg)

    def closeEvent(self, event):
        self._save_config()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = ArmorExtractorWindow()
    window.show()
    sys.exit(app.exec())
