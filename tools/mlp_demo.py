# -*- coding: utf-8 -*-
"""
MLP 分类推理演示工具
加载 .pt / .onnx 模型，自动识别输入通道和大小，
对文件夹图片或视频逐帧进行分类推理并展示结果
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import time
import glob
from collections import Counter

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QDoubleSpinBox, QGroupBox, QRadioButton, QButtonGroup,
    QTextEdit, QProgressBar, QMessageBox, QSplitter, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap

from core.tool_config import ToolConfig


# =========================================================
# 加载编辑风样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()

# 结果预览图长边上限：防止批量分类时全分辨率原图在 _results 中累积撑爆内存
PREVIEW_MAX_LONG_EDGE = 1024


def make_preview(cv_img, max_long_edge: int = PREVIEW_MAX_LONG_EDGE):
    """把全分辨率 BGR 图压缩为长边受限的预览副本；小图原样返回引用。"""
    if cv_img is None:
        return None
    h, w = cv_img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return cv_img
    scale = max_long_edge / float(long_edge)
    return cv2.resize(cv_img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


class MLPLoadWorker(QThread):
    loaded = Signal(object, bool, str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            from core.mlp_classifier import MLPClassifier
            classifier = MLPClassifier(self.model_path)
            labels_preview = ', '.join(classifier.labels[:5])
            if len(classifier.labels) > 5:
                labels_preview += '...'
            channels_str = f"{classifier.input_channels}通道" if classifier.input_channels else "?通道"
            size_str = f"{classifier.input_size[0]}x{classifier.input_size[1]}" if classifier.input_size else "?x?"
            msg = (f"加载成功\n"
                   f"格式: {classifier.model_type}\n"
                   f"输入: {channels_str} {size_str}\n"
                   f"类别数: {len(classifier.labels)}\n"
                   f"类别: {labels_preview}")
            self.loaded.emit(classifier, True, msg)
        except Exception as e:
            self.loaded.emit(None, False, f"模型加载失败: {e}")


class MLPClassifyWorker(QThread):
    result_signal = Signal(str, str, float, float, object)
    progress_signal = Signal(int, int, str)
    finished_signal = Signal(bool, str)
    stats_signal = Signal(dict)

    def __init__(self, classifier, image_paths, image_dir=None,
                 save_image=False, output_dir=""):
        super().__init__()
        self.classifier = classifier
        self.image_paths = image_paths
        self.image_dir = image_dir
        self.save_image = save_image
        self.output_dir = output_dir
        self._stop = False
        self._pause = False
        self._results = []
        from PySide6.QtCore import QMutex, QWaitCondition
        self._pause_mutex = QMutex()
        self._pause_condition = QWaitCondition()

    def stop(self):
        self._stop = True
        self.resume()

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False
        if self._pause_condition:
            self._pause_condition.wakeAll()

    def _apply_exif_orientation(self, img, img_path):
        try:
            import PIL.Image
            import io
            pil_img = PIL.Image.open(img_path)
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(pil_img)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return img

    def run(self):
        total = len(self.image_paths)
        class_counts = Counter()
        total_conf = 0.0
        total_ms = 0.0
        count = 0

        for i, img_path in enumerate(self.image_paths):
            if self._stop:
                self._save_results_json()
                self.finished_signal.emit(True, "已停止")
                return

            self._pause_mutex.lock()
            while self._pause:
                self._pause_condition.wait(self._pause_mutex)
            self._pause_mutex.unlock()

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = self._apply_exif_orientation(img, img_path)

            try:
                class_name, confidence, infer_ms = self.classifier.predict(img)
            except Exception as e:
                class_name = f"错误: {e}"
                confidence = 0.0
                infer_ms = 0.0

            class_counts[class_name] += 1
            total_conf += confidence
            total_ms += infer_ms
            count += 1

            self._results.append({
                "filename": os.path.basename(img_path),
                "class": class_name,
                "confidence": float(confidence),
                "infer_ms": float(infer_ms),
            })

            if self.save_image and self.output_dir and img is not None:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(self.output_dir, f"{basename}_classified.jpg")
                cv2.imwrite(out_path, img)

            filename = os.path.basename(img_path)
            self.result_signal.emit(filename, class_name, confidence, infer_ms, img)
            self.progress_signal.emit(i + 1, total, filename)

            time.sleep(0.02)

        self._save_results_json()

        avg_conf = total_conf / count if count > 0 else 0
        avg_ms = total_ms / count if count > 0 else 0

        stats = {
            "total": count,
            "class_counts": dict(class_counts),
            "avg_conf": avg_conf,
            "avg_ms": avg_ms,
            "labels": self.classifier.labels,
        }
        self.stats_signal.emit(stats)
        self.finished_signal.emit(True, f"推理完成，共 {count} 张图片")

    def _save_results_json(self):
        if self._results and self.output_dir:
            import json
            out_path = os.path.join(self.output_dir, "results.json")
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(self._results, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


class MLPVideoWorker(QThread):
    frame_signal = Signal(object, dict)
    finished_signal = Signal(bool, str)
    stats_signal = Signal(dict)

    def __init__(self, classifier, video_path, save_video=False, output_path=""):
        super().__init__()
        self.classifier = classifier
        self.video_path = video_path
        self.save_video = save_video
        self.output_path = output_path
        self._stop = False
        self._pause = False
        self._results = []
        from PySide6.QtCore import QMutex, QWaitCondition
        self._pause_mutex = QMutex()
        self._pause_condition = QWaitCondition()

    def stop(self):
        self._stop = True
        self.resume()

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False
        if self._pause_condition:
            self._pause_condition.wakeAll()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit(False, f"无法打开视频: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        class_counts = Counter()
        total_conf = 0.0
        total_ms = 0.0
        count = 0

        writer = None
        if self.save_video and self.output_path:
            ret_test, frame_test = cap.read()
            if ret_test:
                h, w = frame_test.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while not self._stop:
            self._pause_mutex.lock()
            while self._pause:
                self._pause_condition.wait(self._pause_mutex)
            self._pause_mutex.unlock()

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            try:
                class_name, confidence, infer_ms = self.classifier.predict(frame)
            except Exception as e:
                class_name = f"错误: {e}"
                confidence = 0.0
                infer_ms = 0.0

            class_counts[class_name] += 1
            total_conf += confidence
            total_ms += infer_ms
            count += 1

            self._results.append({
                "frame": frame_idx,
                "class": class_name,
                "confidence": float(confidence),
                "infer_ms": float(infer_ms),
            })

            if writer:
                writer.write(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

            stats = {"frame": frame_idx, "total_frames": total_frames, "class_name": class_name, "confidence": confidence, "infer_ms": infer_ms}
            self.frame_signal.emit(qimg, stats)

            wait_ms = max(1, int(1000 / fps)) if fps > 0 else 33
            time.sleep(wait_ms / 1000.0 * 0.3)

        cap.release()
        if writer:
            writer.release()

        self._save_results_json()

        avg_conf = total_conf / count if count > 0 else 0
        avg_ms = total_ms / count if count > 0 else 0
        final_stats = {
            "total": count,
            "class_counts": dict(class_counts),
            "avg_conf": avg_conf,
            "avg_ms": avg_ms,
            "labels": self.classifier.labels,
        }
        self.stats_signal.emit(final_stats)
        self.finished_signal.emit(True, f"视频推理完成，共处理 {count} 帧")

    def _save_results_json(self):
        if self._results and self.output_path:
            import json
            out_path = self.output_path.rsplit('.', 1)[0] + '_results.json'
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(self._results, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


class MLPDemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLP 分类推理演示")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)

        self._classifier = None
        self._load_worker = None
        self._classify_worker = None
        self._video_worker = None
        self._image_paths = []
        self._image_index = 0
        self._results = []
        self._is_video_mode = False
        self._current_stats = None

        self.cfg = ToolConfig()
        self._init_ui()
        self._connect_signals()
        self._load_config()

    def _init_ui(self):
        font_bold = QFont("Microsoft YaHei", 10, QFont.Bold)
        font_normal = QFont("Microsoft YaHei", 9)

        self.setStyleSheet(EDITORIAL_QSS)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ==================== 左侧面板 ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # ---- 模型选择 ----
        model_group = QGroupBox("模型选择")
        model_group.setFont(font_bold)
        model_layout = QVBoxLayout(model_group)

        h1 = QHBoxLayout()
        self.edt_model = QLineEdit()
        self.edt_model.setPlaceholderText("选择 MLP 模型文件 (.pt / .onnx)...")
        self.edt_model.setFont(font_normal)
        self.btn_model = QPushButton("浏览")
        self.btn_model.setFont(font_normal)
        h1.addWidget(self.edt_model, 1)
        h1.addWidget(self.btn_model)
        model_layout.addLayout(h1)

        self.lbl_model_info = QLabel("模型: 未加载")
        self.lbl_model_info.setProperty("role", "caption")
        model_layout.addWidget(self.lbl_model_info)

        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.btn_load_model.setProperty("role", "primary")
        self.btn_load_model.setCursor(Qt.PointingHandCursor)
        model_layout.addWidget(self.btn_load_model)

        left_layout.addWidget(model_group)

        # ---- 数据源选择 ----
        source_group = QGroupBox("数据源")
        source_group.setFont(font_bold)
        source_layout = QVBoxLayout(source_group)

        self.radio_image = QRadioButton("图片文件夹")
        self.radio_video = QRadioButton("视频文件")
        self.radio_image.setChecked(True)
        self.radio_image.setFont(font_normal)
        self.radio_video.setFont(font_normal)
        source_btn_group = QButtonGroup(self)
        source_btn_group.addButton(self.radio_image)
        source_btn_group.addButton(self.radio_video)
        sh = QHBoxLayout()
        sh.addWidget(self.radio_image)
        sh.addWidget(self.radio_video)
        sh.addStretch()
        source_layout.addLayout(sh)

        h2 = QHBoxLayout()
        self.edt_source = QLineEdit()
        self.edt_source.setPlaceholderText("选择图片文件夹或视频文件...")
        self.edt_source.setFont(font_normal)
        self.btn_source = QPushButton("浏览")
        self.btn_source.setFont(font_normal)
        h2.addWidget(self.edt_source, 1)
        h2.addWidget(self.btn_source)
        source_layout.addLayout(h2)

        self.lbl_source_info = QLabel("图片: 0 张")
        self.lbl_source_info.setProperty("role", "caption")
        source_layout.addWidget(self.lbl_source_info)

        left_layout.addWidget(source_group)

        # ---- 保存设置 ----
        save_group = QGroupBox("保存设置")
        save_group.setFont(font_bold)
        save_layout = QVBoxLayout(save_group)

        save_opts_layout = QHBoxLayout()
        self.chk_save_image = QCheckBox("保存标注图片")
        self.chk_save_image.setFont(font_normal)
        self.chk_save_video = QCheckBox("保存标注视频")
        self.chk_save_video.setFont(font_normal)
        save_opts_layout.addWidget(self.chk_save_image)
        save_opts_layout.addWidget(self.chk_save_video)
        save_layout.addLayout(save_opts_layout)

        out_h = QHBoxLayout()
        self.edt_out = QLineEdit()
        self.edt_out.setPlaceholderText("选择输出目录...")
        self.edt_out.setFont(font_normal)
        self.edt_out.setReadOnly(True)
        self.btn_out = QPushButton("浏览")
        self.btn_out.setFont(font_normal)
        out_h.addWidget(self.edt_out, 1)
        out_h.addWidget(self.btn_out)
        save_layout.addLayout(out_h)

        left_layout.addWidget(save_group)

        # ---- 操作按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始推理")
        self.btn_start.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.btn_start.setProperty("role", "primary")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.btn_stop.setProperty("role", "danger")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)

        # ---- 进度条 ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # ---- 统计信息 ----
        stats_group = QGroupBox("统计信息")
        stats_group.setFont(font_bold)
        stats_layout = QVBoxLayout(stats_group)
        self.lbl_stats = QLabel("等待推理...")
        self.lbl_stats.setProperty("role", "caption")
        self.lbl_stats.setWordWrap(True)
        stats_layout.addWidget(self.lbl_stats)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()

        # ==================== 右侧显示区域 ====================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.display_label = QLabel("推理结果将在此显示")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(500, 400)
        self.display_label.setProperty("role", "display_mono")
        right_layout.addWidget(self.display_label, 1)

        self.lbl_current_result = QLabel("")
        self.lbl_current_result.setAlignment(Qt.AlignCenter)
        self.lbl_current_result.setMinimumHeight(40)
        self.lbl_current_result.setProperty("role", "mlp_result")
        right_layout.addWidget(self.lbl_current_result)

        # ---- 导航栏 (图片模式) ----
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张 (A)")
        self.btn_next = QPushButton("下一张 (D) ▶")
        self.btn_prev.setFont(font_normal)
        self.btn_next.setFont(font_normal)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addStretch()
        right_layout.addLayout(nav_layout)

        # ---- 结果日志 ----
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        self.log_edit.setFont(QFont("Consolas", 10))
        right_layout.addWidget(self.log_edit)

        # ==================== 布局组合 ====================
        splitter = QSplitter(Qt.Horizontal)
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(450)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def _connect_signals(self):
        self.btn_model.clicked.connect(self._on_select_model)
        self.btn_source.clicked.connect(self._on_select_source)
        self.btn_load_model.clicked.connect(self._on_load_model)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.radio_image.toggled.connect(self._on_source_type_changed)
        self.btn_out.clicked.connect(self._on_select_output)

    def _get_default_dir(self, category):
        base = os.path.dirname(os.path.abspath(__file__))
        if category == "model":
            d = os.path.join(base, "stored_weights", "model", "armor_tradition")
            return d if os.path.exists(d) else os.path.join(base, "output_mlp")
        elif category in ("image", "video", "origin_data"):
            d = os.path.join(base, "origin_data")
            return d if os.path.exists(d) else base
        return base

    def _on_select_model(self):
        default_dir = self.cfg.get_path("mlp_demo", "model_dir", self._get_default_dir("model"))
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 MLP 模型", default_dir,
            "模型文件 (*.pt *.onnx *.pth);;所有文件 (*)"
        )
        if path:
            self.edt_model.setText(path)
            self.cfg.save_path("mlp_demo", "model_dir", os.path.dirname(path))

    def _on_select_source(self):
        if self.radio_image.isChecked():
            default_dir = self.cfg.get_path("mlp_demo", "image_dir", self._get_default_dir("image"))
            path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", default_dir)
            if path:
                self.edt_source.setText(path)
                self.cfg.save_path("mlp_demo", "image_dir", path)
                self._scan_images(path)
        else:
            default_dir = self.cfg.get_path("mlp_demo", "video_dir", self._get_default_dir("video"))
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", default_dir,
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;所有文件 (*)"
            )
            if path:
                self.edt_source.setText(path)
                self.cfg.save_path("mlp_demo", "video_dir", os.path.dirname(path))

    def _on_select_output(self):
        default_dir = self._get_default_dir("origin_data")
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", default_dir)
        if path:
            self.edt_out.setText(path)
            self.cfg.save_path("mlp_demo", "output_dir", path)

    def _scan_images(self, directory):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        self._image_paths = []
        for ext in exts:
            self._image_paths.extend(glob.glob(os.path.join(directory, ext)))
            self._image_paths.extend(glob.glob(os.path.join(directory, ext.upper())))
        self._image_paths.sort()
        self.lbl_source_info.setText(f"图片: {len(self._image_paths)} 张")

    def _on_source_type_changed(self):
        self.edt_source.clear()
        self.lbl_source_info.setText("图片: 0 张")
        self._image_paths = []

    def _on_load_model(self):
        path = self.edt_model.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "模型加载", "请先选择有效的模型文件")
            return

        self.btn_load_model.setEnabled(False)
        self.lbl_model_info.setText("正在加载模型...")
        self.lbl_model_info.setProperty("role", "status_warn")

        self._load_worker = MLPLoadWorker(path)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.start()

    def _on_model_loaded(self, classifier, success, msg):
        self.btn_load_model.setEnabled(True)
        if success:
            self._classifier = classifier
            self.lbl_model_info.setText(msg.replace("\n", " | "))
            self.lbl_model_info.setProperty("role", "status_success")
            self._append_log(f"[模型] 加载成功: {os.path.basename(self.edt_model.text().strip())}")
            self._append_log(f"[模型] 类别: {', '.join(classifier.labels)}")
        else:
            self._classifier = None
            self.lbl_model_info.setText(f"{msg}")
            self.lbl_model_info.setProperty("role", "status_error")
            self._append_log(f"[模型] 加载失败: {msg}")

    def _on_start(self):
        if not self._classifier:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return

        source = self.edt_source.text().strip()
        if not source:
            QMessageBox.warning(self, "提示", "请先选择数据源")
            return

        save_image = self.chk_save_image.isChecked()
        save_video = self.chk_save_video.isChecked()
        output_dir = self.edt_out.text().strip()

        if save_image or save_video:
            if not output_dir:
                QMessageBox.warning(self, "提示", "请选择输出目录")
                return
            os.makedirs(output_dir, exist_ok=True)

        self._results = []
        self._current_stats = None

        if self.radio_image.isChecked():
            if not self._image_paths:
                self._scan_images(source)
            if not self._image_paths:
                QMessageBox.warning(self, "提示", "未找到图片文件")
                return
            self._is_video_mode = False
            self._start_image_classify(save_image, output_dir)
        else:
            if not os.path.isfile(source):
                QMessageBox.warning(self, "提示", "视频文件不存在")
                return
            self._is_video_mode = True
            self._start_video_classify(save_video, output_dir)

    def _start_image_classify(self, save_image=False, output_dir=""):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.progress_bar.setMaximum(len(self._image_paths))
        self.progress_bar.setValue(0)

        self._classify_worker = MLPClassifyWorker(
            self._classifier, self._image_paths,
            save_image=save_image, output_dir=output_dir
        )
        self._classify_worker.result_signal.connect(self._on_image_result)
        self._classify_worker.progress_signal.connect(self._on_image_progress)
        self._classify_worker.stats_signal.connect(self._on_stats)
        self._classify_worker.finished_signal.connect(self._on_classify_finished)
        self._classify_worker.start()

    def _start_video_classify(self, save_video=False, output_dir=""):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setMaximum(0)

        output_path = ""
        if save_video and output_dir:
            basename = os.path.splitext(os.path.basename(self.edt_source.text().strip()))[0]
            output_path = os.path.join(output_dir, f"{basename}_mlp_annotated.mp4")

        self._video_worker = MLPVideoWorker(
            self._classifier, self.edt_source.text().strip(),
            save_video=save_video, output_path=output_path
        )
        self._video_worker.frame_signal.connect(self._on_video_frame)
        self._video_worker.stats_signal.connect(self._on_stats)
        self._video_worker.finished_signal.connect(self._on_video_finished)
        self._video_worker.start()

    def _on_stop(self):
        if self._is_video_mode and self._video_worker:
            self._video_worker.stop()
        elif self._classify_worker:
            self._classify_worker.stop()

    def _on_image_result(self, filename, class_name, confidence, infer_ms, original_img):
        self._results.append({
            "filename": filename,
            "class_name": class_name,
            "confidence": confidence,
            "infer_ms": infer_ms,
            "original": make_preview(original_img),
        })
        if original_img is not None:
            self._display_cv_image(original_img)
        self.lbl_current_result.setText(f"{class_name}  {confidence:.1%}  {infer_ms:.1f}ms")
        self._append_log(f"[{filename}] → {class_name} ({confidence:.2%}) {infer_ms:.1f}ms")

    def _on_image_progress(self, current, total, filename):
        self.progress_bar.setValue(current)

    def _on_video_frame(self, qimg, stats):
        pixmap = QPixmap.fromImage(qimg.copy())
        scaled = pixmap.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled)
        self.lbl_stats.setText(
            f"帧: {stats['frame']}/{stats['total_frames']} | "
            f"分类: {stats['class_name']} | "
            f"置信度: {stats['confidence']:.2%} | "
            f"耗时: {stats['infer_ms']:.1f}ms"
        )

    def _on_stats(self, stats):
        self._current_stats = stats
        lines = [f"总图片: {stats['total']}"]
        lines.append(f"平均置信度: {stats['avg_conf']:.2%}")
        lines.append(f"平均耗时: {stats['avg_ms']:.1f}ms")
        lines.append("")
        lines.append("类别分布:")
        for cls, cnt in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
            pct = cnt / stats["total"] * 100 if stats["total"] > 0 else 0
            lines.append(f"  {cls}: {cnt} ({pct:.1f}%)")
        self.lbl_stats.setText("\n".join(lines))

        if not self._is_video_mode and self._results:
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)

    def _on_classify_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(msg)
        self._save_config()

    def _on_video_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(msg)
        self._save_config()

    def _on_prev(self):
        if self._image_index > 0:
            self._image_index -= 1
            self._show_result(self._image_index)

    def _on_next(self):
        if self._image_index < len(self._results) - 1:
            self._image_index += 1
            self._show_result(self._image_index)

    def _show_result(self, idx):
        if 0 <= idx < len(self._results):
            r = self._results[idx]
            if r["original"] is not None:
                self._display_cv_image(r["original"])
            self.lbl_current_result.setText(
                f"{r['class_name']}  {r['confidence']:.1%}  {r['infer_ms']:.1f}ms"
            )
            self.display_label.setToolTip(
                f"{r['filename']}: {r['class_name']} ({r['confidence']:.2%})"
            )

    def _display_cv_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled)

    def _append_log(self, text):
        self.log_edit.append(text)
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_A or key == Qt.Key_Left:
            self._on_prev()
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self._on_next()
        else:
            super().keyPressEvent(event)

    def _save_config(self):
        self.cfg.save_path("mlp_demo", "model_dir", os.path.dirname(self.edt_model.text()) if self.edt_model.text() else "")
        if self.radio_image.isChecked():
            self.cfg.save_path("mlp_demo", "image_dir", self.edt_source.text())
        else:
            self.cfg.save_path("mlp_demo", "video_dir", os.path.dirname(self.edt_source.text()) if self.edt_source.text() else "")
        self.cfg.set("mlp_demo", "save_image", self.chk_save_image.isChecked())
        self.cfg.set("mlp_demo", "save_video", self.chk_save_video.isChecked())
        self.cfg.save_path("mlp_demo", "output_dir", self.edt_out.text())
        self.cfg.save()

    def _load_config(self):
        model_path = self.cfg.get_path("mlp_demo", "model_dir")
        if model_path and os.path.isdir(model_path):
            pass
        image_dir = self.cfg.get_path("mlp_demo", "image_dir")
        if image_dir and os.path.isdir(image_dir):
            self.edt_source.setText(image_dir)
            self.radio_image.setChecked(True)
            self._scan_images(image_dir)
        self.chk_save_image.setChecked(bool(self.cfg.get("mlp_demo", "save_image", False)))
        self.chk_save_video.setChecked(bool(self.cfg.get("mlp_demo", "save_video", False)))
        output_dir = self.cfg.get_path("mlp_demo", "output_dir")
        if output_dir:
            self.edt_out.setText(output_dir)

    def closeEvent(self, event):
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.stop()
            self._load_worker.wait(2000)
        if self._classify_worker and self._classify_worker.isRunning():
            self._classify_worker.stop()
            self._classify_worker.wait(3000)
        if self._video_worker and self._video_worker.isRunning():
            self._video_worker.stop()
            self._video_worker.wait(3000)
        self._save_config()
        event.accept()
