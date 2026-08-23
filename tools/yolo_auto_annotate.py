# -*- coding: utf-8 -*-
"""
YOLO 自动标注工具 - 基于 Ultralytics YOLO 的批量目标检测/分割/姿态/OBB 自动标注
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import glob
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QGridLayout,
    QProgressBar, QButtonGroup, QRadioButton, QStyle
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtCore import QSettings


# =========================================================
# 加载编辑风样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()


class YoloAutoWindow(QMainWindow):
    def __init__(self, current_dir=None):
        super().__init__()
        from ui.icons import Icon, Icons, C
        self.setWindowTitle("YOLO 自动标注")
        self.setWindowIcon(Icon.icon(Icons.AUTO, size=24, color=C.AMBER))
        self.setMinimumSize(700, 580)
        self.resize(750, 640)
        self.setStyleSheet(EDITORIAL_QSS)

        self.current_dir = current_dir or ""
        self.model = None
        self.task = None
        self.names = None
        self._batch_worker = None
        self._load_worker = None

        self._load_settings()
        self._setup_ui()
        self._connect_signals()
        self._load_config()

    def _load_settings(self):
        self.settings = QSettings("ArmorLabel", "settings")
        self.settings.beginGroup("Parameters")
        self._yolo_conf = float(self.settings.value("YOLO_CONF", 0.25))
        self._yolo_iou = float(self.settings.value("YOLO_IOU", 0.7))
        self._yolo_imgsz = int(self.settings.value("YOLO_IMGSZ", 640))
        self.settings.endGroup()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(14)
        main_layout.setContentsMargins(20, 20, 20, 20)

        font_bold = QFont("Microsoft YaHei", 10, QFont.Bold)
        font_normal = QFont("Microsoft YaHei", 9)

        top_group = QWidget()
        top_layout = QGridLayout(top_group)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        lbl_weight = QLabel("权重文件:")
        lbl_weight.setFont(font_bold)
        self.edt_weight = QLineEdit()
        self.edt_weight.setFont(font_normal)
        self.edt_weight.setPlaceholderText("选择 YOLO .pt 权重文件...")
        self.edt_weight.setReadOnly(True)
        self.btn_weight = QPushButton("选择权重")
        self.btn_weight.setFont(font_normal)
        top_layout.addWidget(lbl_weight, 0, 0, 1, 1)
        top_layout.addWidget(self.edt_weight, 0, 1, 1, 3)
        top_layout.addWidget(self.btn_weight, 0, 4, 1, 1)

        lbl_task = QLabel("任务类型:")
        lbl_task.setFont(font_bold)
        self.lbl_task_value = QLabel("未加载")
        self.lbl_task_value.setFont(font_normal)
        self.lbl_task_value.setProperty("role", "muted")
        top_layout.addWidget(lbl_task, 1, 0, 1, 1)
        top_layout.addWidget(self.lbl_task_value, 1, 1, 1, 1)

        lbl_classes = QLabel("类别数:")
        lbl_classes.setFont(font_bold)
        self.lbl_classes_value = QLabel("-")
        self.lbl_classes_value.setFont(font_normal)
        self.lbl_classes_value.setProperty("role", "muted")
        top_layout.addWidget(lbl_classes, 1, 2, 1, 1)
        top_layout.addWidget(self.lbl_classes_value, 1, 3, 1, 1)

        lbl_status = QLabel("模型状态:")
        lbl_status.setFont(font_bold)
        self.lbl_model_status = QLabel("未加载")
        self.lbl_model_status.setFont(font_normal)
        self.lbl_model_status.setProperty("role", "muted")
        top_layout.addWidget(lbl_status, 1, 4, 1, 1)
        top_layout.addWidget(self.lbl_model_status, 1, 5, 1, 1)

        main_layout.addWidget(top_group)

        from ui.yolo_runtime_switch import YoloRuntimeSwitch
        self.runtime_switch = YoloRuntimeSwitch(self)
        self.runtime_switch.runtime_changed.connect(self._on_runtime_changed)
        main_layout.addWidget(self.runtime_switch)

        param_group = QWidget()
        param_layout = QGridLayout(param_group)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(10)

        lbl_conf = QLabel("置信度阈值:")
        lbl_conf.setFont(font_bold)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setFont(font_normal)
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(self._yolo_conf)
        self.spin_conf.setMinimumWidth(120)

        lbl_iou = QLabel("IOU 阈值:")
        lbl_iou.setFont(font_bold)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setFont(font_normal)
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(self._yolo_iou)

        lbl_imgsz = QLabel("图片大小:")
        lbl_imgsz.setFont(font_bold)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setFont(font_normal)
        self.spin_imgsz.setRange(320, 1280)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(self._yolo_imgsz)
        self.spin_imgsz.setSuffix(" px")

        param_layout.addWidget(lbl_conf, 0, 0)
        param_layout.addWidget(self.spin_conf, 0, 1)
        param_layout.addWidget(lbl_iou, 0, 2)
        param_layout.addWidget(self.spin_iou, 0, 3)
        param_layout.addWidget(lbl_imgsz, 0, 4)
        param_layout.addWidget(self.spin_imgsz, 0, 5)
        param_layout.setColumnStretch(1, 1)
        param_layout.setColumnStretch(3, 1)
        param_layout.setColumnStretch(5, 1)

        main_layout.addWidget(param_group)

        # ---- 单层 / 双层网络模式 ----
        dual_group = QWidget()
        dual_layout = QGridLayout(dual_group)
        dual_layout.setContentsMargins(0, 0, 0, 0)
        dual_layout.setSpacing(10)

        lbl_net_mode = QLabel("网络模式:")
        lbl_net_mode.setFont(font_bold)
        self.net_mode_group = QButtonGroup()
        self.radio_single = QRadioButton("单层 (仅 YOLO)")
        self.radio_dual = QRadioButton("双层 (YOLO + MobileNet)")
        self.radio_single.setChecked(True)
        self.net_mode_group.addButton(self.radio_single, 0)
        self.net_mode_group.addButton(self.radio_dual, 1)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.radio_single)
        mode_row.addWidget(self.radio_dual)
        mode_row.addStretch()
        dual_layout.addWidget(lbl_net_mode, 0, 0, 1, 1)
        dual_layout.addLayout(mode_row, 0, 1, 1, 5)

        lbl_mlp = QLabel("MobileNet .pt:")
        lbl_mlp.setFont(font_bold)
        self.edt_mlp_path = QLineEdit()
        self.edt_mlp_path.setFont(font_normal)
        default_mlp = os.path.join(
            _ROOT, "stored_weights", "model", "armor_tradition", "mobile-net", "mlp_conv.pt"
        )
        self.edt_mlp_path.setPlaceholderText(default_mlp if os.path.exists(default_mlp) else "选择 mlp_conv.pt ...")
        if os.path.exists(default_mlp):
            self.edt_mlp_path.setText(default_mlp)
        self.btn_mlp = QPushButton("选择")
        self.btn_mlp.setFont(font_normal)
        dual_layout.addWidget(lbl_mlp, 1, 0, 1, 1)
        dual_layout.addWidget(self.edt_mlp_path, 1, 1, 1, 4)
        dual_layout.addWidget(self.btn_mlp, 1, 5, 1, 1)

        lbl_mlp_conf = QLabel("分类置信度:")
        lbl_mlp_conf.setFont(font_bold)
        self.spin_mlp_conf = QDoubleSpinBox()
        self.spin_mlp_conf.setFont(font_normal)
        self.spin_mlp_conf.setRange(0.0, 1.0)
        self.spin_mlp_conf.setDecimals(2)
        self.spin_mlp_conf.setSingleStep(0.05)
        self.spin_mlp_conf.setValue(0.50)
        self.spin_mlp_conf.setToolTip("低于此阈值保留 YOLO 类别；armor_id=8(unknown) 也不覆盖")
        dual_layout.addWidget(lbl_mlp_conf, 2, 0, 1, 1)
        dual_layout.addWidget(self.spin_mlp_conf, 2, 1, 1, 1)

        self.lbl_dual_hint = QLabel("双层仅对 pose 任务生效：bbox/关键点听 YOLO，类别 id 听 MobileNet")
        self.lbl_dual_hint.setFont(font_normal)
        self.lbl_dual_hint.setProperty("role", "muted")
        dual_layout.addWidget(self.lbl_dual_hint, 2, 2, 1, 4)

        main_layout.addWidget(dual_group)
        self._set_dual_widgets_enabled(False)

        dir_group = QWidget()
        dir_layout = QGridLayout(dir_group)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(10)

        lbl_dir = QLabel("图片目录:")
        lbl_dir.setFont(font_bold)
        self.edt_dir = QLineEdit()
        self.edt_dir.setFont(font_normal)
        self.edt_dir.setPlaceholderText("选择包含图片的目录...")
        if self.current_dir:
            self.edt_dir.setText(self.current_dir)
        self.btn_dir = QPushButton("选择目录")
        self.btn_dir.setFont(font_normal)
        dir_layout.addWidget(lbl_dir, 0, 0, 1, 1)
        dir_layout.addWidget(self.edt_dir, 0, 1, 1, 4)
        dir_layout.addWidget(self.btn_dir, 0, 5, 1, 1)

        lbl_img_count = QLabel("图片数量:")
        lbl_img_count.setFont(font_bold)
        self.lbl_img_count_value = QLabel("0")
        self.lbl_img_count_value.setFont(font_normal)
        self.lbl_img_count_value.setProperty("role", "accent")
        dir_layout.addWidget(lbl_img_count, 1, 0, 1, 1)
        dir_layout.addWidget(self.lbl_img_count_value, 1, 1, 1, 1)

        lbl_policy = QLabel("处理模式:")
        lbl_policy.setFont(font_bold)
        self.policy_group = QButtonGroup()
        self.radio_overwrite = QRadioButton("覆盖全部")
        self.radio_skip = QRadioButton("跳过已标注")
        self.radio_append = QRadioButton("补全添加")
        self.radio_skip.setChecked(True)
        self.policy_group.addButton(self.radio_overwrite, 0)
        self.policy_group.addButton(self.radio_skip, 1)
        self.policy_group.addButton(self.radio_append, 2)
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_overwrite)
        radio_layout.addWidget(self.radio_skip)
        radio_layout.addWidget(self.radio_append)
        radio_layout.addStretch()
        dir_layout.addWidget(lbl_policy, 1, 2, 1, 1)
        dir_layout.addLayout(radio_layout, 1, 3, 1, 3)

        main_layout.addWidget(dir_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(font_normal)
        self.progress_bar.setMinimumHeight(22)
        self.lbl_progress = QLabel("就绪")
        self.lbl_progress.setFont(font_normal)
        self.lbl_progress.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.lbl_progress)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_start = QPushButton("开始标注")
        self.btn_start.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.btn_start.setEnabled(False)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFont(QFont("Microsoft YaHei", 10))
        self.btn_cancel.setEnabled(False)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(btn_layout)

        self.lbl_status = QLabel("请选择权重文件和图片目录")
        self.lbl_status.setFont(QFont("Microsoft YaHei", 8))
        self.lbl_status.setProperty("role", "hint")
        self.lbl_status.setAlignment(Qt.AlignRight)
        main_layout.addWidget(self.lbl_status)

    def _connect_signals(self):
        self.btn_weight.clicked.connect(self._on_select_weight)
        self.btn_dir.clicked.connect(self._on_select_dir)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.edt_dir.textChanged.connect(self._on_dir_changed)
        self.radio_single.toggled.connect(self._on_net_mode_changed)
        self.radio_dual.toggled.connect(self._on_net_mode_changed)
        self.btn_mlp.clicked.connect(self._on_select_mlp)

    def _set_dual_widgets_enabled(self, enabled: bool):
        self.edt_mlp_path.setEnabled(enabled)
        self.btn_mlp.setEnabled(enabled)
        self.spin_mlp_conf.setEnabled(enabled)

    def _on_net_mode_changed(self, _checked=False):
        self._set_dual_widgets_enabled(self.radio_dual.isChecked())

    def _on_select_mlp(self):
        default_dir = self._get_default_dir("mlp")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 MobileNet 权重 (.pt)", default_dir,
            "PyTorch 权重 (*.pt);;所有文件 (*)"
        )
        if path:
            self.edt_mlp_path.setText(path)

    def _on_runtime_changed(self, runtime, previous):
        from core.yolo_runtime import runtime_label, set_runtime
        if self._batch_worker and self._batch_worker.isRunning():
            QMessageBox.warning(self, "提示", "批量标注进行中，无法切换运行时")
            set_runtime(previous)
            self.runtime_switch.sync_from_settings()
            return
        self.model = None
        self.task = None
        self.names = None
        self.lbl_task_value.setText("未加载")
        self.lbl_classes_value.setText("-")
        self.lbl_model_status.setText(f"已切换 {runtime_label(runtime)}，请重新加载权重")
        self.lbl_model_status.setProperty("role", "status_warn")
        self.btn_start.setEnabled(False)

    def _load_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig().get_tool_config("yolo_auto_annotate")
        self.edt_weight.setText(cfg.get("weight_path", ""))
        weight_path = cfg.get("weight_path", "")
        if weight_path and os.path.exists(weight_path):
            self.lbl_model_status.setText("加载中...")
            self.lbl_model_status.setProperty("role", "status_warn")
            self.btn_start.setEnabled(False)
            self.model = None
            self.task = None
            self.names = None
            if self._load_worker and self._load_worker.isRunning():
                self._load_worker.terminate()
                self._load_worker.wait()
            from core.yolo_client import YoloLoadWorker
            self._load_worker = YoloLoadWorker(weight_path)
            self._load_worker.loaded.connect(self._on_model_loaded)
            self._load_worker.start()
        self.spin_conf.setValue(float(cfg.get("conf", self._yolo_conf)))
        self.spin_iou.setValue(float(cfg.get("iou", self._yolo_iou)))
        self.spin_imgsz.setValue(int(cfg.get("imgsz", self._yolo_imgsz)))
        self.edt_dir.setText(cfg.get("image_dir", self.current_dir or ""))
        mode = cfg.get("process_mode", "skip")
        if mode == "overwrite":
            self.radio_overwrite.setChecked(True)
        elif mode == "append":
            self.radio_append.setChecked(True)
        else:
            self.radio_skip.setChecked(True)
        dual = bool(cfg.get("dual_mode", False))
        if dual:
            self.radio_dual.setChecked(True)
        else:
            self.radio_single.setChecked(True)
        mlp_path = cfg.get("mlp_pt_path", "")
        if mlp_path:
            self.edt_mlp_path.setText(mlp_path)
        self.spin_mlp_conf.setValue(float(cfg.get("mlp_conf", 0.5)))
        self._set_dual_widgets_enabled(self.radio_dual.isChecked())

    def _save_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig()
        cfg.set("yolo_auto_annotate", "weight_path", self.edt_weight.text())
        cfg.set("yolo_auto_annotate", "conf", self.spin_conf.value())
        cfg.set("yolo_auto_annotate", "iou", self.spin_iou.value())
        cfg.set("yolo_auto_annotate", "imgsz", self.spin_imgsz.value())
        cfg.set("yolo_auto_annotate", "image_dir", self.edt_dir.text())
        if self.radio_overwrite.isChecked():
            mode = "overwrite"
        elif self.radio_append.isChecked():
            mode = "append"
        else:
            mode = "skip"
        cfg.set("yolo_auto_annotate", "process_mode", mode)
        cfg.set("yolo_auto_annotate", "dual_mode", self.radio_dual.isChecked())
        cfg.set("yolo_auto_annotate", "mlp_pt_path", self.edt_mlp_path.text())
        cfg.set("yolo_auto_annotate", "mlp_conf", self.spin_mlp_conf.value())
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

    def _on_select_weight(self):
        default_dir = self._get_default_dir("yolo")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重文件", default_dir,
            "权重文件 (*.pt *.onnx *.engine);;所有文件 (*)"
        )
        if not path:
            return

        self.edt_weight.setText(path)
        self.lbl_model_status.setText("加载中...")
        self.lbl_model_status.setProperty("role", "status_warn")
        self.btn_start.setEnabled(False)
        self.model = None
        self.task = None
        self.names = None

        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.terminate()
            self._load_worker.wait()

        from core.yolo_client import YoloLoadWorker
        self._load_worker = YoloLoadWorker(path)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.start()

    def _on_model_loaded(self, model, success, msg):
        if success:
            self.model = model
            self.task = model.task
            self.names = model.names
            self.lbl_task_value.setText(self.task)
            self.lbl_task_value.setProperty("role", "status_bold_success")
            self.lbl_classes_value.setText(str(len(self.names)))
            self.lbl_model_status.setText("已加载")
            self.lbl_model_status.setProperty("role", "status_success")
            self._check_can_start()
        else:
            self.lbl_task_value.setText("加载失败")
            self.lbl_task_value.setProperty("role", "status_bold_error")
            self.lbl_model_status.setText(msg)
            self.lbl_model_status.setProperty("role", "status_error")
            QMessageBox.warning(self, "模型加载失败", msg)

    def _on_select_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片目录", self.current_dir)
        if path:
            self.edt_dir.setText(path)
            self.current_dir = path

    def _on_dir_changed(self, text):
        if not text or not os.path.isdir(text):
            self.lbl_img_count_value.setText("0")
            self.btn_start.setEnabled(False)
            return
        images = self._scan_images(text)
        self.lbl_img_count_value.setText(str(len(images)))
        self._check_can_start()

    def _scan_images(self, dir_path):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
        images = []
        for ext in exts:
            images.extend(glob.glob(os.path.join(dir_path, ext)))
            images.extend(glob.glob(os.path.join(dir_path, "**", ext), recursive=True))
        return sorted(set(images))

    def _check_can_start(self):
        if self.model and self.edt_dir.text() and os.path.isdir(self.edt_dir.text()):
            images = self._scan_images(self.edt_dir.text())
            self.btn_start.setEnabled(len(images) > 0)
        else:
            self.btn_start.setEnabled(False)

    def _on_start(self):
        if not self.model:
            QMessageBox.warning(self, "错误", "请先加载 YOLO 模型")
            return

        dir_path = self.edt_dir.text().strip()
        if not dir_path or not os.path.isdir(dir_path):
            QMessageBox.warning(self, "错误", "请选择有效的图片目录")
            return

        images = self._scan_images(dir_path)
        if not images:
            QMessageBox.warning(self, "错误", "目录下没有找到图片")
            return

        conf = self.spin_conf.value()
        iou = self.spin_iou.value()
        imgsz = self.spin_imgsz.value()
        policy_map = {0: "overwrite", 1: "skip", 2: "append"}
        policy = policy_map[self.policy_group.checkedId()]

        dual_mode = self.radio_dual.isChecked()
        mlp_model = None
        mlp_conf = self.spin_mlp_conf.value()
        if dual_mode:
            if self.task != "pose":
                QMessageBox.information(
                    self, "提示",
                    f"当前任务类型为「{self.task}」，双层模式仅对 pose 生效。\n"
                    "将按单层 YOLO 继续标注。"
                )
                dual_mode = False
            else:
                mlp_path = self.edt_mlp_path.text().strip()
                if not mlp_path or not os.path.isfile(mlp_path):
                    QMessageBox.warning(self, "错误", "双层模式请选择有效的 MobileNet .pt 权重")
                    return
                try:
                    from core.yolo_client import load_mlp_conv_pt
                    self.lbl_status.setText("加载 MobileNet...")
                    mlp_model = load_mlp_conv_pt(mlp_path)
                except Exception as e:
                    QMessageBox.warning(self, "MobileNet 加载失败", str(e))
                    return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_progress.setText("正在标注...")

        from core.yolo_client import YoloBatchWorker
        self._batch_worker = YoloBatchWorker(
            self.model, self.task, self.names, images, dir_path,
            conf, iou, imgsz, policy,
            dual_mode=dual_mode, mlp_model=mlp_model, mlp_conf=mlp_conf,
        )
        self._batch_worker.progress.connect(self._on_progress)
        self._batch_worker.log.connect(self._on_log)
        self._batch_worker.finished.connect(self._on_finished)
        self._batch_worker.start()

    def _on_cancel(self):
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self._batch_worker.wait(3000)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_progress.setText("已取消")
        self.lbl_status.setText("就绪")

    def _on_progress(self, current, total, filename):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.lbl_progress.setText(f"标注中: {current}/{total} ({pct}%) - {filename}")

    def _on_log(self, msg):
        self.lbl_status.setText(msg)

    def _on_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self.lbl_progress.setText(msg)
            self.lbl_status.setText(msg)
            QMessageBox.information(self, "完成", msg)
        else:
            self.lbl_progress.setText(f"失败: {msg}")
            self.lbl_status.setText(msg)
            QMessageBox.warning(self, "错误", msg)

    def closeEvent(self, event):
        self._save_config()
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self._batch_worker.wait(3000)
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.terminate()
            self._load_worker.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = YoloAutoWindow()
    window.show()
    sys.exit(app.exec())
