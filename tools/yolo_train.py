# -*- coding: utf-8 -*-
"""
YOLO 训练工具 - 基于 Ultralytics YOLO 的图形化训练界面
支持保存/加载训练参数配置到本地 yaml 文件
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# 训练全程强制 matplotlib 非交互式后端, 避免在工作线程里创建 Qt 对象导致
# "QThread: Destroyed while thread is still running" 崩溃
os.environ.setdefault("MPLBACKEND", "Agg")
import io
import re
import glob
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QGridLayout,
    QProgressBar, QGroupBox, QCheckBox, QScrollArea, QFrame,
    QPlainTextEdit, QSizePolicy, QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings
from PySide6.QtGui import QFont


# =========================================================
# 加载编辑风样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()


class StreamCaptor(io.TextIOBase):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._buffer = ""

    def write(self, text):
        if text.strip():
            self._callback(text)
        return len(text)

    def flush(self):
        pass


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self._content_widget = None
        self.toggled.connect(self._on_toggle)

    def setContentWidget(self, widget):
        self._content_widget = widget
        if widget:
            widget.setVisible(self.isChecked())

    def _on_toggle(self, checked):
        if self._content_widget:
            self._content_widget.setVisible(checked)


class YoloTrainWorker(QThread):
    training_started = Signal()
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)
    metrics_signal = Signal(dict)

    def __init__(self, model_path, params):
        super().__init__()
        self.model_path = model_path
        self.params = params
        self._stop = False
        self._captor = None
        self._original_stdout = None

    def stop(self):
        self._stop = True

    def _emit_log(self, text):
        self.log_signal.emit(text.rstrip())

    def _parse_metrics(self, text):
        metrics = {}

        epoch_match = re.search(r'Epoch\[(\d+)/(\d+)\]', text)
        if epoch_match:
            metrics['epoch'] = int(epoch_match.group(1))
            metrics['total_epochs'] = int(epoch_match.group(2))

        box_match = re.search(r'box:\s*([\d.]+)', text)
        if box_match:
            metrics['box_loss'] = float(box_match.group(1))

        cls_match = re.search(r'cls:\s*([\d.]+)', text)
        if cls_match:
            metrics['cls_loss'] = float(cls_match.group(1))

        dfl_match = re.search(r'dfl:\s*([\d.]+)', text)
        if dfl_match:
            metrics['dfl_loss'] = float(dfl_match.group(1))

        mAP_match = re.search(r'mAP50\(B\):\s*([\d.]+)', text)
        if mAP_match:
            metrics['mAP50'] = float(mAP_match.group(1))

        mAP95_match = re.search(r'mAP50-95\(B\):\s*([\d.]+)', text)
        if mAP95_match:
            metrics['mAP50-95'] = float(mAP95_match.group(1))

        if metrics:
            self.metrics_signal.emit(metrics)

    def _on_epoch_end(self, trainer):
        if self._stop:
            self._emit_log("检测到停止信号，正在中断训练...")
            raise KeyboardInterrupt("用户停止训练")

    def run(self):
        self._stop = False
        # 强制 matplotlib 使用非交互式 Agg 后端, 避免在工作线程里创建 Qt 对象
        # (否则 ultralytics 验证画图时 qtagg 后端会触发
        #  "QThread: Destroyed while thread is still running" 崩溃)
        os.environ["MPLBACKEND"] = "Agg"
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        try:
            from core.yolo_runtime import apply_runtime, import_yolo_class
            info = apply_runtime()
            YOLO = import_yolo_class()
            self._emit_log(f"Ultralytics 运行时: {info.runtime} v{info.ultralytics_version}")
            self._emit_log(f"加载模型: {self.model_path}")
            model = YOLO(self.model_path)

            model.add_callback("on_train_epoch_end", self._on_epoch_end)

            self._original_stdout = sys.stdout
            sys.stdout = StreamCaptor(self._emit_log)

            self._emit_log(f"开始训练，参数:")
            for k, v in self.params.items():
                if k not in ('model', 'data'):
                    self._emit_log(f"  {k}: {v}")

            self._emit_log("=" * 50)
            self._emit_log("训练开始，进度显示在状态栏...")
            self._emit_log("=" * 50)

            self.training_started.emit()
            results = model.train(**self.params)

            sys.stdout = self._original_stdout

            if self._stop:
                self._emit_log("训练已被用户停止")
                self.finished_signal.emit(False, "训练已停止")
            else:
                save_dir = getattr(results, 'save_dir', 'runs/train')
                self._emit_log(f"训练完成！结果保存至: {save_dir}")
                self.finished_signal.emit(True, f"训练完成: {save_dir}")

        except KeyboardInterrupt:
            if self._original_stdout:
                sys.stdout = self._original_stdout
            self._emit_log("训练已被用户停止")
            self.finished_signal.emit(False, "训练已停止")
        except Exception as e:
            if self._original_stdout:
                sys.stdout = self._original_stdout
            import traceback
            self._emit_log(f"训练失败: {e}")
            self._emit_log(traceback.format_exc())
            self.finished_signal.emit(False, f"训练失败: {e}")


class OnnxExportWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, pt_path, out_dir, imgsz, device, half=False):
        super().__init__()
        self.pt_path = pt_path
        self.out_dir = out_dir
        self.imgsz = imgsz
        self.device = device
        self.half = half

    def run(self):
        import shutil
        orig_stdout = None
        try:
            from core.yolo_runtime import apply_runtime, import_yolo_class
            info = apply_runtime()
            YOLO = import_yolo_class()
            self.log_signal.emit(
                f"[导出] Ultralytics {info.runtime} v{info.ultralytics_version}"
            )
            self.log_signal.emit(f"[导出] 加载模型: {self.pt_path}")
            model = YOLO(self.pt_path)

            self.log_signal.emit(
                f"[导出] 开始导出 ONNX: device={self.device}, "
                f"imgsz={self.imgsz}, half={self.half}"
            )

            orig_stdout = sys.stdout
            sys.stdout = StreamCaptor(lambda t: self.log_signal.emit(t.rstrip()))

            exported = model.export(
                format='onnx',
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
            )

            sys.stdout = orig_stdout
            orig_stdout = None

            onnx_path = str(exported)
            if self.out_dir and os.path.dirname(onnx_path) != self.out_dir:
                os.makedirs(self.out_dir, exist_ok=True)
                dst = os.path.join(self.out_dir, os.path.basename(onnx_path))
                shutil.move(onnx_path, dst)
                onnx_path = dst

            self.log_signal.emit(f"[导出] 完成: {onnx_path}")
            self.finished_signal.emit(True, onnx_path)

        except Exception as e:
            if orig_stdout is not None:
                sys.stdout = orig_stdout
            import traceback
            self.log_signal.emit(f"[导出] 失败: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


class YoloTrainWindow(QMainWindow):
    worker_training_started = Signal()
    worker_training_finished = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 训练工具")
        self.setMinimumSize(900, 750)
        self.resize(950, 820)
        self.setStyleSheet(EDITORIAL_QSS)

        self.model_path = ""
        self.data_yaml_path = ""
        self.config_dir = os.path.join(os.path.dirname(__file__), "yolo_train_configs")
        os.makedirs(self.config_dir, exist_ok=True)

        self._train_worker = None

        self._setup_ui()
        self._connect_signals()
        self._refresh_config_list()

    def _on_runtime_changed(self, runtime, previous):
        from core.yolo_runtime import runtime_label, set_runtime
        if self._train_worker and self._train_worker.isRunning():
            QMessageBox.warning(self, "提示", "训练进行中，无法切换运行时")
            set_runtime(previous)
            self.runtime_switch.sync_from_settings()
            return
        self._append_log(f"已切换至 {runtime_label(runtime)}，请确认模型权重与版本匹配")

    def _setup_ui(self):
        font_bold = QFont("Microsoft YaHei", 10, QFont.Bold)
        font_normal = QFont("Microsoft YaHei", 9)
        font_mono = QFont("Consolas", 9)


        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # ==================== 顶部状态栏 ====================
        status_widget = QFrame()
        status_widget.setProperty("role", "card")
        status_layout = QGridLayout(status_widget)
        status_layout.setSpacing(12)

        self.status_label = QLabel("等待开始训练")
        self.status_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.status_label.setProperty("role", "muted")

        self.epoch_label = QLabel("Epoch: -/-")
        self.epoch_label.setFont(font_bold)
        self.epoch_label.setProperty("role", "accent")

        self.metrics_label = QLabel("box: -  cls: -  dfl: -")
        self.metrics_label.setFont(font_normal)
        self.metrics_label.setProperty("role", "muted")

        self.map_label = QLabel("mAP50: -  mAP50-95: -")
        self.map_label.setFont(font_normal)
        self.map_label.setProperty("role", "accent")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        status_layout.addWidget(self.status_label, 0, 0, 1, 2)
        status_layout.addWidget(self.epoch_label, 1, 0)
        status_layout.addWidget(self.metrics_label, 1, 1)
        status_layout.addWidget(self.map_label, 1, 2)
        status_layout.addWidget(self.progress_bar, 2, 0, 1, 3)

        main_layout.addWidget(status_widget)

        from ui.yolo_runtime_switch import YoloRuntimeSwitch
        self.runtime_switch = YoloRuntimeSwitch(self)
        self.runtime_switch.runtime_changed.connect(self._on_runtime_changed)
        main_layout.addWidget(self.runtime_switch)

        # ==================== 滚动参数区 ====================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(12)
        scroll.setWidget(scroll_widget)

        # ==================== 基本设置 ====================
        basic_group = CollapsibleGroupBox("基本设置")
        basic_layout = QGridLayout()
        basic_layout.setSpacing(10)

        lbl_model = QLabel("模型权重:")
        lbl_model.setFont(font_bold)
        self.edt_model = QLineEdit()
        self.edt_model.setFont(font_normal)
        self.edt_model.setPlaceholderText("选择 YOLO .pt 或 .yaml 文件...")
        self.edt_model.setReadOnly(True)
        self.btn_model = QPushButton("选择文件")
        self.btn_model.setFont(font_normal)
        basic_layout.addWidget(lbl_model, 0, 0)
        basic_layout.addWidget(self.edt_model, 0, 1, 1, 3)
        basic_layout.addWidget(self.btn_model, 0, 4)

        lbl_data = QLabel("数据集yaml:")
        lbl_data.setFont(font_bold)
        self.edt_data = QLineEdit()
        self.edt_data.setFont(font_normal)
        self.edt_data.setPlaceholderText("选择数据集配置文件 (data.yaml)...")
        self.edt_data.setReadOnly(True)
        self.btn_data = QPushButton("选择文件")
        self.btn_data.setFont(font_normal)
        basic_layout.addWidget(lbl_data, 1, 0)
        basic_layout.addWidget(self.edt_data, 1, 1, 1, 3)
        basic_layout.addWidget(self.btn_data, 1, 4)

        lbl_config = QLabel("配置文件:")
        lbl_config.setFont(font_bold)
        self.cmb_config = QComboBox()
        self.cmb_config.setFont(font_normal)
        self.btn_new_config = QPushButton("新建")
        self.btn_new_config.setFont(font_normal)
        self.btn_save_config = QPushButton("保存")
        self.btn_save_config.setFont(font_normal)
        self.btn_load_config = QPushButton("加载")
        self.btn_load_config.setFont(font_normal)
        config_btn_layout = QHBoxLayout()
        config_btn_layout.addWidget(self.btn_new_config)
        config_btn_layout.addWidget(self.btn_save_config)
        config_btn_layout.addWidget(self.btn_load_config)
        config_btn_layout.addStretch()
        basic_layout.addWidget(lbl_config, 2, 0)
        basic_layout.addWidget(self.cmb_config, 2, 1, 1, 3)
        basic_layout.addLayout(config_btn_layout, 2, 4)

        basic_content = QWidget()
        basic_content.setLayout(basic_layout)
        basic_group.setContentWidget(basic_content)
        scroll_layout.addWidget(basic_group)

        # ==================== 训练核心参数 ====================
        core_group = CollapsibleGroupBox("训练核心参数")
        core_layout = QGridLayout()
        core_layout.setSpacing(10)

        row = 0
        lbl_epochs = QLabel("轮次(epochs):")
        lbl_epochs.setFont(font_bold)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setFont(font_normal)
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(100)
        core_layout.addWidget(lbl_epochs, row, 0)
        core_layout.addWidget(self.spin_epochs, row, 1)

        lbl_patience = QLabel("早停耐心(patience):")
        lbl_patience.setFont(font_bold)
        self.spin_patience = QSpinBox()
        self.spin_patience.setFont(font_normal)
        self.spin_patience.setRange(0, 1000)
        self.spin_patience.setValue(100)
        core_layout.addWidget(lbl_patience, row, 2)
        core_layout.addWidget(self.spin_patience, row, 3)

        row += 1
        lbl_batch = QLabel("批量(batch):")
        lbl_batch.setFont(font_bold)
        self.spin_batch = QSpinBox()
        self.spin_batch.setFont(font_normal)
        self.spin_batch.setRange(-1, 512)
        self.spin_batch.setValue(16)
        core_layout.addWidget(lbl_batch, row, 0)
        core_layout.addWidget(self.spin_batch, row, 1)

        lbl_imgsz = QLabel("图片大小(imgsz):")
        lbl_imgsz.setFont(font_bold)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setFont(font_normal)
        self.spin_imgsz.setRange(32, 1280)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        core_layout.addWidget(lbl_imgsz, row, 2)
        core_layout.addWidget(self.spin_imgsz, row, 3)

        row += 1
        lbl_device = QLabel("设备(device):")
        lbl_device.setFont(font_bold)
        self.cmb_device = QComboBox()
        self.cmb_device.setFont(font_normal)
        self.cmb_device.addItems(["0", "1", "2", "3", "0,1,2,3", "cpu"])
        self.cmb_device.setCurrentText("0")
        core_layout.addWidget(lbl_device, row, 0)
        core_layout.addWidget(self.cmb_device, row, 1)

        lbl_workers = QLabel("workers:")
        lbl_workers.setFont(font_bold)
        self.spin_workers = QSpinBox()
        self.spin_workers.setFont(font_normal)
        self.spin_workers.setRange(0, 32)
        self.spin_workers.setValue(8)
        core_layout.addWidget(lbl_workers, row, 2)
        core_layout.addWidget(self.spin_workers, row, 3)

        row += 1
        lbl_optimizer = QLabel("优化器(optimizer):")
        lbl_optimizer.setFont(font_bold)
        self.cmb_optimizer = QComboBox()
        self.cmb_optimizer.setFont(font_normal)
        self.cmb_optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "Adamax"])
        self.cmb_optimizer.setCurrentText("auto")
        core_layout.addWidget(lbl_optimizer, row, 0)
        core_layout.addWidget(self.cmb_optimizer, row, 1)

        lbl_project = QLabel("项目名(project):")
        lbl_project.setFont(font_bold)
        self.edt_project = QLineEdit()
        self.edt_project.setFont(font_normal)
        self.edt_project.setText("runs/train")
        core_layout.addWidget(lbl_project, row, 2)
        core_layout.addWidget(self.edt_project, row, 3)

        row += 1
        lbl_name = QLabel("实验名(name):")
        lbl_name.setFont(font_bold)
        self.edt_name = QLineEdit()
        self.edt_name.setFont(font_normal)
        self.edt_name.setText("exp")
        core_layout.addWidget(lbl_name, row, 0)
        core_layout.addWidget(self.edt_name, row, 1)

        lbl_existok = QLabel("覆盖已有:")
        lbl_existok.setFont(font_bold)
        self.chk_existok = QCheckBox()
        self.chk_existok.setChecked(False)
        core_layout.addWidget(lbl_existok, row, 2)
        core_layout.addWidget(self.chk_existok, row, 3)

        row += 1
        lbl_pretrained = QLabel("预训练(pretrained):")
        lbl_pretrained.setFont(font_bold)
        self.chk_pretrained = QCheckBox()
        self.chk_pretrained.setChecked(True)
        core_layout.addWidget(lbl_pretrained, row, 0)
        core_layout.addWidget(self.chk_pretrained, row, 1)

        lbl_amp = QLabel("AMP混合精度:")
        lbl_amp.setFont(font_bold)
        self.chk_amp = QCheckBox()
        self.chk_amp.setChecked(True)
        core_layout.addWidget(lbl_amp, row, 2)
        core_layout.addWidget(self.chk_amp, row, 3)

        row += 1
        lbl_coslr = QLabel("余弦学习率:")
        lbl_coslr.setFont(font_bold)
        self.chk_coslr = QCheckBox()
        self.chk_coslr.setChecked(False)
        core_layout.addWidget(lbl_coslr, row, 0)
        core_layout.addWidget(self.chk_coslr, row, 1)

        lbl_deterministic = QLabel("确定性模式:")
        lbl_deterministic.setFont(font_bold)
        self.chk_deterministic = QCheckBox()
        self.chk_deterministic.setChecked(True)
        core_layout.addWidget(lbl_deterministic, row, 2)
        core_layout.addWidget(self.chk_deterministic, row, 3)

        row += 1
        lbl_singlecls = QLabel("单类别模式:")
        lbl_singlecls.setFont(font_bold)
        self.chk_singlecls = QCheckBox()
        self.chk_singlecls.setChecked(False)
        core_layout.addWidget(lbl_singlecls, row, 0)
        core_layout.addWidget(self.chk_singlecls, row, 1)

        lbl_rect = QLabel("矩形训练:")
        lbl_rect.setFont(font_bold)
        self.chk_rect = QCheckBox()
        self.chk_rect.setChecked(False)
        core_layout.addWidget(lbl_rect, row, 2)
        core_layout.addWidget(self.chk_rect, row, 3)

        row += 1
        lbl_val = QLabel("训练时验证:")
        lbl_val.setFont(font_bold)
        self.chk_val = QCheckBox()
        self.chk_val.setChecked(True)
        core_layout.addWidget(lbl_val, row, 0)
        core_layout.addWidget(self.chk_val, row, 1)

        lbl_plots = QLabel("保存图表:")
        lbl_plots.setFont(font_bold)
        self.chk_plots = QCheckBox()
        self.chk_plots.setChecked(True)
        core_layout.addWidget(lbl_plots, row, 2)
        core_layout.addWidget(self.chk_plots, row, 3)

        row += 1
        lbl_resume = QLabel("恢复训练:")
        lbl_resume.setFont(font_bold)
        self.chk_resume = QCheckBox()
        self.chk_resume.setChecked(False)
        core_layout.addWidget(lbl_resume, row, 0)
        core_layout.addWidget(self.chk_resume, row, 1)

        lbl_verbose = QLabel("详细输出:")
        lbl_verbose.setFont(font_bold)
        self.chk_verbose = QCheckBox()
        self.chk_verbose.setChecked(True)
        core_layout.addWidget(lbl_verbose, row, 2)
        core_layout.addWidget(self.chk_verbose, row, 3)

        core_content = QWidget()
        core_content.setLayout(core_layout)
        core_group.setContentWidget(core_content)
        scroll_layout.addWidget(core_group)

        # ==================== 学习率与优化 ====================
        lr_group = CollapsibleGroupBox("学习率与优化")
        lr_layout = QGridLayout()
        lr_layout.setSpacing(10)

        row = 0
        lbl_lr0 = QLabel("初始学习率(lr0):")
        lbl_lr0.setFont(font_bold)
        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setFont(font_normal)
        self.spin_lr0.setRange(0.0, 1.0)
        self.spin_lr0.setDecimals(6)
        self.spin_lr0.setSingleStep(0.001)
        self.spin_lr0.setValue(0.01)
        lr_layout.addWidget(lbl_lr0, row, 0)
        lr_layout.addWidget(self.spin_lr0, row, 1)

        lbl_lrf = QLabel("最终学习率(lrf):")
        lbl_lrf.setFont(font_bold)
        self.spin_lrf = QDoubleSpinBox()
        self.spin_lrf.setFont(font_normal)
        self.spin_lrf.setRange(0.0, 1.0)
        self.spin_lrf.setDecimals(4)
        self.spin_lrf.setSingleStep(0.01)
        self.spin_lrf.setValue(0.01)
        lr_layout.addWidget(lbl_lrf, row, 2)
        lr_layout.addWidget(self.spin_lrf, row, 3)

        row += 1
        lbl_momentum = QLabel("动量(momentum):")
        lbl_momentum.setFont(font_bold)
        self.spin_momentum = QDoubleSpinBox()
        self.spin_momentum.setFont(font_normal)
        self.spin_momentum.setRange(0.0, 1.0)
        self.spin_momentum.setDecimals(3)
        self.spin_momentum.setSingleStep(0.01)
        self.spin_momentum.setValue(0.937)
        lr_layout.addWidget(lbl_momentum, row, 0)
        lr_layout.addWidget(self.spin_momentum, row, 1)

        lbl_weightdecay = QLabel("权重衰减:")
        lbl_weightdecay.setFont(font_bold)
        self.spin_weightdecay = QDoubleSpinBox()
        self.spin_weightdecay.setFont(font_normal)
        self.spin_weightdecay.setRange(0.0, 1.0)
        self.spin_weightdecay.setDecimals(5)
        self.spin_weightdecay.setSingleStep(0.0001)
        self.spin_weightdecay.setValue(0.0005)
        lr_layout.addWidget(lbl_weightdecay, row, 2)
        lr_layout.addWidget(self.spin_weightdecay, row, 3)

        row += 1
        lbl_warmup_epochs = QLabel("预热轮次:")
        lbl_warmup_epochs.setFont(font_bold)
        self.spin_warmup_epochs = QDoubleSpinBox()
        self.spin_warmup_epochs.setFont(font_normal)
        self.spin_warmup_epochs.setRange(0.0, 50.0)
        self.spin_warmup_epochs.setDecimals(1)
        self.spin_warmup_epochs.setSingleStep(0.5)
        self.spin_warmup_epochs.setValue(3.0)
        lr_layout.addWidget(lbl_warmup_epochs, row, 0)
        lr_layout.addWidget(self.spin_warmup_epochs, row, 1)

        lbl_warmup_bias_lr = QLabel("预热偏置学习率:")
        lbl_warmup_bias_lr.setFont(font_bold)
        self.spin_warmup_bias_lr = QDoubleSpinBox()
        self.spin_warmup_bias_lr.setFont(font_normal)
        self.spin_warmup_bias_lr.setRange(0.0, 0.1)
        self.spin_warmup_bias_lr.setDecimals(4)
        self.spin_warmup_bias_lr.setSingleStep(0.01)
        self.spin_warmup_bias_lr.setValue(0.0)
        lr_layout.addWidget(lbl_warmup_bias_lr, row, 2)
        lr_layout.addWidget(self.spin_warmup_bias_lr, row, 3)

        row += 1
        lbl_close_mosaic = QLabel("关闭mosaic轮次:")
        lbl_close_mosaic.setFont(font_bold)
        self.spin_close_mosaic = QSpinBox()
        self.spin_close_mosaic.setFont(font_normal)
        self.spin_close_mosaic.setRange(0, 100)
        self.spin_close_mosaic.setValue(10)
        lr_layout.addWidget(lbl_close_mosaic, row, 0)
        lr_layout.addWidget(self.spin_close_mosaic, row, 1)

        lbl_freeze = QLabel("冻结层数:")
        lbl_freeze.setFont(font_bold)
        self.spin_freeze = QSpinBox()
        self.spin_freeze.setFont(font_normal)
        self.spin_freeze.setRange(0, 50)
        self.spin_freeze.setValue(0)
        lr_layout.addWidget(lbl_freeze, row, 2)
        lr_layout.addWidget(self.spin_freeze, row, 3)

        lr_content = QWidget()
        lr_content.setLayout(lr_layout)
        lr_group.setContentWidget(lr_content)
        scroll_layout.addWidget(lr_group)

        # ==================== 损失函数权重 ====================
        loss_group = CollapsibleGroupBox("损失函数权重")
        loss_layout = QGridLayout()
        loss_layout.setSpacing(10)

        row = 0
        lbl_box = QLabel("框损失(box):")
        lbl_box.setFont(font_bold)
        self.spin_box = QDoubleSpinBox()
        self.spin_box.setFont(font_normal)
        self.spin_box.setRange(0.0, 50.0)
        self.spin_box.setDecimals(1)
        self.spin_box.setSingleStep(0.5)
        self.spin_box.setValue(7.5)
        loss_layout.addWidget(lbl_box, row, 0)
        loss_layout.addWidget(self.spin_box, row, 1)

        lbl_cls = QLabel("分类损失(cls):")
        lbl_cls.setFont(font_bold)
        self.spin_cls = QDoubleSpinBox()
        self.spin_cls.setFont(font_normal)
        self.spin_cls.setRange(0.0, 10.0)
        self.spin_cls.setDecimals(1)
        self.spin_cls.setSingleStep(0.1)
        self.spin_cls.setValue(0.5)
        loss_layout.addWidget(lbl_cls, row, 2)
        loss_layout.addWidget(self.spin_cls, row, 3)

        row += 1
        lbl_dfl = QLabel("DFL损失(dfl):")
        lbl_dfl.setFont(font_bold)
        self.spin_dfl = QDoubleSpinBox()
        self.spin_dfl.setFont(font_normal)
        self.spin_dfl.setRange(0.0, 10.0)
        self.spin_dfl.setDecimals(1)
        self.spin_dfl.setSingleStep(0.5)
        self.spin_dfl.setValue(1.5)
        loss_layout.addWidget(lbl_dfl, row, 0)
        loss_layout.addWidget(self.spin_dfl, row, 1)

        lbl_pose = QLabel("姿态损失(pose):")
        lbl_pose.setFont(font_bold)
        self.spin_pose = QDoubleSpinBox()
        self.spin_pose.setFont(font_normal)
        self.spin_pose.setRange(0.0, 50.0)
        self.spin_pose.setDecimals(1)
        self.spin_pose.setSingleStep(1.0)
        self.spin_pose.setValue(12.0)
        loss_layout.addWidget(lbl_pose, row, 2)
        loss_layout.addWidget(self.spin_pose, row, 3)

        row += 1
        lbl_kobj = QLabel("关键点损失(kobj):")
        lbl_kobj.setFont(font_bold)
        self.spin_kobj = QDoubleSpinBox()
        self.spin_kobj.setFont(font_normal)
        self.spin_kobj.setRange(0.0, 10.0)
        self.spin_kobj.setDecimals(1)
        self.spin_kobj.setSingleStep(0.1)
        self.spin_kobj.setValue(1.0)
        loss_layout.addWidget(lbl_kobj, row, 0)
        loss_layout.addWidget(self.spin_kobj, row, 1)

        lbl_dropout = QLabel("Dropout(分类):")
        lbl_dropout.setFont(font_bold)
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setFont(font_normal)
        self.spin_dropout.setRange(0.0, 1.0)
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setValue(0.0)
        loss_layout.addWidget(lbl_dropout, row, 2)
        loss_layout.addWidget(self.spin_dropout, row, 3)

        loss_content = QWidget()
        loss_content.setLayout(loss_layout)
        loss_group.setContentWidget(loss_content)
        scroll_layout.addWidget(loss_group)

        # ==================== 数据增强 ====================
        aug_group = CollapsibleGroupBox("数据增强")
        aug_layout = QGridLayout()
        aug_layout.setSpacing(10)

        row = 0
        lbl_hsv_h = QLabel("HSV-H:")
        lbl_hsv_h.setFont(font_bold)
        self.spin_hsv_h = QDoubleSpinBox()
        self.spin_hsv_h.setFont(font_normal)
        self.spin_hsv_h.setRange(0.0, 1.0)
        self.spin_hsv_h.setDecimals(3)
        self.spin_hsv_h.setSingleStep(0.005)
        self.spin_hsv_h.setValue(0.015)
        aug_layout.addWidget(lbl_hsv_h, row, 0)
        aug_layout.addWidget(self.spin_hsv_h, row, 1)

        lbl_hsv_s = QLabel("HSV-S:")
        lbl_hsv_s.setFont(font_bold)
        self.spin_hsv_s = QDoubleSpinBox()
        self.spin_hsv_s.setFont(font_normal)
        self.spin_hsv_s.setRange(0.0, 1.0)
        self.spin_hsv_s.setDecimals(2)
        self.spin_hsv_s.setSingleStep(0.1)
        self.spin_hsv_s.setValue(0.7)
        aug_layout.addWidget(lbl_hsv_s, row, 2)
        aug_layout.addWidget(self.spin_hsv_s, row, 3)

        row += 1
        lbl_hsv_v = QLabel("HSV-V:")
        lbl_hsv_v.setFont(font_bold)
        self.spin_hsv_v = QDoubleSpinBox()
        self.spin_hsv_v.setFont(font_normal)
        self.spin_hsv_v.setRange(0.0, 1.0)
        self.spin_hsv_v.setDecimals(2)
        self.spin_hsv_v.setSingleStep(0.1)
        self.spin_hsv_v.setValue(0.4)
        aug_layout.addWidget(lbl_hsv_v, row, 0)
        aug_layout.addWidget(self.spin_hsv_v, row, 1)

        lbl_degrees = QLabel("旋转(degrees):")
        lbl_degrees.setFont(font_bold)
        self.spin_degrees = QDoubleSpinBox()
        self.spin_degrees.setFont(font_normal)
        self.spin_degrees.setRange(0.0, 180.0)
        self.spin_degrees.setDecimals(1)
        self.spin_degrees.setSingleStep(1.0)
        self.spin_degrees.setValue(0.0)
        aug_layout.addWidget(lbl_degrees, row, 2)
        aug_layout.addWidget(self.spin_degrees, row, 3)

        row += 1
        lbl_translate = QLabel("平移(translate):")
        lbl_translate.setFont(font_bold)
        self.spin_translate = QDoubleSpinBox()
        self.spin_translate.setFont(font_normal)
        self.spin_translate.setRange(0.0, 1.0)
        self.spin_translate.setDecimals(2)
        self.spin_translate.setSingleStep(0.05)
        self.spin_translate.setValue(0.1)
        aug_layout.addWidget(lbl_translate, row, 0)
        aug_layout.addWidget(self.spin_translate, row, 1)

        lbl_scale = QLabel("缩放(scale):")
        lbl_scale.setFont(font_bold)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setFont(font_normal)
        self.spin_scale.setRange(0.0, 2.0)
        self.spin_scale.setDecimals(2)
        self.spin_scale.setSingleStep(0.1)
        self.spin_scale.setValue(0.5)
        aug_layout.addWidget(lbl_scale, row, 2)
        aug_layout.addWidget(self.spin_scale, row, 3)

        row += 1
        lbl_shear = QLabel("剪切(shear):")
        lbl_shear.setFont(font_bold)
        self.spin_shear = QDoubleSpinBox()
        self.spin_shear.setFont(font_normal)
        self.spin_shear.setRange(0.0, 30.0)
        self.spin_shear.setDecimals(1)
        self.spin_shear.setSingleStep(0.5)
        self.spin_shear.setValue(0.0)
        aug_layout.addWidget(lbl_shear, row, 0)
        aug_layout.addWidget(self.spin_shear, row, 1)

        lbl_fliplr = QLabel("左右翻转概率:")
        lbl_fliplr.setFont(font_bold)
        self.spin_fliplr = QDoubleSpinBox()
        self.spin_fliplr.setFont(font_normal)
        self.spin_fliplr.setRange(0.0, 1.0)
        self.spin_fliplr.setDecimals(2)
        self.spin_fliplr.setSingleStep(0.1)
        self.spin_fliplr.setValue(0.5)
        aug_layout.addWidget(lbl_fliplr, row, 2)
        aug_layout.addWidget(self.spin_fliplr, row, 3)

        row += 1
        lbl_mosaic = QLabel("mosaic概率:")
        lbl_mosaic.setFont(font_bold)
        self.spin_mosaic = QDoubleSpinBox()
        self.spin_mosaic.setFont(font_normal)
        self.spin_mosaic.setRange(0.0, 1.0)
        self.spin_mosaic.setDecimals(2)
        self.spin_mosaic.setSingleStep(0.1)
        self.spin_mosaic.setValue(1.0)
        aug_layout.addWidget(lbl_mosaic, row, 0)
        aug_layout.addWidget(self.spin_mosaic, row, 1)

        lbl_mixup = QLabel("mixup概率:")
        lbl_mixup.setFont(font_bold)
        self.spin_mixup = QDoubleSpinBox()
        self.spin_mixup.setFont(font_normal)
        self.spin_mixup.setRange(0.0, 1.0)
        self.spin_mixup.setDecimals(2)
        self.spin_mixup.setSingleStep(0.1)
        self.spin_mixup.setValue(0.0)
        aug_layout.addWidget(lbl_mixup, row, 2)
        aug_layout.addWidget(self.spin_mixup, row, 3)

        aug_content = QWidget()
        aug_content.setLayout(aug_layout)
        aug_group.setContentWidget(aug_content)
        scroll_layout.addWidget(aug_group)

        main_layout.addWidget(scroll, 1)

        # ==================== 日志输出 ====================
        log_label = QLabel("训练日志")
        log_label.setFont(font_bold)
        main_layout.addWidget(log_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setFont(font_mono)
        self.log_text.setMaximumHeight(220)
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        # ==================== 按钮行 ====================
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_start = QPushButton("开始训练")
        self.btn_start.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        main_layout.addLayout(btn_layout)

        # ==================== ONNX 导出 ====================
        export_group = QGroupBox("ONNX 导出")
        export_group.setFont(font_bold)
        main_layout.addWidget(export_group)

        export_inner = QWidget()
        export_layout = QGridLayout(export_inner)
        export_layout.setContentsMargins(8, 8, 8, 8)

        self.edt_export_pt = QLineEdit()
        self.edt_export_pt.setPlaceholderText("选择 .pt 权重文件...")
        self.edt_export_pt.setFont(font_normal)
        self.btn_export_pt = QPushButton("浏览")
        self.btn_export_pt.setFont(font_normal)
        export_layout.addWidget(QLabel("PT 模型:"), 0, 0)
        export_layout.addWidget(self.edt_export_pt, 0, 1)
        export_layout.addWidget(self.btn_export_pt, 0, 2)

        self.edt_export_out = QLineEdit()
        self.edt_export_out.setPlaceholderText("ONNX 输出目录（留空则与 PT 同目录）")
        self.edt_export_out.setFont(font_normal)
        self.btn_export_out = QPushButton("浏览")
        self.btn_export_out.setFont(font_normal)
        export_layout.addWidget(QLabel("输出目录:"), 1, 0)
        export_layout.addWidget(self.edt_export_out, 1, 1)
        export_layout.addWidget(self.btn_export_out, 1, 2)

        self.cmb_export_device = QComboBox()
        self.cmb_export_device.setFont(font_normal)
        self.cmb_export_device.addItems(["cuda", "cpu"])
        self.cmb_export_device.setCurrentText("cuda")
        self.spin_export_imgsz = QSpinBox()
        self.spin_export_imgsz.setFont(font_normal)
        self.spin_export_imgsz.setRange(32, 1280)
        self.spin_export_imgsz.setSingleStep(32)
        self.spin_export_imgsz.setValue(640)

        self.btn_export_onnx = QPushButton("导出 ONNX")
        self.btn_export_onnx.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("设备:"))
        row2.addWidget(self.cmb_export_device)
        row2.addSpacing(12)
        row2.addWidget(QLabel("输入尺寸:"))
        row2.addWidget(self.spin_export_imgsz)
        row2.addSpacing(12)
        self.chk_export_half = QCheckBox("FP16 精度")
        self.chk_export_half.setFont(font_normal)
        self.chk_export_half.setChecked(False)
        row2.addWidget(self.chk_export_half)
        row2.addStretch()
        row2.addWidget(self.btn_export_onnx)
        export_layout.addWidget(QLabel("选项:"), 2, 0)
        export_layout.addLayout(row2, 2, 1, 1, 2)

        export_group.setLayout(QVBoxLayout())
        export_group.layout().addWidget(export_inner)

    def _connect_signals(self):
        self.btn_model.clicked.connect(self._on_select_model)
        self.btn_data.clicked.connect(self._on_select_data)
        self.btn_new_config.clicked.connect(self._on_new_config)
        self.btn_save_config.clicked.connect(self._on_save_config)
        self.btn_load_config.clicked.connect(self._on_load_config)
        self.btn_start.clicked.connect(self._on_start_train)
        self.btn_stop.clicked.connect(self._on_stop_train)

        self.btn_export_pt.clicked.connect(self._on_select_export_pt)
        self.btn_export_out.clicked.connect(self._on_select_export_out)
        self.btn_export_onnx.clicked.connect(self._on_export_onnx)

    def _get_default_dir(self, category="origin_data"):
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
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", default_dir,
            "模型文件 (*.pt *.yaml);;所有文件 (*)"
        )
        if path:
            self.model_path = path
            self.edt_model.setText(path)

    def _on_select_data(self):
        default_dir = self._get_default_dir("origin_data")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集配置文件", default_dir,
            "YAML文件 (*.yaml *.yml);;所有文件 (*)"
        )
        if path:
            self.data_yaml_path = path
            self.edt_data.setText(path)

    def _refresh_config_list(self):
        self.cmb_config.clear()
        if os.path.isdir(self.config_dir):
            files = sorted(glob.glob(os.path.join(self.config_dir, "*.yaml")))
            for f in files:
                self.cmb_config.addItem(os.path.basename(f))

    def _on_new_config(self):
        name, ok = QFileDialog.getSaveFileName(
            self, "新建配置文件", self.config_dir,
            "YAML文件 (*.yaml)"
        )
        if ok and name:
            if not name.endswith(".yaml"):
                name += ".yaml"
            self._save_config_to_file(name)
            self._refresh_config_list()
            idx = self.cmb_config.findText(os.path.basename(name))
            if idx >= 0:
                self.cmb_config.setCurrentIndex(idx)
            self._append_log(f"[配置] 已创建新配置: {name}")

    def _on_save_config(self):
        name = self.cmb_config.currentText()
        if not name:
            self._append_log("[配置] 请先选择或新建一个配置文件")
            return
        path = os.path.join(self.config_dir, name)
        self._save_config_to_file(path)
        self._append_log(f"[配置] 已保存: {path}")

    def _save_config_to_file(self, path):
        params = self._collect_params()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(params, f, allow_unicode=True, sort_keys=False)

    def _on_load_config(self):
        name = self.cmb_config.currentText()
        if not name:
            self._append_log("[配置] 请先选择一个配置文件")
            return
        path = os.path.join(self.config_dir, name)
        if not os.path.exists(path):
            self._append_log(f"[配置] 配置文件不存在: {path}")
            return
        self._load_config_from_file(path)
        self._append_log(f"[配置] 已加载: {path}")

    def _load_config_from_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                import yaml
                params = yaml.safe_load(f)
            if not params:
                return
            self._apply_params(params)
            self._append_log(f"[配置] 已应用配置: {path}")
        except Exception as e:
            self._append_log(f"[配置] 加载失败: {e}")

    def _collect_params(self):
        params = {}
        if self.edt_model.text():
            params["model"] = self.edt_model.text()
        if self.edt_data.text():
            params["data"] = self.edt_data.text()
        params["epochs"] = self.spin_epochs.value()
        params["patience"] = self.spin_patience.value()
        params["batch"] = self.spin_batch.value()
        params["imgsz"] = self.spin_imgsz.value()
        params["device"] = self.cmb_device.currentText()
        params["workers"] = self.spin_workers.value()
        params["optimizer"] = self.cmb_optimizer.currentText()
        params["project"] = self.edt_project.text() or "runs/train"
        params["name"] = self.edt_name.text() or "exp"
        params["exist_ok"] = self.chk_existok.isChecked()
        params["pretrained"] = self.chk_pretrained.isChecked()
        params["amp"] = self.chk_amp.isChecked()
        params["cos_lr"] = self.chk_coslr.isChecked()
        params["deterministic"] = self.chk_deterministic.isChecked()
        params["single_cls"] = self.chk_singlecls.isChecked()
        params["rect"] = self.chk_rect.isChecked()
        params["val"] = self.chk_val.isChecked()
        params["plots"] = self.chk_plots.isChecked()
        params["resume"] = self.chk_resume.isChecked()
        params["verbose"] = self.chk_verbose.isChecked()
        params["lr0"] = self.spin_lr0.value()
        params["lrf"] = self.spin_lrf.value()
        params["momentum"] = self.spin_momentum.value()
        params["weight_decay"] = self.spin_weightdecay.value()
        params["warmup_epochs"] = self.spin_warmup_epochs.value()
        params["warmup_bias_lr"] = self.spin_warmup_bias_lr.value()
        params["close_mosaic"] = self.spin_close_mosaic.value()
        params["freeze"] = self.spin_freeze.value()
        params["box"] = self.spin_box.value()
        params["cls"] = self.spin_cls.value()
        params["dfl"] = self.spin_dfl.value()
        params["pose"] = self.spin_pose.value()
        params["kobj"] = self.spin_kobj.value()
        params["dropout"] = self.spin_dropout.value()
        params["hsv_h"] = self.spin_hsv_h.value()
        params["hsv_s"] = self.spin_hsv_s.value()
        params["hsv_v"] = self.spin_hsv_v.value()
        params["degrees"] = self.spin_degrees.value()
        params["translate"] = self.spin_translate.value()
        params["scale"] = self.spin_scale.value()
        params["shear"] = self.spin_shear.value()
        params["fliplr"] = self.spin_fliplr.value()
        params["mosaic"] = self.spin_mosaic.value()
        params["mixup"] = self.spin_mixup.value()
        return params

    def _apply_params(self, params):
        def safe_set_spin(spin, value):
            if value is None:
                return
            try:
                spin.setValue(float(value))
            except Exception:
                pass

        def safe_set_check(chk, value):
            if value is None:
                return
            chk.setChecked(bool(value))

        def safe_set_combo(combo, value):
            if value is None:
                return
            idx = combo.findText(str(value))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        def safe_set_text(edt, value):
            if value is None:
                return
            edt.setText(str(value))

        safe_set_text(self.edt_model, params.get("model"))
        safe_set_text(self.edt_data, params.get("data"))
        safe_set_spin(self.spin_epochs, params.get("epochs"))
        safe_set_spin(self.spin_patience, params.get("patience"))
        safe_set_spin(self.spin_batch, params.get("batch"))
        safe_set_spin(self.spin_imgsz, params.get("imgsz"))
        safe_set_combo(self.cmb_device, params.get("device"))
        safe_set_spin(self.spin_workers, params.get("workers"))
        safe_set_combo(self.cmb_optimizer, params.get("optimizer"))
        safe_set_text(self.edt_project, params.get("project"))
        safe_set_text(self.edt_name, params.get("name"))
        safe_set_check(self.chk_existok, params.get("exist_ok"))
        safe_set_check(self.chk_pretrained, params.get("pretrained"))
        safe_set_check(self.chk_amp, params.get("amp"))
        safe_set_check(self.chk_coslr, params.get("cos_lr"))
        safe_set_check(self.chk_deterministic, params.get("deterministic"))
        safe_set_check(self.chk_singlecls, params.get("single_cls"))
        safe_set_check(self.chk_rect, params.get("rect"))
        safe_set_check(self.chk_val, params.get("val"))
        safe_set_check(self.chk_plots, params.get("plots"))
        safe_set_check(self.chk_resume, params.get("resume"))
        safe_set_check(self.chk_verbose, params.get("verbose"))
        safe_set_spin(self.spin_lr0, params.get("lr0"))
        safe_set_spin(self.spin_lrf, params.get("lrf"))
        safe_set_spin(self.spin_momentum, params.get("momentum"))
        safe_set_spin(self.spin_weightdecay, params.get("weight_decay"))
        safe_set_spin(self.spin_warmup_epochs, params.get("warmup_epochs"))
        safe_set_spin(self.spin_warmup_bias_lr, params.get("warmup_bias_lr"))
        safe_set_spin(self.spin_close_mosaic, params.get("close_mosaic"))
        safe_set_spin(self.spin_freeze, params.get("freeze"))
        safe_set_spin(self.spin_box, params.get("box"))
        safe_set_spin(self.spin_cls, params.get("cls"))
        safe_set_spin(self.spin_dfl, params.get("dfl"))
        safe_set_spin(self.spin_pose, params.get("pose"))
        safe_set_spin(self.spin_kobj, params.get("kobj"))
        safe_set_spin(self.spin_dropout, params.get("dropout"))
        safe_set_spin(self.spin_hsv_h, params.get("hsv_h"))
        safe_set_spin(self.spin_hsv_s, params.get("hsv_s"))
        safe_set_spin(self.spin_hsv_v, params.get("hsv_v"))
        safe_set_spin(self.spin_degrees, params.get("degrees"))
        safe_set_spin(self.spin_translate, params.get("translate"))
        safe_set_spin(self.spin_scale, params.get("scale"))
        safe_set_spin(self.spin_shear, params.get("shear"))
        safe_set_spin(self.spin_fliplr, params.get("fliplr"))
        safe_set_spin(self.spin_mosaic, params.get("mosaic"))
        safe_set_spin(self.spin_mixup, params.get("mixup"))

    def _append_log(self, msg):
        self.log_text.appendPlainText(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _reset_status(self):
        self.status_label.setText("等待开始训练")
        self.status_label.setProperty("role", "muted")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.epoch_label.setText("Epoch: -/-")
        self.metrics_label.setText("box: -  cls: -  dfl: -")
        self.map_label.setText("mAP50: -  mAP50-95: -")
        self.progress_bar.setValue(0)

    def _on_start_train(self):
        if self._train_worker and self._train_worker.isRunning():
            self._append_log("[提示] 训练已在进行中")
            return
        if not self.edt_model.text():
            self._append_log("[错误] 请先选择模型权重文件")
            return
        if not self.edt_data.text():
            self._append_log("[错误] 请先选择数据集配置文件")
            return

        params = self._collect_params()
        self.log_text.clear()
        self._reset_status()

        self._append_log(f"[训练] 开始训练...")
        self._append_log(f"[训练] 模型: {self.edt_model.text()}")
        self._append_log(f"[训练] 数据: {self.edt_data.text()}")
        self._append_log("=" * 50)

        self.status_label.setText("训练进行中...")
        self.status_label.setProperty("role", "accent")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._train_worker = YoloTrainWorker(self.edt_model.text(), params)
        self._train_worker.training_started.connect(self.worker_training_started)
        self._train_worker.log_signal.connect(self._append_log)
        self._train_worker.metrics_signal.connect(self._on_metrics_update)
        self._train_worker.finished_signal.connect(self._on_train_finished)
        self._train_worker.start()

    def _on_metrics_update(self, metrics):
        if 'epoch' in metrics and 'total_epochs' in metrics:
            self.epoch_label.setText(f"Epoch: {metrics['epoch']}/{metrics['total_epochs']}")
            pct = int(metrics['epoch'] / metrics['total_epochs'] * 100)
            self.progress_bar.setValue(pct)

        loss_parts = []
        if 'box_loss' in metrics:
            loss_parts.append(f"box: {metrics['box_loss']:.3f}")
        if 'cls_loss' in metrics:
            loss_parts.append(f"cls: {metrics['cls_loss']:.3f}")
        if 'dfl_loss' in metrics:
            loss_parts.append(f"dfl: {metrics['dfl_loss']:.3f}")
        if loss_parts:
            self.metrics_label.setText("  ".join(loss_parts))

        map_parts = []
        if 'mAP50' in metrics:
            map_parts.append(f"mAP50: {metrics['mAP50']:.3f}")
        if 'mAP50-95' in metrics:
            map_parts.append(f"mAP50-95: {metrics['mAP50-95']:.3f}")
        if map_parts:
            self.map_label.setText("  ".join(map_parts))

    def _on_stop_train(self):
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.stop()
            self._train_worker.wait(5000)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("训练已停止")
        self.status_label.setProperty("role", "muted")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self._append_log("[训练] 用户停止了训练")

    def _on_select_export_pt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PT 文件", self._get_default_dir("yolo"),
            "权重文件 (*.pt);;所有文件 (*)"
        )
        if path:
            self.edt_export_pt.setText(path)

    def _on_select_export_out(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self._get_default_dir("yolo")
        )
        if path:
            self.edt_export_out.setText(path)

    def _on_export_onnx(self):
        pt = self.edt_export_pt.text().strip()
        if not pt or not os.path.isfile(pt):
            self._append_log("[导出] 请先选择有效的 .pt 文件")
            return
        out_dir = self.edt_export_out.text().strip()
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        device = self.cmb_export_device.currentText()
        imgsz = self.spin_export_imgsz.value()
        self.btn_export_onnx.setEnabled(False)
        self._append_log(f"[导出] 启动导出任务...")
        half = self.chk_export_half.isChecked()
        self._export_worker = OnnxExportWorker(pt, out_dir if out_dir else "", imgsz, device, half=half)
        self._export_worker.log_signal.connect(self._append_log)
        self._export_worker.finished_signal.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_finished(self, success, msg):
        self.btn_export_onnx.setEnabled(True)
        if success:
            self._append_log(f"[导出] 成功: {msg}")
            QMessageBox.information(self, "ONNX 导出", f"导出成功\n{msg}")
        else:
            self._append_log(f"[导出] 失败: {msg}")
            QMessageBox.warning(self, "ONNX 导出", f"导出失败\n{msg}")

    def _on_train_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(f"[训练] {msg}")

        if success:
            self.status_label.setText("训练完成")
            self.status_label.setProperty("role", "status_bold_success")
            self.progress_bar.setValue(100)
        else:
            if "停止" not in msg:
                self.status_label.setText("训练失败")
                self.status_label.setProperty("role", "status_bold_error")

        self.worker_training_finished.emit()

    def closeEvent(self, event):
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.stop()
            self._train_worker.wait(5000)
        if hasattr(self, '_export_worker') and self._export_worker.isRunning():
            self._export_worker.wait(5000)
        event.accept()


if __name__ == '__main__':
    import yaml
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = YoloTrainWindow()
    window.show()
    sys.exit(app.exec())
