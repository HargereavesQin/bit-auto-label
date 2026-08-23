# -*- coding: utf-8 -*-
"""
数据集融合对话框 - 将多个数据集（YOLO已导出/LabelMe原始/Pascal VOC原始）合并为一个YOLO数据集
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton,
    QGroupBox, QTextEdit, QComboBox, QLineEdit,
    QRadioButton, QButtonGroup, QSpinBox, QMessageBox,
    QFileDialog, QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.dataset_fusion import DatasetFusionWorker, detect_format, merge_class_maps


class DatasetFusionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据集融合")
        self.setMinimumSize(800, 600)
        self.resize(900, 680)
        self.sources = []
        self._init_ui()
        self._load_config()

    def _init_ui(self):
        from .icons import Icon, Icons, C
        from .dialog_utils import build_dialog_header

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.setWindowIcon(Icon.icon(Icons.FUSION, size=24, color=C.AMBER))
        outer.addWidget(build_dialog_header(
            self, Icons.FUSION,
            "数据集融合", "DATASET FUSION  /  合并多格式数据集"))

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 8, 16, 8)
        main_layout.setSpacing(10)
        outer.addLayout(main_layout, 1)

        content_layout = QGridLayout()

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("输出 YOLO 格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["检测 (detect)", "分割 (segment)", "旋转矩形 (OBB)", "姿态 (pose)"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        content_layout.addLayout(format_layout, 0, 0, 1, 2)

        src_group = QGroupBox("源文件夹列表")
        src_layout = QVBoxLayout(src_group)

        self.src_list = QListWidget()
        self.src_list.setMinimumHeight(140)
        src_layout.addWidget(self.src_list)

        src_btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("+ 添加源文件夹")
        self.add_btn.setObjectName("addBtn")
        self.add_btn.clicked.connect(self._add_source)
        self.remove_btn = QPushButton("删除选中")
        self.remove_btn.setObjectName("removeBtn")
        self.remove_btn.clicked.connect(self._remove_source)
        self.clear_btn = QPushButton("清空全部")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self._clear_sources)
        src_btn_layout.addWidget(self.add_btn)
        src_btn_layout.addWidget(self.remove_btn)
        src_btn_layout.addWidget(self.clear_btn)
        src_btn_layout.addStretch()
        src_layout.addLayout(src_btn_layout)

        content_layout.addWidget(src_group, 1, 0)

        out_group = QGroupBox("输出目录")
        out_layout = QVBoxLayout(out_group)
        out_path_layout = QHBoxLayout()
        self.out_input = QLineEdit()
        self.out_input.setPlaceholderText("请选择输出目录...")
        self.out_btn = QPushButton("选择")
        self.out_btn.clicked.connect(lambda: self._select_dir(self.out_input))
        out_path_layout.addWidget(self.out_input)
        out_path_layout.addWidget(self.out_btn)
        out_layout.addLayout(out_path_layout)
        content_layout.addWidget(out_group, 1, 1)

        split_group = QGroupBox("划分策略")
        split_layout = QVBoxLayout(split_group)
        split_radio_layout = QHBoxLayout()
        self.split_group = QButtonGroup()
        self.radio_shuffle = QRadioButton("重新打乱划分")
        self.radio_shuffle.setChecked(True)
        self.radio_keep = QRadioButton("保持原划分")
        self.split_group.addButton(self.radio_shuffle, 0)
        self.split_group.addButton(self.radio_keep, 1)
        split_radio_layout.addWidget(self.radio_shuffle)
        split_radio_layout.addWidget(self.radio_keep)
        split_radio_layout.addStretch()
        split_layout.addLayout(split_radio_layout)

        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("train:"))
        self.train_spin = QSpinBox()
        self.train_spin.setRange(1, 99)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix(" %")
        ratio_layout.addWidget(self.train_spin)
        ratio_layout.addWidget(QLabel("val:"))
        self.val_spin = QSpinBox()
        self.val_spin.setRange(1, 99)
        self.val_spin.setValue(20)
        self.val_spin.setSuffix(" %")
        ratio_layout.addWidget(self.val_spin)
        ratio_layout.addWidget(QLabel("test:"))
        self.test_spin = QSpinBox()
        self.test_spin.setRange(1, 99)
        self.test_spin.setValue(10)
        self.test_spin.setSuffix(" %")
        ratio_layout.addWidget(self.test_spin)
        ratio_layout.addStretch()
        self.radio_shuffle.toggled.connect(lambda checked: ratio_layout.setEnabled(checked))
        split_layout.addLayout(ratio_layout)

        content_layout.addWidget(split_group, 2, 0, 1, 2)

        preview_group = QGroupBox("融合预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("请添加源文件夹以查看预览...")
        self.preview_label.setProperty("role", "caption")
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        content_layout.addWidget(preview_group, 3, 0, 1, 2)

        main_layout.addLayout(content_layout)

        console_label = QLabel("处理日志")
        console_label.setProperty("role", "accent")
        main_layout.addWidget(console_label)

        self.console = QTextEdit()
        self.console.setMaximumHeight(160)
        self.console.setReadOnly(True)
        main_layout.addWidget(self.console)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_btn = QPushButton("开始融合")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_fusion)
        self.start_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        main_layout.addLayout(btn_layout)

    def _load_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig().get_tool_config("dataset_fusion")
        idx = int(cfg.get("format_index", 0))
        if 0 <= idx < self.format_combo.count():
            self.format_combo.setCurrentIndex(idx)
        self.out_input.setText(cfg.get("output_dir", ""))
        if cfg.get("split_strategy") == "keep":
            self.radio_keep.setChecked(True)
        else:
            self.radio_shuffle.setChecked(True)
        self.train_spin.setValue(int(cfg.get("train_ratio", 70)))
        self.val_spin.setValue(int(cfg.get("val_ratio", 20)))
        self.test_spin.setValue(int(cfg.get("test_ratio", 10)))

    def _save_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig()
        cfg.set("dataset_fusion", "format_index", self.format_combo.currentIndex())
        cfg.set("dataset_fusion", "output_dir", self.out_input.text())
        cfg.set("dataset_fusion", "split_strategy", "keep" if self.radio_keep.isChecked() else "shuffle")
        cfg.set("dataset_fusion", "train_ratio", self.train_spin.value())
        cfg.set("dataset_fusion", "val_ratio", self.val_spin.value())
        cfg.set("dataset_fusion", "test_ratio", self.test_spin.value())
        cfg.save()

    def closeEvent(self, event):
        self._save_config()
        event.accept()

    def _get_default_dir(self, category="origin_data"):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    def _select_dir(self, line_edit):
        default_dir = self._get_default_dir("origin_data")
        d = QFileDialog.getExistingDirectory(self, "选择目录", default_dir)
        if d:
            line_edit.setText(d)

    def _add_source(self):
        default_dir = self._get_default_dir("origin_data")
        d = QFileDialog.getExistingDirectory(self, "选择源文件夹", default_dir)
        if not d:
            return
        if d in self.sources:
            QMessageBox.warning(self, "提示", "该文件夹已添加！")
            return
        info, label = detect_format(d)
        if info is None:
            QMessageBox.warning(self, "无法识别", f"无法识别该目录的格式:\n{d}\n\n{label}")
            return
        info['path'] = d
        info['label'] = label
        self.sources.append(info)

        img_count = info.get('img_count', 0)
        class_count = len(info.get('class_map', {}))
        class_names = list(info.get('class_map', {}).keys())[:5]
        classes_str = ', '.join(class_names)
        if len(info.get('class_map', {})) > 5:
            classes_str += '...'

        display = f"[{len(self.sources)}] {os.path.basename(d)}"
        display += f"\n    格式: {label}"
        display += f" | 图片: {img_count}张"
        display += f" | 类别: {class_count}个 ({classes_str})"

        self.src_list.addItem(display)
        self._update_preview()

    def _remove_source(self):
        row = self.src_list.currentRow()
        if row >= 0:
            self.src_list.takeItem(row)
            self.sources.pop(row)
            self._update_preview()

    def _clear_sources(self):
        self.src_list.clear()
        self.sources = []
        self._update_preview()

    def _update_preview(self):
        if not self.sources:
            self.preview_label.setText("请添加源文件夹以查看预览...")
            self.preview_label.setProperty("role", "caption")
            self.start_btn.setEnabled(False)
            return

        total_imgs = sum(s.get('img_count', 0) for s in self.sources)
        all_classes = [s.get('class_map', {}) for s in self.sources]
        merged = {}
        next_id = 0
        for cm in all_classes:
            for name, old_id in cm.items():
                if name not in merged:
                    merged[name] = next_id
                    next_id += 1

        class_list = sorted(merged.keys(), key=lambda k: merged[k])
        classes_str = ', '.join(class_list[:8])
        if len(class_list) > 8:
            classes_str += f' ... (共{len(class_list)}类)'

        preview_text = f"总图片: <b>{total_imgs}</b> 张  |  合并后类别: <b>{len(merged)}</b> 个"
        preview_text += f"<br>类别列表: {classes_str}"

        subtypes = [s.get('subtype', '') or s.get('type', '') for s in self.sources]
        yolo_types = [s for s in subtypes if s]
        if yolo_types:
            preview_text += f"<br>YOLO子格式: {', '.join(set(yolo_types))}"

        self.preview_label.setText(preview_text)
        self.preview_label.setProperty("role", "accent")
        self.start_btn.setEnabled(True)

    def _append_log(self, msg):
        self.console.append(msg)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _start_fusion(self):
        if not self.sources:
            QMessageBox.warning(self, "提示", "请先添加源文件夹！")
            return

        out_dir = self.out_input.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请选择输出目录！")
            return

        format_map = {0: 'detect', 1: 'seg', 2: 'obb', 3: 'pose'}
        output_format = format_map[self.format_combo.currentIndex()]

        split_policy = 'shuffle' if self.radio_shuffle.isChecked() else 'keep'
        ratios = (self.train_spin.value() / 100.0,
                  self.val_spin.value() / 100.0,
                  self.test_spin.value() / 100.0)

        self.console.clear()
        self.start_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        src_paths = [s['path'] for s in self.sources]

        self.worker = DatasetFusionWorker(src_paths, out_dir, output_format, split_policy, ratios)
        self.worker.log_signal.connect(self._append_log)
        self.worker.finish_signal.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, success, msg):
        self.start_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "融合完成", msg)
        else:
            QMessageBox.warning(self, "融合失败", msg)
