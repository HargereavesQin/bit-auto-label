# -*- coding: utf-8 -*-
"""
统计对话框 — RM 主题 (走 style.qss).
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QPushButton, QTabWidget, QWidget, QGroupBox,
                               QGridLayout, QSpinBox)
from PySide6.QtCore import Qt

from .icons import Icon, Icons, C
from .dialog_utils import build_dialog_header


class StatisticsDialog(QDialog):
    """统计信息展示对话框"""

    def __init__(self, report, current_image_report=None, parent=None,
                 current_row=0, total_count=0, current_filename="", main_window=None):
        super().__init__(parent)
        self.report = report
        self.current_image_report = current_image_report
        self.current_row = current_row
        self.total_count = total_count
        self.current_filename = current_filename
        self.main_window = main_window
        self.setWindowTitle("数据集统计报告")
        self.setWindowIcon(Icon.icon(Icons.CHART, size=24, color=C.AMBER))
        self.setMinimumSize(700, 520)

        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(build_dialog_header(
            self, Icons.CHART,
            "数据集统计报告", "DATASET STATISTICS  /  标注概览"))

        body = QVBoxLayout()
        body.setContentsMargins(16, 8, 16, 8)
        body.setSpacing(10)
        outer.addLayout(body, 1)

        # 标签页
        self.tabs = QTabWidget()
        body.addWidget(self.tabs, 1)

        # ---- 标签页1: 概览 ----
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        overview_layout.setContentsMargins(4, 4, 4, 4)

        overview_group = QGroupBox("数据集概览  /  OVERVIEW")
        overview_group.setProperty("accent", "amber")
        grid = QGridLayout(overview_group)
        grid.setContentsMargins(10, 14, 10, 10)
        grid.setSpacing(10)
        grid.setColumnStretch(1, 1)

        report = self.report
        format_stats = report.get("format_stats", {})
        items = [
            ("总图片数",     str(report["total_images"]),                          C.AMBER),
            ("已标注图片",   str(report["annotated_images"]),                       C.SUCCESS),
            ("未标注图片",   str(report["unannotated_images"]),                     C.DANGER if report["unannotated_images"] > 0 else C.TEXT_MUTED),
            ("总标注数",     str(report["total_annotations"]),                      C.TEXT_MUTED),
            ("类别总数",     str(len(report["class_stats"])),                       C.AMBER),
            ("JSON 标注",    str(format_stats.get("json", 0)),                      C.TEXT_MUTED),
            ("YOLO 标注",    str(format_stats.get("yolo", 0)),                      C.AMBER),
            ("XML 标注",     str(format_stats.get("xml", 0)),                       C.SUCCESS),
        ]

        for i, (label, value, color) in enumerate(items):
            lbl_title = QLabel(label)
            lbl_title.setProperty("role", "caption")
            grid.addWidget(lbl_title, i, 0)
            lbl_value = QLabel(value)
            lbl_value.setProperty("role", "stat_value")
            # 动态颜色部分保留内联 (文字颜色按统计项变化)
            if color != C.AMBER:  # 默认琥珀色已在 QSS 中, 只有非默认时才覆盖
                lbl_value.setStyleSheet(f"color: {color};")
            lbl_value.setAlignment(Qt.AlignRight)
            grid.addWidget(lbl_value, i, 1)

        overview_layout.addWidget(overview_group)
        overview_layout.addStretch()

        # ---- 标签页2: 类别统计 ----
        class_widget = QWidget()
        class_layout = QVBoxLayout(class_widget)
        class_layout.setContentsMargins(4, 4, 4, 4)

        class_group = QGroupBox("各类别标注统计  /  CLASSES")
        class_group.setProperty("accent", "blue")
        class_inner_layout = QVBoxLayout(class_group)
        class_inner_layout.setContentsMargins(10, 14, 10, 10)

        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.setHorizontalHeaderLabels(["类别名称", "标注次数", "包含图片数"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.class_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.class_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.class_table.setAlternatingRowColors(True)
        self.class_table.setProperty("role", "mono")

        class_stats = report.get("class_stats", {})
        if class_stats:
            sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]["count"], reverse=True)
            self.class_table.setRowCount(len(sorted_classes))
            for i, (label, data) in enumerate(sorted_classes):
                self.class_table.setItem(i, 0, QTableWidgetItem(label))
                self.class_table.setItem(i, 1, QTableWidgetItem(str(data["count"])))
                self.class_table.setItem(i, 2, QTableWidgetItem(str(data["images"])))
        else:
            self.class_table.setRowCount(1)
            self.class_table.setItem(0, 0, QTableWidgetItem("暂无数据"))
            self.class_table.setItem(0, 1, QTableWidgetItem("-"))
            self.class_table.setItem(0, 2, QTableWidgetItem("-"))

        class_inner_layout.addWidget(self.class_table)
        class_layout.addWidget(class_group)

        # ---- 标签页3: 当前图片统计 ----
        if self.current_image_report:
            current_widget = QWidget()
            current_layout = QVBoxLayout(current_widget)
            current_layout.setContentsMargins(4, 4, 4, 4)

            nav_group = QGroupBox("图片导航  /  NAVIGATE")
            nav_group.setProperty("accent", "blue")
            nav_layout = QHBoxLayout(nav_group)
            nav_layout.setContentsMargins(10, 14, 10, 10)
            nav_layout.setSpacing(8)

            pos_label = QLabel(f"当前: 第 <b style='color:#C9A36B;'>{self.current_row + 1}</b>/{self.total_count} 张")
            pos_label.setTextFormat(Qt.RichText)
            pos_label.setProperty("role", "caption")
            nav_layout.addWidget(pos_label)
            nav_layout.addStretch()

            if self.current_filename:
                fname_label = QLabel(self.current_filename)
                fname_label.setProperty("role", "caption")
                fname_label.setWordWrap(True)
                nav_layout.addWidget(fname_label)

            nav_layout.addSpacing(16)

            nav_layout.addWidget(QLabel("跳转到:"))
            self.jump_spin = QSpinBox()
            self.jump_spin.setRange(1, max(1, self.total_count))
            self.jump_spin.setValue(self.current_row + 1)
            self.jump_spin.setProperty("role", "mono")
            self.jump_spin.setMinimumWidth(70)
            nav_layout.addWidget(self.jump_spin)
            nav_layout.addWidget(QLabel("张"))

            jump_btn = QPushButton("跳转")
            jump_btn.setCursor(Qt.PointingHandCursor)
            jump_btn.setProperty("role", "primary")
            jump_btn.clicked.connect(self._on_jump)
            nav_layout.addWidget(jump_btn)

            current_layout.addWidget(nav_group)

            current_group = QGroupBox("当前图片标注  /  CURRENT")
            current_group.setProperty("accent", "amber")
            current_inner_layout = QVBoxLayout(current_group)
            current_inner_layout.setContentsMargins(10, 14, 10, 10)

            total_label = QLabel(f"当前图片标注总数: <b style='color:#C9A36B;'>{self.current_image_report['total']}</b>")
            total_label.setTextFormat(Qt.RichText)
            total_label.setProperty("role", "caption")
            current_inner_layout.addWidget(total_label)

            self.current_class_table = QTableWidget()
            self.current_class_table.setColumnCount(2)
            self.current_class_table.setHorizontalHeaderLabels(["类别名称", "数量"])
            self.current_class_table.horizontalHeader().setStretchLastSection(True)
            self.current_class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.current_class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.current_class_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.current_class_table.setAlternatingRowColors(True)
            self.current_class_table.setProperty("role", "mono")

            cur_stats = self.current_image_report.get("class_stats", {})
            if cur_stats:
                sorted_cur = sorted(cur_stats.items(), key=lambda x: x[1], reverse=True)
                self.current_class_table.setRowCount(len(sorted_cur))
                for i, (label, count) in enumerate(sorted_cur):
                    self.current_class_table.setItem(i, 0, QTableWidgetItem(label))
                    self.current_class_table.setItem(i, 1, QTableWidgetItem(str(count)))
            else:
                self.current_class_table.setRowCount(1)
                self.current_class_table.setItem(0, 0, QTableWidgetItem("暂无标注"))
                self.current_class_table.setItem(0, 1, QTableWidgetItem("0"))

            current_inner_layout.addWidget(self.current_class_table)
            current_layout.addWidget(current_group)

            self.tabs.addTab(current_widget, "当前图片")

        self.tabs.addTab(overview_widget, "概览")
        self.tabs.addTab(class_widget, "类别统计")

        # 关闭按钮
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(16, 8, 16, 14)
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setProperty("role", "ghost")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _on_jump(self):
        target_row = self.jump_spin.value() - 1
        if self.main_window and 0 <= target_row < self.total_count:
            self.main_window.listFiles.setCurrentRow(target_row)
            self.accept()
