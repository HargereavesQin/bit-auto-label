# -*- coding: utf-8 -*-
"""
批量合并类别对话框 — RM 主题.
"""
import os
import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QCheckBox,
                               QComboBox, QPushButton, QGroupBox, QTextEdit,
                               QFormLayout, QMessageBox)
from PySide6.QtCore import Qt

from .icons import Icon, Icons, C
from .dialog_utils import build_dialog_header


class MergeClassDialog(QDialog):
    def __init__(self, class_list, pose_config, parent=None):
        super().__init__(parent)
        self.class_list = class_list
        self.pose_config = pose_config
        self.setWindowTitle("批量合并类别")
        self.setWindowIcon(Icon.icon(Icons.MERGE, size=24, color=C.AMBER))
        self.setMinimumSize(620, 480)
        self._init_ui()

    def _get_class_info(self, cls_name):
        if cls_name in self.pose_config.categories:
            cat = self.pose_config.categories[cls_name]
            kpt_count = cat.get("kpt_count", 0)
            return f"pose, {kpt_count} kpts"
        return "普通类别"

    def _get_kpt_count(self, cls_name):
        if cls_name in self.pose_config.categories:
            return self.pose_config.categories[cls_name].get("kpt_count", 0)
        return -1

    def _is_pose_class(self, cls_name):
        return cls_name in self.pose_config.categories

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(build_dialog_header(
            self, Icons.MERGE,
            "批量合并类别", "MERGE CLASSES  /  多类别合并到目标"))

        body = QHBoxLayout()
        body.setContentsMargins(16, 8, 16, 8)
        body.setSpacing(12)
        outer.addLayout(body, 1)

        # 左侧: 源类别 (多选)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(0)
        source_group = QGroupBox("选择要合并的类别  /  SOURCE")
        source_group.setProperty("accent", "blue")
        sg = QVBoxLayout(source_group)
        sg.setContentsMargins(10, 14, 10, 10)
        sg.setSpacing(6)

        self.class_list_widget = QListWidget()
        self.class_list_widget.setMaximumHeight(280)
        self.class_list_widget.setSpacing(2)

        self.class_checkboxes = {}
        for cls_name in self.class_list:
            info = self._get_class_info(cls_name)
            display_text = f"{cls_name}  ({info})"

            item = QListWidgetItem()
            item.setSizeHint(item.sizeHint() + item.sizeHint() * 0.3)

            checkbox = QCheckBox(display_text)
            checkbox.class_name = cls_name
            checkbox.stateChanged.connect(self._on_selection_changed)
            self.class_list_widget.addItem(item)
            self.class_list_widget.setItemWidget(item, checkbox)
            self.class_checkboxes[cls_name] = checkbox

        sg.addWidget(self.class_list_widget)
        left_layout.addWidget(source_group)
        body.addLayout(left_layout, 1)

        # 右侧: 目标 + 校验信息
        right_layout = QVBoxLayout()
        right_layout.setSpacing(0)
        target_group = QGroupBox("目标类别  /  TARGET")
        target_group.setProperty("accent", "red")
        tg = QFormLayout(target_group)
        tg.setContentsMargins(10, 14, 10, 10)
        tg.setSpacing(8)

        self.target_combo = QComboBox()
        self.target_combo.addItems(self.class_list)
        self.target_combo.currentTextChanged.connect(self._on_selection_changed)
        tg.addRow("合并到:", self.target_combo)

        self.info_label = QLabel("请选择要合并的类别")
        self.info_label.setProperty("role", "caption")
        self.info_label.setWordWrap(True)
        self.info_label.setProperty("role", "info")
        tg.addRow("", self.info_label)
        right_layout.addWidget(target_group)

        hint_label = QLabel("提示: pose 类别只能合并到 kpt 数量相同的 pose 类别; 普通类别可以合并到任意类别")
        hint_label.setProperty("role", "caption")
        right_layout.addSpacing(8)
        right_layout.addWidget(hint_label)

        body.addLayout(right_layout, 1)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(16, 8, 16, 14)
        btn_row.setSpacing(8)
        btn_row.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setProperty("role", "ghost")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.merge_btn = QPushButton("  合并")
        self.merge_btn.setCursor(Qt.PointingHandCursor)
        self.merge_btn.setProperty("role", "primary")
        self.merge_btn.setIcon(Icon.icon(Icons.MERGE, size=14, color=C.BG_APP))
        self.merge_btn.clicked.connect(self._on_merge)
        btn_row.addWidget(self.merge_btn)
        outer.addLayout(btn_row)

    def _get_selected_classes(self):
        return [cls for cls, cb in self.class_checkboxes.items() if cb.isChecked()]

    def _validate_and_get_preview(self):
        selected = self._get_selected_classes()
        target = self.target_combo.currentText()

        if len(selected) == 0:
            return False, "请至少选择一个类别"

        if target in selected:
            return False, "目标类别不能在被合并的类别中"

        selected_poses = [c for c in selected if self._is_pose_class(c)]
        target_is_pose = self._is_pose_class(target)

        if selected_poses:
            if not target_is_pose:
                return False, "pose 类别只能合并到 pose 类别"

            target_kpt = self._get_kpt_count(target)
            for c in selected_poses:
                if self._get_kpt_count(c) != target_kpt:
                    return False, f"pose 类别 {c} 的 kpt 数量({self._get_kpt_count(c)})与目标类别 {target} 的 kpt 数量({target_kpt})不一致"

        return True, f"确定将 {len(selected)} 个类别的标注合并到「{target}」?"

    def _on_selection_changed(self):
        valid, msg = self._validate_and_get_preview()
        self.info_label.setText(msg)
        self.info_label.setProperty("role", "info_success" if valid else "info_error")

    def _on_merge(self):
        valid, msg = self._validate_and_get_preview()
        if not valid:
            QMessageBox.warning(self, "提示", msg)
            return

        selected = self._get_selected_classes()
        target = self.target_combo.currentText()

        reply = QMessageBox.question(self, "确认合并",
                                     f"确定要将以下 {len(selected)} 个类别的标注合并到「{target}」?\n\n类别: {', '.join(selected)}\n\n此操作不可逆, 请确认!",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self.selected_classes = selected
        self.target_class = target
        self.accept()

    def get_merge_info(self):
        return getattr(self, 'selected_classes', []), getattr(self, 'target_class', '')
