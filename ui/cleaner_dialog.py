# -*- coding: utf-8 -*-
"""
清理无标签图片确认对话框 — RM 主题 (走 style.qss).
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QGroupBox, QCheckBox, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt

from .icons import Icon, Icons, C
from .dialog_utils import build_dialog_header, build_dialog_footer


class CleanerDialog(QDialog):
    """批量删除无标签图片的确认对话框"""

    def __init__(self, unlabeled_images, empty_label_images=None, parent=None):
        super().__init__(parent)
        self.unlabeled_images = unlabeled_images
        self.empty_label_images = empty_label_images or []
        self.selected_for_delete = []

        self.setWindowTitle("清理无标签图片")
        self.setWindowIcon(Icon.icon(Icons.BROOM, size=24, color=C.AMBER))
        self.setMinimumSize(600, 480)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 标题条
        outer.addWidget(build_dialog_header(
            self, Icons.BROOM,
            "清理无标签图片", "CLEAN UNLABELED  /  批量清理"))

        body = QVBoxLayout()
        body.setContentsMargins(16, 8, 16, 8)
        body.setSpacing(10)
        outer.addLayout(body, 1)

        # 顶部统计
        total_unlabeled = len(self.unlabeled_images)
        total_empty = len(self.empty_label_images)
        total = total_unlabeled + total_empty

        info_label = QLabel(
            f"共发现 <b style='color:#B5756A;'>{total}</b> 张需要清理的图片"
            f"（无标注: <b style='color:#C9A36B;'>{total_unlabeled}</b> 张"
            f"，空标注: <b style='color:#C9A36B;'>{total_empty}</b> 张）"
        )
        info_label.setTextFormat(Qt.RichText)
        info_label.setProperty("role", "info")
        body.addWidget(info_label)

        # 删除选项
        self.trash_checkbox = QCheckBox("移动到回收站（.trash 目录），可恢复")
        self.trash_checkbox.setChecked(True)
        body.addWidget(self.trash_checkbox)

        # 无标注列表
        if self.unlabeled_images:
            group1 = QGroupBox(f"无标注文件的图片  ({total_unlabeled} 张)")
            group1.setProperty("accent", "red")
            list1 = QListWidget()
            list1.setSelectionMode(QAbstractItemView.ExtendedSelection)
            for img_path in self.unlabeled_images:
                item = QListWidgetItem(os.path.basename(img_path))
                item.setToolTip(img_path)
                item.setData(Qt.UserRole, img_path)
                item.setCheckState(Qt.Checked)
                list1.addItem(item)
            g1 = QVBoxLayout(group1)
            g1.setContentsMargins(10, 14, 10, 10)
            g1.addWidget(list1)
            body.addWidget(group1)
            self.list_unlabeled = list1

        if self.empty_label_images:
            group2 = QGroupBox(f"标注文件为空的图片  ({total_empty} 张)")
            group2.setProperty("accent", "amber")
            list2 = QListWidget()
            list2.setSelectionMode(QAbstractItemView.ExtendedSelection)
            for img_path, anno_path, fmt in self.empty_label_images:
                display = f"{os.path.basename(img_path)}   (标注: {os.path.basename(anno_path)})"
                item = QListWidgetItem(display)
                item.setToolTip(f"图片: {img_path}\n标注: {anno_path}")
                item.setData(Qt.UserRole, img_path)
                item.setCheckState(Qt.Checked)
                list2.addItem(item)
            g2 = QVBoxLayout(group2)
            g2.setContentsMargins(10, 14, 10, 10)
            g2.addWidget(list2)
            body.addWidget(group2)
            self.list_empty = list2

        # 底部按钮行 (左: 全选, 右: 取消 + 删除)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(16, 8, 16, 14)
        btn_row.setSpacing(8)

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.setCursor(Qt.PointingHandCursor)
        self.select_all_btn.setProperty("role", "ghost")
        self.select_all_btn.clicked.connect(self._toggle_select_all)
        btn_row.addWidget(self.select_all_btn)
        btn_row.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setProperty("role", "ghost")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.delete_btn = QPushButton(f"  删除选中 ({total} 张)")
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setProperty("role", "danger")
        self.delete_btn.setIcon(Icon.icon(Icons.TRASH, size=14, color="#FFFFFF"))
        self.delete_btn.setIconSize(self.delete_btn.iconSize())
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)

        outer.addLayout(btn_row)

    def _toggle_select_all(self):
        lists = []
        if hasattr(self, 'list_unlabeled'):
            lists.append(self.list_unlabeled)
        if hasattr(self, 'list_empty'):
            lists.append(self.list_empty)
        if not lists:
            return

        all_checked = True
        for lst in lists:
            for i in range(lst.count()):
                if lst.item(i).checkState() != Qt.Checked:
                    all_checked = False
                    break
            if not all_checked:
                break

        new_state = Qt.Unchecked if all_checked else Qt.Checked
        for lst in lists:
            for i in range(lst.count()):
                lst.item(i).setCheckState(new_state)
        self.select_all_btn.setText("取消全选" if all_checked else "全选")

    def _on_delete(self):
        selected = []
        lists = []
        if hasattr(self, 'list_unlabeled'):
            lists.append(self.list_unlabeled)
        if hasattr(self, 'list_empty'):
            lists.append(self.list_empty)

        for lst in lists:
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    selected.append(item.data(Qt.UserRole))

        if not selected:
            QMessageBox.warning(self, "提示", "请先选择要删除的图片")
            return

        to_trash = self.trash_checkbox.isChecked()
        action_text = "移动到回收站" if to_trash else "永久删除"
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要 {action_text} 选中的 <b>{len(selected)}</b> 张图片吗？\n\n"
            + ("（图片及对应的标注文件将被移动到 .trash 目录，可恢复）" if to_trash else "（此操作不可恢复！）"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.selected_for_delete = selected
            self.accept()

    def get_selected_files(self):
        return self.selected_for_delete

    def is_to_trash(self):
        return self.trash_checkbox.isChecked()
