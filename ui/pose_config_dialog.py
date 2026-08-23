# -*- coding: utf-8 -*-
"""
位姿类别定义对话框 — RM 主题.
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QTextEdit, QListWidget,
                               QListWidgetItem, QGroupBox, QMessageBox, QSplitter,
                               QWidget)
from PySide6.QtCore import Qt

from core.pose_annotation_sync import snapshot_categories, compute_pose_config_changes, merge_pending_renames
from .icons import Icon, Icons, C
from .dialog_utils import build_dialog_header


class PoseConfigDialog(QDialog):
    """位姿类别定义对话框"""

    def __init__(self, pose_config, parent=None, preselect_name: str | None = None):
        super().__init__(parent)
        self.pose_config = pose_config
        self._snapshot = snapshot_categories(pose_config)
        self._editing_name: str | None = None
        self._pending_renames: dict[str, str] = {}
        self.setWindowTitle("位姿类别定义")
        self.setWindowIcon(Icon.icon(Icons.POSE, size=24, color=C.AMBER))
        self.setMinimumSize(680, 500)

        self._setup_ui()
        self._load_categories()
        if preselect_name:
            self._select_category(preselect_name)

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(build_dialog_header(
            self, Icons.POSE,
            "位姿类别定义", "POSE CLASSES  /  关键点定义"))

        body = QVBoxLayout()
        body.setContentsMargins(16, 8, 16, 8)
        body.setSpacing(10)
        outer.addLayout(body, 1)

        hint = QLabel(
            "选中左侧类别可编辑；关键点减少后保存将全局裁剪多余点；"
            "删除的类别会同步移除目录内对应位姿标注。"
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        body.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)

        # ---- 左侧: 已有类别 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(6)

        header = QLabel("已定义的位姿类别")
        header.setProperty("role", "section_header")
        left_layout.addWidget(header)

        self.category_list = QListWidget()
        self.category_list.currentRowChanged.connect(self._on_category_selected)
        left_layout.addWidget(self.category_list, 1)

        btn_del = QPushButton("  删除选中类别")
        btn_del.setCursor(Qt.PointingHandCursor)
        btn_del.setProperty("role", "danger")
        btn_del.setIcon(Icon.icon(Icons.TRASH, size=14, color="#FFFFFF"))
        btn_del.clicked.connect(self._delete_category)
        left_layout.addWidget(btn_del)

        splitter.addWidget(left_widget)

        # ---- 右侧: 编辑区 ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        right_layout.setSpacing(8)

        edit_group = QGroupBox("编辑  /  EDIT")
        edit_group.setProperty("accent", "blue")
        eg = QVBoxLayout(edit_group)
        eg.setContentsMargins(10, 14, 10, 10)
        eg.setSpacing(8)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("类别名称:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如: pillar, exchange, BT...")
        name_row.addWidget(self.name_input, 1)
        eg.addLayout(name_row)

        eg.addWidget(QLabel("关键点名称 (每行一个, 按顺序):"))
        self.kpt_input = QTextEdit()
        self.kpt_input.setPlaceholderText(
            "kpt0\nkpt1\nkpt2\nkpt3\nkpt4\n..."
        )
        self.kpt_input.setMinimumHeight(180)
        eg.addWidget(self.kpt_input, 1)

        edit_btn_row = QHBoxLayout()
        edit_btn_row.setSpacing(6)
        self.btn_add = QPushButton("  添加 / 更新")
        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_add.setProperty("role", "primary")
        self.btn_add.setIcon(Icon.icon(Icons.PLUS, size=14, color=C.BG_APP))
        self.btn_add.clicked.connect(self._add_category)
        edit_btn_row.addWidget(self.btn_add)

        btn_clear = QPushButton("清空输入")
        btn_clear.setCursor(Qt.PointingHandCursor)
        btn_clear.setProperty("role", "ghost")
        btn_clear.clicked.connect(self._clear_input)
        edit_btn_row.addWidget(btn_clear)
        edit_btn_row.addStretch()
        eg.addLayout(edit_btn_row)

        right_layout.addWidget(edit_group, 1)
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 460])
        body.addWidget(splitter, 1)

        # 底部按钮
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(16, 8, 16, 14)
        btn_row.setSpacing(8)
        btn_row.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setProperty("role", "ghost")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        btn_save = QPushButton("  保存并关闭")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setProperty("role", "primary")
        btn_save.setIcon(Icon.icon(Icons.CHECK, size=14, color=C.BG_APP))
        btn_save.clicked.connect(self._save_and_close)
        btn_row.addWidget(btn_save)
        outer.addLayout(btn_row)

    def _select_category(self, name: str):
        for i in range(self.category_list.count()):
            if self.category_list.item(i).data(Qt.UserRole) == name:
                self.category_list.setCurrentRow(i)
                return

    def _load_categories(self):
        self.category_list.clear()
        for name in self.pose_config.get_category_names():
            cat = self.pose_config.get_category(name)
            item = QListWidgetItem(f"{name}   ({cat['kpt_count']} kpts)")
            item.setData(Qt.UserRole, name)
            self.category_list.addItem(item)

    def _on_category_selected(self, row):
        if row < 0:
            self._editing_name = None
            return
        item = self.category_list.item(row)
        name = item.data(Qt.UserRole)
        self._editing_name = name
        cat = self.pose_config.get_category(name)
        if cat:
            self.name_input.setText(name)
            self.kpt_input.setText("\n".join(cat["keypoints"]))

    def _add_category(self):
        name = self.name_input.text().strip()
        kpt_text = self.kpt_input.toPlainText().strip()

        if not name:
            QMessageBox.warning(self, "提示", "请输入类别名称")
            return
        if not kpt_text:
            QMessageBox.warning(self, "提示", "请输入关键点名称 (每行一个)")
            return

        keypoints = [k.strip() for k in kpt_text.split("\n") if k.strip()]
        if len(keypoints) < 1:
            QMessageBox.warning(self, "提示", "至少需要 1 个关键点")
            return

        old_name = self._editing_name
        old_cat = self.pose_config.get_category(old_name) if old_name else None

        if old_name and old_name != name:
            if self.pose_config.has_category(name):
                QMessageBox.warning(self, "提示", f"类别 '{name}' 已存在")
                return
        elif not old_name and self.pose_config.has_category(name):
            old_cat = self.pose_config.get_category(name)

        if old_cat and len(keypoints) < old_cat["kpt_count"]:
            reply = QMessageBox.question(
                self, "确认裁剪关键点",
                f"类别 '{old_name or name}' 关键点从 {old_cat['kpt_count']} "
                f"减少到 {len(keypoints)}。\n\n"
                "保存后将全局删除多余关键点，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if old_name and old_name != name:
            self._pending_renames[old_name] = name
            self.pose_config.remove_category(old_name)

        self.pose_config.add_category(name, keypoints)
        self._editing_name = name
        self._load_categories()
        self._select_category(name)

    def _delete_category(self):
        row = self.category_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先选中要删除的类别")
            return

        item = self.category_list.item(row)
        name = item.data(Qt.UserRole)

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除位姿类别 '{name}' 吗？\n"
            "保存后将移除当前目录内该类的全部位姿标注。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self._editing_name == name:
                self._editing_name = None
            for old, new in list(self._pending_renames.items()):
                if new == name:
                    del self._pending_renames[old]
            self.pose_config.remove_category(name)
            self._load_categories()
            self._clear_input()

    def _clear_input(self):
        self._editing_name = None
        self.name_input.clear()
        self.kpt_input.clear()
        self.category_list.clearSelection()

    def get_changes(self) -> dict:
        """相对打开对话框时的快照，计算全部变更。"""
        after = snapshot_categories(self.pose_config)
        changes = compute_pose_config_changes(self._snapshot, after)
        return merge_pending_renames(changes, self._pending_renames)

    def _save_and_close(self):
        self.accept()
