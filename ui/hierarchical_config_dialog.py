# -*- coding: utf-8 -*-
"""Hierarchical class definition dialog — 编辑 14 类兵种与 L2 子类."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QGroupBox, QMessageBox, QSplitter,
    QWidget, QSpinBox,
)
from PySide6.QtCore import Qt

from .icons import Icon, Icons, C
from .dialog_utils import build_dialog_header
from core.hierarchical_config import (
    HierarchicalConfig, L2_LEVELS_DEFAULT, DEFAULT_KPT_NAMES, NUM_CLASSES,
)


class HierarchicalConfigDialog(QDialog):
    """编辑层级标注的兵种与 L2 子类定义 (14 类固定, 可改 L2)."""

    def __init__(self, hier_config: HierarchicalConfig, parent=None, preselect_name: str | None = None):
        super().__init__(parent)
        self.hier_config = hier_config
        self._editing_id: int | None = None
        self.setWindowTitle("层级标注定义")
        self.setWindowIcon(Icon.icon(Icons.POSE, size=24, color=C.AMBER))
        self.setMinimumSize(680, 500)
        self._setup_ui()
        self._load_classes()
        if preselect_name:
            self._select_class_by_name(preselect_name)

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_dialog_header(
            self, Icons.POSE,
            "层级标注定义", "HIERARCHY  /  14 类 + L2 子类"))

        body = QVBoxLayout()
        body.setContentsMargins(16, 8, 16, 8)
        body.setSpacing(10)
        outer.addLayout(body, 1)

        hint = QLabel(
            "14 类固定 (0:BG, 1-6:蓝方, 7-13:红方, 颜色已并入 class_id)。"
            "勾选「有 L2 子类」后可在下方编辑 up/mid/down。"
            "KP 0-3 为四角点, KP 4 = L2 子类 (visibility 编码, 坐标复用 bbox 中心)。"
        )
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        body.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._label("已定义的兵种 (14)"))
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._on_class_selected)
        left_layout.addWidget(self.class_list, 1)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._label("编辑兵种"))

        form = QVBoxLayout()
        form.setSpacing(6)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Class ID:"))
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 13)
        self.id_spin.setEnabled(False)
        id_row.addWidget(self.id_spin)
        id_row.addWidget(QLabel("兵种名:"))
        self.name_edit = QLineEdit()
        id_row.addWidget(self.name_edit, 1)
        form.addLayout(id_row)

        self.has_sublevel_btn = QPushButton("有 L2 子类: 关")
        self.has_sublevel_btn.setCheckable(True)
        self.has_sublevel_btn.setCursor(Qt.PointingHandCursor)
        self.has_sublevel_btn.setProperty("role", "tag")
        self.has_sublevel_btn.toggled.connect(self._on_has_sublevel_toggled)
        form.addWidget(self.has_sublevel_btn)

        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("L2 子类 (逗号分隔):"))
        self.sublevel_edit = QLineEdit()
        self.sublevel_edit.setPlaceholderText("如 up,mid,down")
        self.sublevel_edit.setEnabled(False)
        sub_row.addWidget(self.sublevel_edit, 1)
        form.addLayout(sub_row)

        self.kpt_preview = QTextEdit()
        self.kpt_preview.setReadOnly(True)
        self.kpt_preview.setMaximumHeight(120)
        form.addWidget(self.kpt_preview)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("  保存兵种")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setProperty("role", "primary")
        btn_save.clicked.connect(self._save_class)
        btn_row.addWidget(btn_save)
        form.addLayout(btn_row)
        right_layout.addLayout(form)
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        body.addWidget(splitter, 1)

        btn_close = QPushButton("  完成")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setProperty("role", "primary")
        btn_close.clicked.connect(self.accept)
        body.addWidget(btn_close)

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("role", "section_header")
        return lbl

    def _load_classes(self):
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for c in self.hier_config.classes:
            cid = c.get("id")
            name = c.get("name")
            has_sub = " +L2" if c.get("l2_levels") else ""
            item = QListWidgetItem(f"{cid}: {name}{has_sub}")
            item.setData(Qt.UserRole, cid)
            self.class_list.addItem(item)
        self.class_list.blockSignals(False)

    def _on_class_selected(self, row: int):
        if row < 0:
            return
        cid = self.class_list.item(row).data(Qt.UserRole)
        c = self.hier_config.get_class_by_id(cid)
        if not c:
            return
        self._editing_id = cid
        self.id_spin.setValue(cid)
        self.name_edit.setText(c.get("name", ""))
        has_sub = bool(c.get("l2_levels"))
        self.has_sublevel_btn.setChecked(has_sub)
        if has_sub:
            keys = [k for k in c["l2_levels"].keys() if k != "none"]
            self.sublevel_edit.setText(",".join(keys))
        else:
            self.sublevel_edit.setText("")
        self._update_kpt_preview(c)

    def _on_has_sublevel_toggled(self, checked: bool):
        self.has_sublevel_btn.setText("有 L2 子类: 开" if checked else "有 L2 子类: 关")
        self.sublevel_edit.setEnabled(checked)

    def _update_kpt_preview(self, c: dict):
        kpt_names = self.hier_config.kpt_names
        lines = [f"KP {i}: {name}" for i, name in enumerate(kpt_names)]
        l2_str = ", ".join(f"{k}={v}" for k, v in c.get("l2_levels", {}).items()) or "无"
        lines.append(f"\nL2: {l2_str}")
        self.kpt_preview.setPlainText("\n".join(lines))

    def _save_class(self):
        if self._editing_id is None:
            QMessageBox.warning(self, "提示", "请先在左侧选择一个兵种")
            return
        cid = self._editing_id
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "名称错误", "兵种名不能为空")
            return
        has_sub = self.has_sublevel_btn.isChecked()
        l2_levels = {}
        if has_sub:
            sub_text = self.sublevel_edit.text().strip()
            if sub_text:
                keys = [s.strip() for s in sub_text.split(",") if s.strip()]
                l2_levels["none"] = 0
                for i, key in enumerate(keys, start=1):
                    l2_levels[key] = i
        # 更新对应 class
        for c in self.hier_config.classes:
            if c.get("id") == cid:
                c["name"] = name
                c["l2_levels"] = l2_levels
                break
        self._load_classes()
        self._select_class_by_id(cid)

    def _select_class_by_id(self, cid: int):
        for i in range(self.class_list.count()):
            if self.class_list.item(i).data(Qt.UserRole) == cid:
                self.class_list.setCurrentRow(i)
                return

    def _select_class_by_name(self, name: str):
        c = self.hier_config.get_class_by_name(name)
        if c:
            self._select_class_by_id(c["id"])
