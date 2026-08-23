# -*- coding: utf-8 -*-
"""Hierarchical annotation panel — 14-class 兵种 + L2 子类 (仅 BO/RO)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QGroupBox, QComboBox, QButtonGroup, QRadioButton,
    QScrollArea, QMenu,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QAction

from .icons import Icon, Icons, C
from core.hierarchical_config import (
    HierarchicalConfig,
    VIS_L2_NONE,
)


class HierarchicalPanel(QWidget):
    """层级标注右侧面板 — 14 类下拉 + L2 子类单选 (仅 BO/RO 可见) + 实例列表."""

    category_changed = Signal(str)        # 兵种名
    l2_code_changed = Signal(int)         # L2 子类码
    instance_selected = Signal(object)
    instance_delete_requested = Signal(object)
    config_edit_requested = Signal()
    status_message = Signal(str, str)

    def __init__(self, hier_config: HierarchicalConfig, parent=None):
        super().__init__(parent)
        self.hier_config = hier_config
        self.current_category: str | None = None
        self._l2_code = VIS_L2_NONE

        self._setup_ui()
        self._load_categories()
        self._load_persisted_l2()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setMaximumWidth(340)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_title())
        layout.addWidget(self._build_class_group())
        layout.addWidget(self._build_l2_group())
        layout.addWidget(self._build_instance_group())
        layout.addWidget(self._build_tips_group())
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_title(self) -> QWidget:
        row = QHBoxLayout()
        row.setSpacing(8)
        icon_label = QLabel()
        icon_label.setPixmap(Icon.pixmap(Icons.POSE, size=18, color=C.AMBER))
        row.addWidget(icon_label)
        title = QLabel("层级标注 / HIERARCHY")
        title.setProperty("role", "panel_title")
        row.addWidget(title, 1)
        wrapper = QWidget()
        wrapper.setLayout(row)
        return wrapper

    def _build_class_group(self) -> QWidget:
        group = QGroupBox("机器人兵种  /  CLASS (14)")
        group.setProperty("accent", "blue")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        layout.addWidget(self.class_combo)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_edit = QPushButton("编辑层级定义")
        btn_edit.setCursor(Qt.PointingHandCursor)
        btn_edit.setProperty("role", "ghost")
        btn_edit.clicked.connect(self.config_edit_requested.emit)
        btn_row.addWidget(btn_edit)
        layout.addLayout(btn_row)

        hint = QLabel("0:BG蓝哨兵 7:RG红哨兵 | BO/RO前哨含L2 | 其余4角点")
        hint.setProperty("role", "caption")
        layout.addWidget(hint)
        return group

    def _build_l2_group(self) -> QWidget:
        self.l2_group = QGroupBox("L2 子类  /  SUB-LEVEL")
        self.l2_group.setProperty("accent", "amber")
        layout = QVBoxLayout(self.l2_group)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(4)

        self.l2_button_group = QButtonGroup(self)
        self.l2_button_group.setExclusive(True)
        self.l2_radios: dict[str, QRadioButton] = {}
        for label_text in ("无", "up", "mid", "down"):
            radio = QRadioButton(label_text)
            radio.toggled.connect(lambda checked, lbl=label_text: self._on_l2_toggled(lbl, checked))
            self.l2_button_group.addButton(radio)
            self.l2_radios[label_text] = radio
            layout.addWidget(radio)
        self.l2_radios["无"].setChecked(True)

        self.l2_hint = QLabel("当前兵种无 L2 子类")
        self.l2_hint.setProperty("role", "caption")
        layout.addWidget(self.l2_hint)

        return self.l2_group

    def _build_instance_group(self) -> QWidget:
        group = QGroupBox("当前图片实例  /  INSTANCES")
        group.setProperty("accent", "muted")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self.instance_list = QListWidget()
        self.instance_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.instance_list.customContextMenuRequested.connect(self._on_instance_right_click)
        self.instance_list.itemClicked.connect(self._on_instance_clicked)
        layout.addWidget(self.instance_list)
        return group

    def _build_tips_group(self) -> QWidget:
        group = QGroupBox("操作提示  /  HOTKEYS")
        group.setProperty("accent", "muted")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.setSpacing(3)

        tips = [
            ("drag",  "拖拽绘制 bbox"),
            ("click", "点击放置 4 角点 (TL/TR/BL/BR)"),
            ("auto",  "KP4 L2 仅 BO/RO；RG 等为 4 点"),
            ("rclk",  "右键关键点切换可见性"),
        ]
        for key, desc in tips:
            row = QHBoxLayout()
            row.setSpacing(8)
            chip = QLabel(key.upper())
            chip.setProperty("role", "chip")
            chip.setProperty("accent", "blue")
            chip.setFixedWidth(50)
            chip.setAlignment(Qt.AlignCenter)
            label = QLabel(desc)
            label.setProperty("role", "caption")
            row.addWidget(chip)
            row.addWidget(label, 1)
            layout.addLayout(row)
        return group

    # ---------------------------------------------------------
    def _load_categories(self):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for c in self.hier_config.get_annotatable_classes():
            cid = c["id"]
            name = c["name"]
            has_sub = " (含L2)" if c.get("l2_levels") else ""
            display = f"{cid}: {name}{has_sub}"
            self.class_combo.addItem(display, name)
        self.class_combo.blockSignals(False)

    def _load_persisted_l2(self):
        from core.settings import Settings
        self._l2_code = Settings.app().get_int("Parameters/HIER_L2_CODE", VIS_L2_NONE)
        self._apply_l2_radio_from_code(self._l2_code)

    def _on_class_changed(self, combo_index: int):
        if combo_index < 0:
            self.current_category = None
            self.category_changed.emit("")
            return
        name = self.class_combo.itemData(combo_index)
        if not name:
            return
        self.current_category = name
        has_sub = self.hier_config.has_sublevels(name)
        self.l2_group.setVisible(has_sub)
        if has_sub:
            l2_levels = self.hier_config.get_l2_levels(name)
            self.l2_hint.setText(f"子类: {', '.join(l2_levels.keys())}")
        else:
            self.l2_hint.setText("当前兵种无 L2 子类")
            self._l2_code = VIS_L2_NONE
            self._apply_l2_radio_from_code(VIS_L2_NONE)
        self.category_changed.emit(name)

    def _on_l2_toggled(self, label_text: str, checked: bool):
        if not checked or not self.current_category:
            return
        l2_levels = self.hier_config.get_l2_levels(self.current_category)
        key = self._ui_label_to_key(label_text)
        self._l2_code = l2_levels.get(key, VIS_L2_NONE)
        from core.settings import Settings
        Settings.app().set("Parameters/HIER_L2_CODE", self._l2_code)
        Settings.app().sync()
        self.l2_code_changed.emit(self._l2_code)

    @staticmethod
    def _ui_label_to_key(label_text: str) -> str:
        return "none" if label_text == "无" else label_text.lower()

    def _apply_l2_radio_from_code(self, code: int):
        from core.hierarchical_config import L2_LEVELS_DEFAULT
        for key, value in L2_LEVELS_DEFAULT.items():
            if value == code:
                ui_label = "无" if key == "none" else key
                if ui_label in self.l2_radios:
                    self.l2_radios[ui_label].setChecked(True)
                return

    def get_l2_code(self) -> int:
        return self._l2_code

    def get_current_category(self) -> str | None:
        return self.current_category

    def reset_l2_for_new_image(self):
        """切图时调用: 重置 L2 子类为 none, 并按当前兵种决定 L2 栏是否显示。

        - L2 code 归零 (none), radio 回到 "无"
        - 当前兵种非 BO/RO → 隐藏 L2 栏
        - 当前兵种是 BO/RO → L2 栏保持可见, 但选择归 none (用户需重新选)
        """
        self._l2_code = VIS_L2_NONE
        self._apply_l2_radio_from_code(VIS_L2_NONE)
        # 持久化也归零, 避免下次开 app 仍带旧值
        from core.settings import Settings
        Settings.app().set("Parameters/HIER_L2_CODE", VIS_L2_NONE)
        Settings.app().sync()
        # L2 栏可见性跟随当前兵种
        has_sub = bool(self.current_category) and self.hier_config.has_sublevels(self.current_category)
        self.l2_group.setVisible(has_sub)
        if has_sub:
            l2_levels = self.hier_config.get_l2_levels(self.current_category)
            self.l2_hint.setText(f"子类: {', '.join(l2_levels.keys())}")
        else:
            self.l2_hint.setText("当前兵种无 L2 子类")
        # 通知 canvas 更新 hier_meta_l2_code
        self.l2_code_changed.emit(VIS_L2_NONE)

    def refresh(self):
        self._load_categories()
        if self.current_category:
            self._select_category(self.current_category)

    def _select_category(self, name: str):
        for i in range(self.class_combo.count()):
            if self.class_combo.itemData(i) == name:
                self.class_combo.setCurrentIndex(i)
                return

    def select_prev(self):
        i = self.class_combo.currentIndex()
        if i > 0:
            self.class_combo.setCurrentIndex(i - 1)

    def select_next(self):
        i = self.class_combo.currentIndex()
        if i < self.class_combo.count() - 1:
            self.class_combo.setCurrentIndex(i + 1)

    # ---------------------------------------------------------
    def _on_instance_clicked(self, item):
        shape = item.data(Qt.UserRole)
        if shape:
            self.instance_selected.emit(shape)

    def _on_instance_right_click(self, pos):
        item = self.instance_list.itemAt(pos)
        if not item:
            return
        shape = item.data(Qt.UserRole)
        if not shape:
            return
        menu = QMenu(self)
        delete_action = QAction("删除此实例", self)
        delete_action.setIcon(Icon.icon(Icons.TRASH, size=14, color=C.DANGER))
        delete_action.triggered.connect(lambda: self.instance_delete_requested.emit(shape))
        menu.addAction(delete_action)
        menu.exec(self.instance_list.mapToGlobal(pos))

    def update_instance_list(self, instances):
        self.instance_list.blockSignals(True)
        self.instance_list.clear()
        for inst in instances:
            idx = inst["index"]
            label = inst["label"]
            kpt_done = inst["kpt_done"]
            kpt_total = inst["kpt_total"]
            shape = inst["shape"]
            display = f"#{idx:02d}  {label}   {kpt_done}/{kpt_total}"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, shape)
            if kpt_done >= kpt_total:
                item.setForeground(QColor(C.AMBER))
            elif kpt_done > 0:
                item.setForeground(QColor("#DCBA80"))
            else:
                item.setForeground(QColor(C.TEXT_MUTED))
            self.instance_list.addItem(item)
        self.instance_list.blockSignals(False)
