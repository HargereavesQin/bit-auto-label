# -*- coding: utf-8 -*-
"""
位姿标注独立 UI 面板 — 模块化设计, 不含 SAM 推理.

视觉: RM 竞技风 (走 ui/style.qss).
"""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QGroupBox, QInputDialog, QMessageBox, QMenu, QScrollArea,
    QSizePolicy, QLineEdit, QFileDialog, QButtonGroup,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QAction

from .icons import Icon, Icons, C


class PosePanel(QWidget):
    """位姿标注右侧面板."""

    category_changed = Signal(str)
    request_new_category = Signal()
    category_edit_requested = Signal(str)
    category_delete_requested = Signal(str)
    instance_selected = Signal(object)
    instance_delete_requested = Signal(object)
    rm_auto_toggled = Signal(bool)
    full_auto_requested = Signal()
    batch_auto_requested = Signal()
    coverage_apply_requested = Signal()   # 等效覆盖: 当前图 pose 标注复制到后续所有图
    lora_manual_requested = Signal(str)   # LoRA 手动: 类别名
    lora_auto_requested = Signal()        # LoRA 自动: 遍历全部类别
    auto_mode_changed = Signal(str)
    buff_yolo_rough_requested = Signal()
    buff_sam_refine_requested = Signal()
    buff_sam3_auto_requested = Signal()
    buff_weight_select_requested = Signal(str)
    kpt_only_toggled = Signal(bool)
    status_message = Signal(str, str)   # (msg, level) level ∈ info/warning/error/success

    def __init__(self, pose_config, parent=None):
        super().__init__(parent)
        self.pose_config = pose_config
        self.current_category = None
        self._auto_mode = "armor"

        self._setup_ui()
        self._load_categories()
        self._update_auto_mode_ui()
        self._load_kpt_only_setting()

    # ---------------------------------------------------------
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

        # ----- 标题 -----
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_icon = QLabel()
        title_icon.setPixmap(Icon.pixmap(Icons.POSE, size=18, color=C.AMBER))
        title_row.addWidget(title_icon)
        title = QLabel("位姿标注 / POSE")
        title.setProperty("role", "panel_title")
        title_row.addWidget(title, 1)
        layout.addLayout(title_row)

        # ----- 位姿类别组 -----
        list_group = QGroupBox("位姿类别  /  W / S 切换")
        list_group.setProperty("accent", "blue")
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(8, 12, 8, 8)
        list_layout.setSpacing(6)

        self.category_list = QListWidget()
        self.category_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.category_list.currentRowChanged.connect(self._on_selection_changed)
        self.category_list.customContextMenuRequested.connect(self._on_category_right_click)
        list_layout.addWidget(self.category_list)

        btn_new = QPushButton("  新建位姿类别")
        btn_new.setCursor(Qt.PointingHandCursor)
        btn_new.setProperty("role", "primary")
        btn_new.setIcon(Icon.icon(Icons.PLUS, size=14, color=C.BG_APP))
        btn_new.setIconSize(btn_new.iconSize())
        btn_new.clicked.connect(self._on_new_category)
        list_layout.addWidget(btn_new)

        btn_edit = QPushButton("  编辑选中类别")
        btn_edit.setCursor(Qt.PointingHandCursor)
        btn_edit.setProperty("role", "ghost")
        btn_edit.setIcon(Icon.icon(Icons.GEAR, size=14, color=C.TEXT))
        btn_edit.clicked.connect(self._on_edit_category)
        list_layout.addWidget(btn_edit)

        layout.addWidget(list_group)

        # ----- 手动标注选项 -----
        manual_group = QGroupBox("手动标注  /  MANUAL")
        manual_group.setProperty("accent", "amber")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.setContentsMargins(8, 12, 8, 8)
        manual_layout.setSpacing(6)

        self.kpt_only_btn = QPushButton("仅标点: 关")
        self.kpt_only_btn.setCursor(Qt.PointingHandCursor)
        self.kpt_only_btn.setCheckable(True)
        self.kpt_only_btn.setProperty("role", "tag")
        self.kpt_only_btn.toggled.connect(self._on_kpt_only_toggled)
        manual_layout.addWidget(self.kpt_only_btn)

        self.kpt_only_tip = QLabel("关闭：先拖拽 bbox，再点击添加关键点")
        self.kpt_only_tip.setProperty("role", "caption")
        self.kpt_only_tip.setWordWrap(True)
        manual_layout.addWidget(self.kpt_only_tip)

        layout.addWidget(manual_group)

        # ----- 自动标注模式组 -----
        auto_group = QGroupBox("自动标注模式  /  AUTO MODE")
        auto_group.setProperty("accent", "muted")
        auto_layout = QVBoxLayout(auto_group)
        auto_layout.setContentsMargins(8, 12, 8, 8)
        auto_layout.setSpacing(8)

        # Armor / Buff 切换
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.mode_armor_btn = QPushButton("Armor")
        self.mode_buff_btn = QPushButton("Buff-Merge")
        for btn in (self.mode_armor_btn, self.mode_buff_btn):
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("role", "tag")
        self.mode_armor_btn.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.mode_armor_btn)
        self.mode_group.addButton(self.mode_buff_btn)
        self.mode_armor_btn.toggled.connect(lambda c: c and self._set_auto_mode("armor"))
        self.mode_buff_btn.toggled.connect(lambda c: c and self._set_auto_mode("buff-merge"))
        mode_row.addWidget(self.mode_armor_btn)
        mode_row.addWidget(self.mode_buff_btn)
        auto_layout.addLayout(mode_row)

        # ---- RM Armor 子组 ----
        self.rm_group = QGroupBox("RM 装甲板  /  椭圆拟合")
        self.rm_group.setProperty("accent", "red")
        rm_layout = QVBoxLayout(self.rm_group)
        rm_layout.setContentsMargins(8, 12, 8, 8)
        rm_layout.setSpacing(6)

        self.rm_auto_btn = QPushButton("半自动: 关")
        self.rm_auto_btn.setCursor(Qt.PointingHandCursor)
        self.rm_auto_btn.setCheckable(True)
        self.rm_auto_btn.setProperty("role", "tag")
        self.rm_auto_btn.toggled.connect(self._on_rm_auto_toggled)
        rm_layout.addWidget(self.rm_auto_btn)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.full_auto_btn = QPushButton("  全自动")
        self.full_auto_btn.setCursor(Qt.PointingHandCursor)
        self.full_auto_btn.setProperty("role", "primary")
        self.full_auto_btn.setIcon(Icon.icon(Icons.SPARK, size=14, color=C.BG_APP))
        self.full_auto_btn.clicked.connect(self._on_full_auto_clicked)
        action_row.addWidget(self.full_auto_btn)

        self.batch_auto_btn = QPushButton("  批量")
        self.batch_auto_btn.setCursor(Qt.PointingHandCursor)
        self.batch_auto_btn.setProperty("role", "danger")
        self.batch_auto_btn.setIcon(Icon.icon(Icons.AUTO, size=14, color=C.DANGER))
        self.batch_auto_btn.clicked.connect(self._on_batch_auto_clicked)
        action_row.addWidget(self.batch_auto_btn)
        rm_layout.addLayout(action_row)

        # 等效覆盖: 把当前图的 pose 标注 (bbox + 关键点位置) 复制覆盖到后续所有图
        self.coverage_btn = QPushButton("  等效覆盖")
        self.coverage_btn.setCursor(Qt.PointingHandCursor)
        self.coverage_btn.setProperty("role", "tag")
        self.coverage_btn.setToolTip(
            "把当前图片的 pose 标注 (bbox + 关键点位置) 复制覆盖到后续所有图片。\n"
            "适用于固定机位/目标位置不变的连续帧 (如视频抽帧)。")
        self.coverage_btn.clicked.connect(self._on_coverage_clicked)
        rm_layout.addWidget(self.coverage_btn)

        # ---- LoRA-SAM 端到端（精度不足，暂时从面板隐藏；代码保留） ----
        lora_row = QHBoxLayout()
        lora_row.setSpacing(6)
        self.lora_manual_btn = QPushButton("  LoRA 手动")
        self.lora_manual_btn.setCursor(Qt.PointingHandCursor)
        self.lora_manual_btn.setProperty("role", "tag")
        self.lora_manual_btn.setToolTip(
            "输入类别名（如 B1），用 LoRA-SAM 端到端检测并标注 Pose。"
        )
        self.lora_manual_btn.clicked.connect(self._on_lora_manual_clicked)
        self.lora_manual_btn.setVisible(False)
        lora_row.addWidget(self.lora_manual_btn)

        self.lora_auto_btn = QPushButton("  LoRA 自动")
        self.lora_auto_btn.setCursor(Qt.PointingHandCursor)
        self.lora_auto_btn.setProperty("role", "primary")
        self.lora_auto_btn.setToolTip(
            "遍历全部 14 类，LoRA-SAM 检测 + minAreaRect 关键点标注。"
        )
        self.lora_auto_btn.clicked.connect(self._on_lora_auto_clicked)
        self.lora_auto_btn.setVisible(False)
        lora_row.addWidget(self.lora_auto_btn)
        rm_layout.addLayout(lora_row)

        lora_tip = QLabel("需在设置中配置 LoRA Adapter 路径并重启")
        lora_tip.setProperty("role", "caption")
        lora_tip.setWordWrap(True)
        lora_tip.setVisible(False)
        rm_layout.addWidget(lora_tip)

        self.lora_status_label = QLabel("LoRA 就绪")
        self.lora_status_label.setProperty("role", "caption")
        self.lora_status_label.setWordWrap(True)
        self.lora_status_label.setMinimumHeight(36)
        self.lora_status_label.setVisible(False)
        rm_layout.addWidget(self.lora_status_label)

        auto_layout.addWidget(self.rm_group)

        # ---- Buff-Merge 子组 ----
        self.buff_group = QGroupBox("能量机关 BUFF  /  YOLO + SAM")
        self.buff_group.setProperty("accent", "blue")
        buff_layout = QVBoxLayout(self.buff_group)
        buff_layout.setContentsMargins(8, 12, 8, 8)
        buff_layout.setSpacing(6)

        self.buff_weight_edit = QLineEdit()
        self.buff_weight_edit.setPlaceholderText("YOLO pose 权重 (.pt / .onnx / .engine)")
        self.buff_weight_edit.setReadOnly(True)
        self.buff_weight_edit.setProperty("role", "mono")
        buff_layout.addWidget(self.buff_weight_edit)

        buff_weight_row = QHBoxLayout()
        buff_weight_row.setSpacing(6)
        self.buff_weight_btn = QPushButton("选择权重")
        self.buff_yolo_rough_btn = QPushButton("YOLO 粗标")
        for btn in (self.buff_weight_btn, self.buff_yolo_rough_btn):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("role", "tag")
        self.buff_weight_btn.clicked.connect(self._on_buff_weight_clicked)
        self.buff_yolo_rough_btn.clicked.connect(self._on_buff_yolo_rough_clicked)
        buff_weight_row.addWidget(self.buff_weight_btn)
        buff_weight_row.addWidget(self.buff_yolo_rough_btn)
        buff_layout.addLayout(buff_weight_row)

        self.buff_sam3_auto_btn = QPushButton("  BUFF SAM3 一键全自动 (ONNX+SAM3)")
        self.buff_sam3_auto_btn.setCursor(Qt.PointingHandCursor)
        self.buff_sam3_auto_btn.setProperty("role", "primary")
        self.buff_sam3_auto_btn.setIcon(Icon.icon(Icons.SPARK, size=14, color=C.BG_APP))
        self.buff_sam3_auto_btn.setToolTip("Step 0: 用 YOLO11n ONNX 圈定 ROI\nStep 1~3: 在 ROI 内用 SAM3 分割 circle 和 R 标，极轴几何解算五点")
        self.buff_sam3_auto_btn.clicked.connect(self._on_buff_sam3_auto_clicked)
        buff_layout.addWidget(self.buff_sam3_auto_btn)

        self.buff_sam_refine_btn = QPushButton("  SAM 精修关键点 (交互)")
        self.buff_sam_refine_btn.setCursor(Qt.PointingHandCursor)
        self.buff_sam_refine_btn.setProperty("role", "secondary")
        self.buff_sam_refine_btn.setIcon(Icon.icon(Icons.SPARK, size=14, color=C.BG_APP))
        self.buff_sam_refine_btn.clicked.connect(self._on_buff_sam_refine_clicked)
        buff_layout.addWidget(self.buff_sam_refine_btn)

        self.buff_semi_auto_btn = QPushButton("  扇叶半自动 (拉框 → SAM3 五点)")
        self.buff_semi_auto_btn.setCursor(Qt.PointingHandCursor)
        self.buff_semi_auto_btn.setCheckable(True)
        self.buff_semi_auto_btn.setChecked(True)
        self.buff_semi_auto_btn.setProperty("role", "tag")
        self.buff_semi_auto_btn.setToolTip(
            "开启后：在能量机关扇叶周围手动拉框，\n"
            "自动在框内用 SAM3 分割 circle 和 R 标，极轴几何解算并放置五点"
        )
        buff_layout.addWidget(self.buff_semi_auto_btn)

        buff_tip = QLabel("半自动: 拉框 → SAM3 圈圆&R 标自动落五点\n全自动: 自动 ONNX ROI + SAM3 圈圆&R 标求五点\n交互精修: 基于选中 YOLO 粗标点 SAM 精修")
        buff_tip.setProperty("role", "caption")
        buff_tip.setWordWrap(True)
        buff_layout.addWidget(buff_tip)

        auto_layout.addWidget(self.buff_group)
        layout.addWidget(auto_group)

        # ----- 当前选中信息 -----
        self.info_group = QGroupBox("当前选中  /  SELECTED")
        self.info_group.setProperty("accent", "amber")
        info_layout = QVBoxLayout(self.info_group)
        info_layout.setContentsMargins(8, 12, 8, 8)
        info_layout.setSpacing(4)

        self.info_label = QLabel("未选择位姿类别")
        self.info_label.setProperty("role", "caption")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        self.kpt_info = QLabel("")
        self.kpt_info.setProperty("role", "mono")
        self.kpt_info.setAlignment(Qt.AlignCenter)
        self.kpt_info.setWordWrap(True)
        info_layout.addWidget(self.kpt_info)

        layout.addWidget(self.info_group)

        # ----- 当前图片实例 -----
        instance_group = QGroupBox("当前图片实例  /  INSTANCES")
        instance_group.setProperty("accent", "red")
        instance_layout = QVBoxLayout(instance_group)
        instance_layout.setContentsMargins(8, 12, 8, 8)
        instance_layout.setSpacing(6)

        self.instance_list = QListWidget()
        self.instance_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.instance_list.customContextMenuRequested.connect(self._on_instance_right_click)
        self.instance_list.itemClicked.connect(self._on_instance_clicked)
        instance_layout.addWidget(self.instance_list)
        layout.addWidget(instance_group)

        # ----- 操作提示 -----
        tips_group = QGroupBox("操作提示  /  HOTKEYS")
        tips_group.setProperty("accent", "muted")
        tips_layout = QVBoxLayout(tips_group)
        tips_layout.setContentsMargins(10, 12, 10, 8)
        tips_layout.setSpacing(3)
        tips = [
            ("drag",  "拖拽绘制 bbox（仅标点模式关闭时）"),
            ("click", "点击放置关键点"),
            ("rclk",  "右键关键点切换可见性"),
            ("w",     "上一个类别 / 切换实例类别"),
            ("s",     "下一个类别 / 切换实例类别"),
        ]
        self._tip_desc_labels = {}
        for key, desc in tips:
            row = QHBoxLayout()
            row.setSpacing(8)
            k = QLabel(key.upper())
            k.setProperty("role", "chip")
            k.setProperty("accent", "blue")
            k.setFixedWidth(50)
            k.setAlignment(Qt.AlignCenter)
            d = QLabel(desc)
            d.setProperty("role", "caption")
            row.addWidget(k)
            row.addWidget(d, 1)
            self._tip_desc_labels[key] = d
            tips_layout.addLayout(row)
        layout.addWidget(tips_group)

        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ---------------------------------------------------------
    def _load_categories(self):
        self.category_list.blockSignals(True)
        self.category_list.clear()
        for name in self.pose_config.get_category_names():
            cat = self.pose_config.get_category(name)
            display = f"{name}   ({cat['kpt_count']} kpts)"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, name)
            self.category_list.addItem(item)
        self.category_list.blockSignals(False)

        if self.current_category:
            self._select_category(self.current_category)
        # 不再自动选第一个, 让用户主动选择

    def _on_selection_changed(self, row):
        if row < 0:
            self.current_category = None
            self.info_label.setText("未选择位姿类别")
            self.kpt_info.setText("")
            self.category_changed.emit("")
            return

        item = self.category_list.item(row)
        name = item.data(Qt.UserRole) if item else None
        if not name:
            return
        self.current_category = name
        cat = self.pose_config.get_category(name)

        if cat:
            kpt_list = "\n".join([f"  {i+1}. {k}" for i, k in enumerate(cat["keypoints"])])
            self.info_label.setText(f"<b style='color:#C9A36B;'>{name}</b>")
            self.kpt_info.setText(f"关键点 ({cat['kpt_count']} 个):\n{kpt_list}")
            self.kpt_info.setTextFormat(Qt.RichText)
        else:
            self.info_label.setText(name)
            self.kpt_info.setText("")

        self.category_changed.emit(name)

    def _on_new_category(self):
        self.request_new_category.emit()

    def _on_edit_category(self):
        if not self.current_category:
            self._emit_status("请先在位姿类别里选择一项", "warning")
            return
        self.category_edit_requested.emit(self.current_category)

    def _set_auto_mode(self, mode):
        if self._auto_mode == mode:
            return
        self._auto_mode = mode
        self._update_auto_mode_ui()
        self.auto_mode_changed.emit(mode)

    def _update_auto_mode_ui(self):
        is_armor = self._auto_mode == "armor"
        self.rm_group.setVisible(is_armor)
        self.buff_group.setVisible(not is_armor)
        if not is_armor and self.rm_auto_btn.isChecked():
            self.rm_auto_btn.blockSignals(True)
            self.rm_auto_btn.setChecked(False)
            self.rm_auto_btn.blockSignals(False)

    def get_auto_mode(self):
        return self._auto_mode

    def is_buff_semi_auto(self):
        return self.buff_semi_auto_btn.isChecked()

    def set_buff_weight_path(self, path):
        self.buff_weight_edit.setText(path or "")

    def _on_buff_weight_clicked(self):
        default_dir = os.path.dirname(self.buff_weight_edit.text()) if self.buff_weight_edit.text() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择能量机关 YOLO 权重", default_dir,
            "权重文件 (*.pt *.onnx *.engine);;所有文件 (*)"
        )
        if path:
            self.buff_weight_edit.setText(path)
            self.buff_weight_select_requested.emit(path)

    def _on_buff_yolo_rough_clicked(self):
        self.buff_yolo_rough_requested.emit()

    def _on_buff_sam_refine_clicked(self):
        self.buff_sam_refine_requested.emit()

    def _on_buff_sam3_auto_clicked(self):
        self.buff_sam3_auto_requested.emit()

    def _on_rm_auto_toggled(self, checked):
        if checked and self._auto_mode == "buff-merge":
            self.rm_auto_btn.blockSignals(True)
            self.rm_auto_btn.setChecked(False)
            self.rm_auto_btn.blockSignals(False)
            return
        self.rm_auto_toggled.emit(checked)
        self.rm_auto_btn.setText(f"半自动: {'开' if checked else '关'}")

    def _on_full_auto_clicked(self):
        if not self.current_category:
            self._emit_status("请先在位姿类别里选择一项", "warning")
            return
        if self.full_auto_btn.isEnabled() is False:
            return
        self.full_auto_requested.emit()

    def _on_batch_auto_clicked(self):
        if not self.current_category:
            self._emit_status("请先在位姿类别里选择一项", "warning")
            return
        if self.batch_auto_btn.isEnabled() is False:
            return
        self.batch_auto_requested.emit()

    def _on_coverage_clicked(self):
        if not self.current_category:
            self._emit_status("请先在位姿类别里选择一项", "warning")
            return
        self.coverage_apply_requested.emit()

    def _load_kpt_only_setting(self):
        from core.settings import Settings
        enabled = Settings.app().get_bool("Parameters/POSE_KPT_ONLY", False)
        self.kpt_only_btn.blockSignals(True)
        self.kpt_only_btn.setChecked(enabled)
        self.kpt_only_btn.blockSignals(False)
        self._apply_kpt_only_ui(enabled)

    def _on_kpt_only_toggled(self, checked: bool):
        from core.settings import Settings
        Settings.app().set("Parameters/POSE_KPT_ONLY", checked)
        Settings.app().sync()
        self._apply_kpt_only_ui(checked)
        self.kpt_only_toggled.emit(checked)

    def _apply_kpt_only_ui(self, enabled: bool):
        self.kpt_only_btn.setText("仅标点: 开" if enabled else "仅标点: 关")
        if enabled:
            self.kpt_only_tip.setText("开启：点击空白处直接标点，bbox 自动计算并隐藏")
            if "drag" in self._tip_desc_labels:
                self._tip_desc_labels["drag"].setText("（仅标点模式下无需拖拽 bbox）")
        else:
            self.kpt_only_tip.setText("关闭：先拖拽 bbox，再点击添加关键点")
            if "drag" in self._tip_desc_labels:
                self._tip_desc_labels["drag"].setText("拖拽绘制 bbox（仅标点模式关闭时）")

    def is_kpt_only(self) -> bool:
        return self.kpt_only_btn.isChecked()

    def set_kpt_only(self, enabled: bool):
        self.kpt_only_btn.setChecked(enabled)

    # ---------------------------------------------------------
    # 状态反馈 + 忙时禁用
    # ---------------------------------------------------------
    def _emit_status(self, msg: str, level: str = "info"):
        """把面板内部状态消息转发到主窗口 (兼容旧 signal)."""
        self.status_message.emit(msg, level)

    def set_auto_running(self, running: bool, busy_text: str = "标注中…",
                         scope: str = "all"):
        """主窗口在全自动 / 批量 / LoRA 进行时调用。

        scope: "all" | "lora" | "rm" — 只禁用对应按钮组。
        """
        touch_rm = scope in ("all", "rm")
        touch_lora = scope in ("all", "lora")
        if running:
            if touch_rm:
                self.full_auto_btn.setEnabled(False)
                self.batch_auto_btn.setEnabled(False)
                self.full_auto_btn.setText(f"  {busy_text}")
            if touch_lora and hasattr(self, "lora_manual_btn"):
                self.lora_manual_btn.setEnabled(False)
                self.lora_auto_btn.setEnabled(False)
                self.lora_auto_btn.setText(f"  {busy_text}")
        else:
            if touch_rm:
                self.full_auto_btn.setEnabled(True)
                self.batch_auto_btn.setEnabled(True)
                self.full_auto_btn.setText("  全自动")
                self.batch_auto_btn.setText("  批量")
            if touch_lora and hasattr(self, "lora_manual_btn"):
                self.lora_manual_btn.setEnabled(True)
                self.lora_auto_btn.setEnabled(True)
                self.lora_auto_btn.setText("  LoRA 自动")

    def set_lora_progress(self, done: int, total: int, cls: str, placed: int):
        """更新 LoRA 面板进度文案与按钮文字。"""
        if not hasattr(self, "lora_status_label"):
            return
        if total <= 0:
            text = "LoRA 自动中…"
            btn = "  自动中…"
        elif cls:
            text = f"检测中 [{cls}] {done}/{total} · 已放置 {placed}"
            btn = f"  自动中 {done}/{total}"
        else:
            text = f"LoRA 自动中… {done}/{total} · 已放置 {placed}"
            btn = f"  自动中 {done}/{total}"
        self.lora_status_label.setText(text)
        self.lora_status_label.setProperty("role", "msg_warning")
        self._repolish(self.lora_status_label)
        if hasattr(self, "lora_auto_btn"):
            self.lora_auto_btn.setText(btn)

    def set_lora_result(self, msg: str, level: str = "info"):
        """显示 LoRA 最终结果。"""
        if not hasattr(self, "lora_status_label"):
            return
        role = {
            "info": "caption",
            "warning": "msg_warning",
            "error": "msg_error",
            "success": "msg_success",
        }.get(level, "caption")
        self.lora_status_label.setText(msg or "LoRA 就绪")
        self.lora_status_label.setProperty("role", role)
        self._repolish(self.lora_status_label)

    @staticmethod
    def _repolish(widget):
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def set_lora_enabled(self, enabled: bool):
        """LoRA-SAM 未配置时禁用按钮。"""
        if hasattr(self, "lora_manual_btn"):
            self.lora_manual_btn.setEnabled(enabled)
            self.lora_auto_btn.setEnabled(enabled)
        if hasattr(self, "lora_status_label"):
            if enabled:
                self.set_lora_result("LoRA 就绪", "info")
            else:
                self.set_lora_result("LoRA 未启用（设置 Adapter 后重启）", "warning")

    def _on_lora_manual_clicked(self):
        default = self.current_category or "B1"
        text, ok = QInputDialog.getText(
            self, "LoRA 手动标注", "输入类别名（如 B1 / R3）:",
            text=default,
        )
        if ok and text.strip():
            self.lora_manual_requested.emit(text.strip())

    def _on_lora_auto_clicked(self):
        self.lora_auto_requested.emit()

    def _on_category_right_click(self, pos):
        item = self.category_list.itemAt(pos)
        if not item:
            return
        name = item.data(Qt.UserRole)
        menu = QMenu(self)
        edit_action = QAction("编辑此类别", self)
        edit_action.setIcon(Icon.icon(Icons.GEAR, size=14, color=C.AMBER))
        edit_action.triggered.connect(lambda: self.category_edit_requested.emit(name))
        menu.addAction(edit_action)
        delete_action = QAction("删除此类别", self)
        delete_action.setIcon(Icon.icon(Icons.TRASH, size=14, color=C.DANGER))
        delete_action.triggered.connect(lambda: self.category_delete_requested.emit(name))
        menu.addAction(delete_action)
        menu.exec(self.category_list.mapToGlobal(pos))

    def refresh(self):
        self._load_categories()

    def select_prev(self):
        row = self.category_list.currentRow()
        if row > 0:
            self.category_list.setCurrentRow(row - 1)

    def select_next(self):
        row = self.category_list.currentRow()
        if row < self.category_list.count() - 1:
            self.category_list.setCurrentRow(row + 1)

    def get_current_category(self):
        return self.current_category

    def _select_category(self, name):
        for i in range(self.category_list.count()):
            if self.category_list.item(i).data(Qt.UserRole) == name:
                self.category_list.setCurrentRow(i)
                return

    def _on_instance_clicked(self, item):
        shape = item.data(Qt.UserRole)
        if shape:
            self.instance_selected.emit(shape)

    def _on_instance_right_click(self, pos):
        item = self.instance_list.itemAt(pos)
        if item:
            shape = item.data(Qt.UserRole)
            if shape:
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
