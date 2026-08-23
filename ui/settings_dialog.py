# -*- coding: utf-8 -*-
"""
参数设置对话框 — 集中管理所有可调参数.

样式走全局 ui/style.qss, 这里只做布局 / role 标注 / objectName 标记.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QFormLayout, QScrollArea, QWidget, QComboBox,
    QLineEdit, QFileDialog, QCheckBox,
)
from PySide6.QtCore import Qt, QSettings

from .icons import Icon, Icons, C


class SettingsDialog(QDialog):
    """参数设置对话框."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("ArmorLabel", "settings")
        self.original_values = {}
        self._load_settings()
        self._init_ui()

    def _load_settings(self):
        self.settings.beginGroup("Parameters")
        self._rm_vertical_angle = float(self.settings.value("RM_VERTICAL_ANGLE_THRESHOLD", 50.0))
        self._rm_parallel = float(self.settings.value("RM_PARALLEL_THRESHOLD", 45.0))
        self._rm_bar_length = float(self.settings.value("RM_BAR_LENGTH_MIN_RATIO", 0.5))
        self._rm_search_radius = float(self.settings.value("RM_SEARCH_RADIUS_RATIO", 1.5))
        self._rm_roi_padding = int(self.settings.value("RM_ROI_PADDING", 10))
        self._max_history = int(self.settings.value("MAX_HISTORY_STEPS", 20))
        self._snap_threshold = int(self.settings.value("SNAP_THRESHOLD", 8))
        self._rm_overlay_timeout = int(self.settings.value("RM_OVERLAY_TIMEOUT", 1000))
        self._yolo_conf = float(self.settings.value("YOLO_CONF", 0.25))
        self._yolo_iou = float(self.settings.value("YOLO_IOU", 0.7))
        self._yolo_imgsz = int(self.settings.value("YOLO_IMGSZ", 640))
        self._yolo_device = str(self.settings.value("YOLO_DEVICE", "cuda"))
        self._sam_device = str(self.settings.value("SAM_DEVICE", "cuda"))
        self._lora_adapter_path = str(self.settings.value("LORA_ADAPTER_PATH", ""))
        self._lora_sam3_checkpoint = str(
            self.settings.value("LORA_SAM3_CHECKPOINT", "models/sam3.pt")
        )
        self._lora_enabled = bool(self.settings.value("LORA_ENABLED", False, type=bool))
        self._mlp_roi_light_len = int(self.settings.value("MLP_ROI_LIGHT_LEN", 12))
        self._mlp_roi_persp_h = int(self.settings.value("MLP_ROI_PERSP_H", 28))
        self._mlp_roi_persp_w = int(self.settings.value("MLP_ROI_PERSP_W", 32))
        self._mlp_roi_crop_w = int(self.settings.value("MLP_ROI_CROP_W", 16))
        self._mlp_roi_expand_ratio = float(self.settings.value("MLP_ROI_EXPAND_RATIO", 1.3))
        self._mlp_roi_center_ratio = float(self.settings.value("MLP_ROI_CENTER_RATIO", 0.5))
        self.settings.endGroup()

    def _save_settings(self):
        self.settings.beginGroup("Parameters")
        self.settings.setValue("RM_VERTICAL_ANGLE_THRESHOLD", self._rm_vertical_angle)
        self.settings.setValue("RM_PARALLEL_THRESHOLD", self._rm_parallel)
        self.settings.setValue("RM_BAR_LENGTH_MIN_RATIO", self._rm_bar_length)
        self.settings.setValue("RM_SEARCH_RADIUS_RATIO", self._rm_search_radius)
        self.settings.setValue("RM_ROI_PADDING", self._rm_roi_padding)
        self.settings.setValue("MAX_HISTORY_STEPS", self._max_history)
        self.settings.setValue("SNAP_THRESHOLD", self._snap_threshold)
        self.settings.setValue("RM_OVERLAY_TIMEOUT", self._rm_overlay_timeout)
        self.settings.setValue("YOLO_CONF", self._yolo_conf)
        self.settings.setValue("YOLO_IOU", self._yolo_iou)
        self.settings.setValue("YOLO_IMGSZ", self._yolo_imgsz)
        self.settings.setValue("YOLO_DEVICE", self._yolo_device)
        self.settings.setValue("SAM_DEVICE", self._sam_device)
        self.settings.setValue("LORA_ADAPTER_PATH", self._lora_adapter_path)
        self.settings.setValue("LORA_SAM3_CHECKPOINT", self._lora_sam3_checkpoint)
        self.settings.setValue("LORA_ENABLED", self._lora_enabled)
        self.settings.setValue("MLP_ROI_LIGHT_LEN", self._mlp_roi_light_len)
        self.settings.setValue("MLP_ROI_PERSP_H", self._mlp_roi_persp_h)
        self.settings.setValue("MLP_ROI_PERSP_W", self._mlp_roi_persp_w)
        self.settings.setValue("MLP_ROI_CROP_W", self._mlp_roi_crop_w)
        self.settings.setValue("MLP_ROI_EXPAND_RATIO", self._mlp_roi_expand_ratio)
        self.settings.setValue("MLP_ROI_CENTER_RATIO", self._mlp_roi_center_ratio)
        self.settings.endGroup()
        self.settings.sync()

    def _apply_to_classes(self):
        from core.rm_armor_auto import RMArmorAuto
        RMArmorAuto.VERTICAL_ANGLE_THRESHOLD = self._rm_vertical_angle
        RMArmorAuto.PARALLEL_THRESHOLD = self._rm_parallel
        RMArmorAuto.BAR_LENGTH_MIN_RATIO = self._rm_bar_length
        RMArmorAuto.SEARCH_RADIUS_RATIO = self._rm_search_radius
        RMArmorAuto.ROI_PADDING = self._rm_roi_padding

        from core.mlp_classifier import MLPClassifier
        MLPClassifier.set_roi_params(
            light_len=self._mlp_roi_light_len,
            persp_h=self._mlp_roi_persp_h,
            persp_w=self._mlp_roi_persp_w,
            crop_w=self._mlp_roi_crop_w,
            expand_ratio=self._mlp_roi_expand_ratio,
            center_ratio=self._mlp_roi_center_ratio
        )

    def _init_ui(self):
        self.setWindowTitle("参数设置")
        self.setMinimumSize(520, 640)
        self.setWindowIcon(Icon.icon(Icons.SLIDERS, size=24, color=C.AMBER))

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 标题条
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_icon = QLabel()
        title_icon.setPixmap(Icon.pixmap(Icons.SLIDERS, size=22, color=C.AMBER))
        title_row.addWidget(title_icon)
        title = QLabel("参数设置 / PARAMETERS")
        title.setProperty("role", "panel_title")
        title_row.addWidget(title, 1)
        main_layout.addLayout(title_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)
        scroll.setWidget(scroll_widget)

        # ---- RM 装甲板 ----
        rm_group = QGroupBox("RM 装甲板半自动参数  /  ARMOR")
        rm_group.setProperty("accent", "red")
        rm_layout = QFormLayout(rm_group)
        rm_layout.setContentsMargins(10, 14, 10, 10)
        rm_layout.setSpacing(10)
        rm_layout.setLabelAlignment(Qt.AlignLeft)

        self.vertical_angle_spin = self._make_dspin(0.0, 90.0, 1, " °", self._rm_vertical_angle, 0.1)
        self.vertical_angle_spin.valueChanged.connect(lambda v: self._on_value_changed("RM_VERTICAL_ANGLE_THRESHOLD", v))
        rm_layout.addRow("灯条垂直角度阈值", self.vertical_angle_spin)

        self.parallel_spin = self._make_dspin(0.0, 90.0, 1, " °", self._rm_parallel, 0.1)
        self.parallel_spin.valueChanged.connect(lambda v: self._on_value_changed("RM_PARALLEL_THRESHOLD", v))
        rm_layout.addRow("灯条平行角度阈值", self.parallel_spin)

        self.bar_length_spin = self._make_dspin(0.01, 2.0, 2, "", self._rm_bar_length, 0.05)
        self.bar_length_spin.valueChanged.connect(lambda v: self._on_value_changed("RM_BAR_LENGTH_MIN_RATIO", v))
        rm_layout.addRow("灯条长度最小比例", self.bar_length_spin)

        self.search_radius_spin = self._make_dspin(0.5, 5.0, 1, "", self._rm_search_radius, 0.1)
        self.search_radius_spin.valueChanged.connect(lambda v: self._on_value_changed("RM_SEARCH_RADIUS_RATIO", v))
        rm_layout.addRow("搜索半径比例", self.search_radius_spin)

        self.roi_padding_spin = self._make_spin(0, 100, " px", self._rm_roi_padding)
        self.roi_padding_spin.valueChanged.connect(lambda v: self._on_value_changed("RM_ROI_PADDING", v))
        rm_layout.addRow("ROI 边距", self.roi_padding_spin)
        scroll_layout.addWidget(rm_group)

        # ---- 通用 ----
        general_group = QGroupBox("通用参数  /  GENERAL")
        general_group.setProperty("accent", "muted")
        general_layout = QFormLayout(general_group)
        general_layout.setContentsMargins(10, 14, 10, 10)
        general_layout.setSpacing(10)

        self.history_spin = self._make_spin(5, 100, "", self._max_history)
        self.history_spin.valueChanged.connect(lambda v: self._on_value_changed("MAX_HISTORY_STEPS", v))
        general_layout.addRow("撤销/重做栈深度", self.history_spin)

        self.snap_spin = self._make_spin(1, 50, " px", self._snap_threshold)
        self.snap_spin.valueChanged.connect(lambda v: self._on_value_changed("SNAP_THRESHOLD", v))
        general_layout.addRow("边缘吸附阈值", self.snap_spin)

        self.overlay_timeout_spin = self._make_spin(200, 30000, " ms", self._rm_overlay_timeout, 200)
        self.overlay_timeout_spin.valueChanged.connect(lambda v: self._on_value_changed("RM_OVERLAY_TIMEOUT", v))
        general_layout.addRow("覆盖层显示时长", self.overlay_timeout_spin)

        self.sam_device_combo = QComboBox()
        self.sam_device_combo.addItems(["cuda", "cpu", "0", "1", "2", "3"])
        self.sam_device_combo.setCurrentText(self._sam_device)
        self.sam_device_combo.currentTextChanged.connect(lambda v: self._on_value_changed("SAM_DEVICE", v))
        general_layout.addRow("SAM 设备", self.sam_device_combo)

        # LoRA-SAM（独立模块，默认不加载）
        self.lora_enabled_chk = QCheckBox("启动时加载 LoRA-SAM（默认关，开启需重启）")
        self.lora_enabled_chk.setChecked(self._lora_enabled)
        self.lora_enabled_chk.toggled.connect(
            lambda v: self._on_value_changed("LORA_ENABLED", v)
        )
        general_layout.addRow("", self.lora_enabled_chk)

        lora_hint = QLabel("开启后启动时再加载一份完整 SAM3 + adapter，内存/显存占用翻倍。")
        lora_hint.setProperty("role", "caption")
        lora_hint.setWordWrap(True)
        general_layout.addRow("", lora_hint)

        self.lora_adapter_edit = QLineEdit(self._lora_adapter_path)
        self.lora_adapter_edit.setPlaceholderText("如 sam3_rm_lora/epoch_1")
        self.lora_adapter_edit.setProperty("role", "mono")
        self.lora_adapter_edit.textChanged.connect(
            lambda v: self._on_value_changed("LORA_ADAPTER_PATH", v)
        )
        lora_adapter_row = QHBoxLayout()
        lora_adapter_row.addWidget(self.lora_adapter_edit)
        lora_browse_btn = QPushButton("浏览…")
        lora_browse_btn.setCursor(Qt.PointingHandCursor)
        lora_browse_btn.clicked.connect(self._browse_lora_adapter)
        lora_adapter_row.addWidget(lora_browse_btn)
        general_layout.addRow("LoRA Adapter 目录", lora_adapter_row)

        self.lora_ckpt_edit = QLineEdit(self._lora_sam3_checkpoint)
        self.lora_ckpt_edit.setPlaceholderText("models/sam3.pt")
        self.lora_ckpt_edit.setProperty("role", "mono")
        self.lora_ckpt_edit.textChanged.connect(
            lambda v: self._on_value_changed("LORA_SAM3_CHECKPOINT", v)
        )
        general_layout.addRow("LoRA 基础权重", self.lora_ckpt_edit)

        scroll_layout.addWidget(general_group)

        # ---- YOLO ----
        yolo_group = QGroupBox("YOLO 自动标注参数  /  YOLO")
        yolo_group.setProperty("accent", "amber")
        yolo_layout = QFormLayout(yolo_group)
        yolo_layout.setContentsMargins(10, 14, 10, 10)
        yolo_layout.setSpacing(10)

        from ui.yolo_runtime_switch import YoloRuntimeSwitch
        self.yolo_runtime_switch = YoloRuntimeSwitch(self, compact=True)
        yolo_layout.addRow("Ultralytics 版本", self.yolo_runtime_switch)

        self.yolo_conf_spin = self._make_dspin(0.01, 1.0, 2, "", self._yolo_conf, 0.05)
        self.yolo_conf_spin.valueChanged.connect(lambda v: self._on_value_changed("YOLO_CONF", v))
        yolo_layout.addRow("置信度阈值", self.yolo_conf_spin)

        self.yolo_iou_spin = self._make_dspin(0.1, 1.0, 2, "", self._yolo_iou, 0.05)
        self.yolo_iou_spin.valueChanged.connect(lambda v: self._on_value_changed("YOLO_IOU", v))
        yolo_layout.addRow("IOU 阈值", self.yolo_iou_spin)

        self.yolo_imgsz_spin = self._make_spin(320, 1280, " px", self._yolo_imgsz, 32)
        self.yolo_imgsz_spin.valueChanged.connect(lambda v: self._on_value_changed("YOLO_IMGSZ", v))
        yolo_layout.addRow("推理图片大小", self.yolo_imgsz_spin)

        self.yolo_device_combo = QComboBox()
        self.yolo_device_combo.addItems(["cuda", "cpu", "0", "1", "2", "3"])
        self.yolo_device_combo.setCurrentText(self._yolo_device)
        self.yolo_device_combo.currentTextChanged.connect(lambda v: self._on_value_changed("YOLO_DEVICE", v))
        yolo_layout.addRow("推理设备", self.yolo_device_combo)
        scroll_layout.addWidget(yolo_group)

        # ---- MLP ROI ----
        mlp_group = QGroupBox("MLP ROI 透视变换参数  /  MLP")
        mlp_group.setProperty("accent", "blue")
        mlp_layout = QFormLayout(mlp_group)
        mlp_layout.setContentsMargins(10, 14, 10, 10)
        mlp_layout.setSpacing(10)

        self.mlp_light_len_spin = self._make_spin(4, 64, " px", self._mlp_roi_light_len)
        self.mlp_light_len_spin.valueChanged.connect(lambda v: self._on_value_changed("MLP_ROI_LIGHT_LEN", v))
        mlp_layout.addRow("灯条长度 (light_len)", self.mlp_light_len_spin)

        self.mlp_persp_h_spin = self._make_spin(16, 128, " px", self._mlp_roi_persp_h)
        self.mlp_persp_h_spin.valueChanged.connect(lambda v: self._on_value_changed("MLP_ROI_PERSP_H", v))
        mlp_layout.addRow("透视高 (persp_h)", self.mlp_persp_h_spin)

        self.mlp_persp_w_spin = self._make_spin(16, 128, " px", self._mlp_roi_persp_w)
        self.mlp_persp_w_spin.valueChanged.connect(lambda v: self._on_value_changed("MLP_ROI_PERSP_W", v))
        mlp_layout.addRow("透视宽 (persp_w)", self.mlp_persp_w_spin)

        self.mlp_crop_w_spin = self._make_spin(4, 64, " px", self._mlp_roi_crop_w)
        self.mlp_crop_w_spin.valueChanged.connect(lambda v: self._on_value_changed("MLP_ROI_CROP_W", v))
        mlp_layout.addRow("裁剪宽 (crop_w)", self.mlp_crop_w_spin)

        self.mlp_expand_ratio_spin = self._make_dspin(0.5, 5.0, 2, "", self._mlp_roi_expand_ratio, 0.1)
        self.mlp_expand_ratio_spin.valueChanged.connect(lambda v: self._on_value_changed("MLP_ROI_EXPAND_RATIO", v))
        mlp_layout.addRow("bbox 扩展比例", self.mlp_expand_ratio_spin)

        self.mlp_center_ratio_spin = self._make_dspin(0.1, 1.0, 2, "", self._mlp_roi_center_ratio, 0.05)
        self.mlp_center_ratio_spin.valueChanged.connect(lambda v: self._on_value_changed("MLP_ROI_CENTER_RATIO", v))
        mlp_layout.addRow("中心保留比例", self.mlp_center_ratio_spin)
        scroll_layout.addWidget(mlp_group)

        main_layout.addWidget(scroll, 1)

        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.setProperty("role", "ghost")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("  确定")
        ok_btn.setProperty("role", "primary")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setIcon(Icon.icon(Icons.CHECK, size=14, color=C.BG_APP))
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        main_layout.addLayout(btn_layout)

    # ---------------------------------------------------------
    def _make_spin(self, lo, hi, suffix, value, step=1):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSuffix(suffix)
        s.setSingleStep(step)
        s.setValue(value)
        s.setProperty("role", "mono")
        return s

    def _make_dspin(self, lo, hi, decimals, suffix, value, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setSingleStep(step)
        s.setValue(value)
        s.setProperty("role", "mono")
        return s

    def _on_value_changed(self, key, value):
        handler = {
            "RM_VERTICAL_ANGLE_THRESHOLD": lambda v: setattr(self, "_rm_vertical_angle", v),
            "RM_PARALLEL_THRESHOLD":       lambda v: setattr(self, "_rm_parallel", v),
            "RM_BAR_LENGTH_MIN_RATIO":     lambda v: setattr(self, "_rm_bar_length", v),
            "RM_SEARCH_RADIUS_RATIO":      lambda v: setattr(self, "_rm_search_radius", v),
            "RM_ROI_PADDING":              lambda v: setattr(self, "_rm_roi_padding", v),
            "MAX_HISTORY_STEPS":           lambda v: setattr(self, "_max_history", v),
            "SNAP_THRESHOLD":              lambda v: setattr(self, "_snap_threshold", v),
            "RM_OVERLAY_TIMEOUT":          lambda v: setattr(self, "_rm_overlay_timeout", v),
            "YOLO_CONF":                   lambda v: setattr(self, "_yolo_conf", v),
            "YOLO_IOU":                    lambda v: setattr(self, "_yolo_iou", v),
            "YOLO_IMGSZ":                  lambda v: setattr(self, "_yolo_imgsz", v),
            "YOLO_DEVICE":                 lambda v: setattr(self, "_yolo_device", v),
            "SAM_DEVICE":                  lambda v: setattr(self, "_sam_device", v),
            "LORA_ADAPTER_PATH":           lambda v: setattr(self, "_lora_adapter_path", v),
            "LORA_SAM3_CHECKPOINT":        lambda v: setattr(self, "_lora_sam3_checkpoint", v),
            "LORA_ENABLED":                lambda v: setattr(self, "_lora_enabled", bool(v)),
            "MLP_ROI_LIGHT_LEN":           lambda v: setattr(self, "_mlp_roi_light_len", v),
            "MLP_ROI_PERSP_H":             lambda v: setattr(self, "_mlp_roi_persp_h", v),
            "MLP_ROI_PERSP_W":             lambda v: setattr(self, "_mlp_roi_persp_w", v),
            "MLP_ROI_CROP_W":              lambda v: setattr(self, "_mlp_roi_crop_w", v),
            "MLP_ROI_EXPAND_RATIO":        lambda v: setattr(self, "_mlp_roi_expand_ratio", v),
            "MLP_ROI_CENTER_RATIO":        lambda v: setattr(self, "_mlp_roi_center_ratio", v),
        }
        fn = handler.get(key)
        if fn:
            fn(value)

    def _browse_lora_adapter(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择 LoRA Adapter 目录", self.lora_adapter_edit.text() or "."
        )
        if path:
            self.lora_adapter_edit.setText(path)

    def _on_ok(self):
        self._apply_to_classes()
        if self.parent():
            self.parent()._rm_overlay_timeout = self._rm_overlay_timeout
        self._save_settings()
        self.accept()

    def get_max_history_steps(self):
        return self._max_history

    @staticmethod
    def load_max_history():
        s = QSettings("ArmorLabel", "settings")
        s.beginGroup("Parameters")
        val = int(s.value("MAX_HISTORY_STEPS", 20))
        s.endGroup()
        return val
