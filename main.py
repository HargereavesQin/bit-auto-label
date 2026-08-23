from core.bootstrap import suppress_third_party_warnings

suppress_third_party_warnings()

import sys
import warnings
import os
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog, QMessageBox, QLabel, QListWidgetItem, QMenu, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, QSettings, QThread, Signal
from PySide6.QtGui import QPolygonF, QAction, QFont, QPen, QBrush, QColor

from core.pose_annotation_sync import (
    snapshot_categories,
    apply_pose_config_changes_to_dir,
    describe_pose_config_changes,
)

from tools.main_dataset_tool import DatasetToolWindow
from tools.video_frame_extractor import VideoFrameExtractorWindow
from tools.armor_data_extractor import ArmorExtractorWindow
from tools.yolo_auto_annotate import YoloAutoWindow
from tools.yolo_train import YoloTrainWindow
from tools.yolo_inference import YoloInferenceWindow
from tools.mlp_train import MLPTrainWindow
from ui.main_window import Ui_MainWindow
from core.canvas import Canvas, CanvasMode
from core.sam_client import SAMClient
from core.exporter import Exporter
from core.shapes import RectShape, PolyShape, PointShape, RotatedRectShape
from core.pose_shapes import PoseShape
from core.pose_config import PoseConfig
from core.rm_armor_auto import RMArmorAuto
from core.rm_buff_merge_auto import RMBuffMergeAuto
from core.lora_sam_client import LoraSamClient
from core.lora_armor_auto import LoraArmorAuto
from core.statistics import DatasetStatistics
from core.cleaner import DatasetCleaner
from core.constants import (
    ANNO_FORMAT_JSON,
    ANNO_FORMAT_YOLO,
    DATASET_LAYOUT_FLAT,
    DATASET_LAYOUT_YOLO,
    KPT_COLORS,
)
from core.dataset_layout import (
    resolve_open_layout,
    sidecar_json_path,
    yolo_label_path_for_image,
)
from core.debug import debug_log, init_debug_mode
from ui.statistics_dialog import StatisticsDialog
from ui.cleaner_dialog import CleanerDialog
from ui.merge_class_dialog import MergeClassDialog
from ui.dataset_fusion_dialog import DatasetFusionDialog
from ui.batch_rename_dialog import BatchRenameDialog
from ui.settings_dialog import SettingsDialog
from ui.pose_config_dialog import PoseConfigDialog
from ui.pose_panel import PosePanel
from tools.mlp_demo import MLPDemoWindow
from utils.message import DialogOver


# 应用入口 — 先初始化调试模式(默认关闭, --debug 开启)
if __name__ != "__main__":
    from core.debug import init_debug_mode
    init_debug_mode()


class BuffSam3Worker(QThread):
    finished = Signal(str, list, str)

    def __init__(self, processor, image_path, onnx_path=None, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.image_path = image_path
        self.onnx_path = onnx_path

    def run(self):
        try:
            import cv2
            from buff_sam3_auto.strategy import BuffAutoStrategy
            image = cv2.imread(self.image_path)
            if image is None:
                self.finished.emit(self.image_path, [], f"读取图像失败: {self.image_path}")
                return
            strategy = BuffAutoStrategy(
                onnx_path=self.onnx_path,
                processor=self.processor,
            )
            annotations, debug = strategy.annotate(image)
            self.finished.emit(self.image_path, annotations, f"成功完成 {len(annotations)} 个能量机关五点解算")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(self.image_path, [], f"BUFF SAM3 标注过程报错: {e}")


class BuffSam3SemiWorker(QThread):
    """半自动：对单个手动拉框 ROI 做 SAM3 双轮廓 + 极轴五点解算。"""

    finished = Signal(str, list, str)  # image_path, pixel_kps, msg

    def __init__(self, processor, image_path, roi_xywh, label, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.image_path = image_path
        self.roi_xywh = roi_xywh  # (x, y, w, h) 原图像素坐标
        self.label = label

    def run(self):
        try:
            import cv2
            from buff_sam3_auto.segmenter import RSegmenter
            from buff_sam3_auto.strategy import solve_keypoints_in_roi

            image = cv2.imread(self.image_path)
            if image is None:
                self.finished.emit(self.image_path, [], f"读取图像失败: {self.image_path}")
                return
            ih, iw = image.shape[:2]
            x, y, w, h = self.roi_xywh
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(iw, int(x + w))
            y2 = min(ih, int(y + h))
            if x2 - x1 < 8 or y2 - y1 < 8:
                self.finished.emit(self.image_path, [], "拉框 ROI 过小，无法解算")
                return
            roi = image[y1:y2, x1:x2]

            segmenter = RSegmenter(device="cuda", processor=self.processor)
            local_kps, dbg = solve_keypoints_in_roi(
                segmenter, roi, is_red=(str(self.label).upper() != "BT")
            )
            if local_kps is None:
                reason = dbg.get("fail_reason", "unknown")
                self.finished.emit(self.image_path, [], f"五点解算失败: {reason}")
                return
            pixel_kps = [(kx + x1, ky + y1, v) for kx, ky, v in local_kps]
            self.finished.emit(self.image_path, pixel_kps, "半自动五点解算完成")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(self.image_path, [], f"半自动五点解算报错: {e}")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        from core.yolo_runtime import apply_runtime
        try:
            apply_runtime()
        except Exception:
            pass

        with open(os.path.join(os.path.dirname(__file__), "ui/style.qss"), "r", encoding="utf-8") as f:
            self.setStyleSheet(f.read())

        self.scene = Canvas(self)
        self.view.setScene(self.scene)

        self.current_image_path = None
        self.current_dir = None
        self.class_list = []
        self.current_format = "json"

        # SAM 加载状态（防止加载中切换图片导致崩溃）
        self.sam_loading = False
        self.lora_sam_loading = False
        self.lora_sam_client = None
        self.lora_armor_auto = None
        self._nav_status_ts = 0.0  # A/D 边界/加载提示 debounce
        self._nav_switch_ts = 0.0  # A/D 长按连切节流
        self._lora_load_token = 0
        self._sam_load_token = 0
        self._lora_image_load_count = 0
        self._sam_image_load_count = 0
        self._lora_load_watchdog = None
        self._sam_load_watchdog = None
        self._lora_suppress_finish_toast = False

        # 位姿标注相关
        self.pose_config = PoseConfig()
        self.pose_mode_active = False  # 是否处于位姿标注模式
        self.pose_drawing_bbox = False  # 是否正在绘制位姿的 bbox
        self.pose_bbox_start = None  # 位姿 bbox 起点
        self.pose_temp_rect = None  # 位姿临时矩形
        self.pose_current_shape = None  # 当前正在标注关键点的 PoseShape
        self.pose_selected_label = None  # 当前选中的位姿类别名
        self.pose_panel = None  # 位姿面板实例

        # 层级标注状态
        self.hier_config = None  # HierarchicalConfig 实例
        self.hier_panel = None  # HierarchicalPanel 实例
        self.hier_selected_label = None  # 当前选中的层级兵种名
        self.hier_mode_active = False  # 层级标注模式是否激活

        # 批量全标注状态机
        self._batch_mode = False
        self._batch_images = []  # [(path, has_annotation), ...]
        self._batch_index = 0
        self._batch_total = 0
        self._batch_policy = "skip"  # "overwrite" | "skip" | "append"
        self._coverage_mode = False  # 等效覆盖模式: 跳过 SAM 图片加载, 避免阻塞

        # 标签数量 / 类别 / 几何筛选缓存
        self._image_label_counts = {}   # {path: count}
        self._image_label_sets = {}     # {path: set[str]}
        self._image_geom = {}           # {path: geom dict}
        self._filter_mode = "all"       # "all"/"none"/"any"/"ge"/"le"/"eq"
        self._filter_value = 1          # 与 filterCountSpin 默认值对齐
        self._filter_class = ""         # "" = 全部类别
        self._filter_sort = "default"   # default/area_desc/area_asc/aspect_*
        self._filter_aspect_anomaly = False
        self._filter_aspect_thresh = 4.0
        self._async_bound_image_path = None  # Buff/RM/LoRA 异步结果绑定路径
        self._buff_semi_pending_shape = None  # Buff 半自动: 等待五点解算的 PoseShape
        self._buff_semi_worker = None

        # 数据集布局（扁平 JSON / YOLO images+labels）
        self._dataset_layout = DATASET_LAYOUT_FLAT
        self._images_dir = None
        self._labels_dir = None
        self._dataset_root = None
        self._anno_map = {}
        self._anno_format = {}
        self._yolo_label_targets = {}
        self._nav_rows = []
        self._slider_syncing = False

        self.modeLabel = QLabel("模式: 矩形标注")
        self.statusBar.addWidget(self.modeLabel)

        self.helpLabel = QLabel("状态: 正在初始化")
        self.statusBar.addWidget(self.helpLabel)

        self.sam_client = SAMClient(self)
        self.sam_client.inference_result.connect(self.scene.handle_sam_result)
        self.sam_client.text_result_ready.connect(self.handle_text_results)
        self.sam_client.model_status_changed.connect(self.update_model_status)
        self.sam_client.image_load_finished.connect(self._on_sam_image_loaded)
        self.scene.sam_client = self.sam_client

        # 撤销/重做时数据栈
        self.undo_stack = []
        self.redo_stack = []
        self._load_settings()

        self._connect_signals()
        self._set_mode(CanvasMode.RECT)
        model_path = os.path.join(os.path.dirname(__file__), "models", "sam3.pt")
        self.sam_client.load_model_async(model_path)

        # 可选: 独立 LoRA-SAM（与原 SAM 解耦，需设置 LORA_ADAPTER_PATH）
        self._init_lora_sam()

        # RM 全自动可视化覆盖层计时器
        from core.settings import Settings
        self._rm_overlay_timeout = Settings.app().get_int("Parameters/RM_OVERLAY_TIMEOUT", 1000)
        self.rm_overlay_timer = QTimer(self)
        self.rm_overlay_timer.setSingleShot(True)
        self.rm_overlay_timer.timeout.connect(self.scene.clear_rm_overlays)

        # 光条外接半自动标注
        self._rect_light_fit_active = False
        self._rect_light_fit_pending_rect = None

        # MLP分类标注
        self._current_mlp_label = None

    def _init_lora_sam(self):
        """按设置可选启动独立 LoRA-SAM（不影响原 sam_client）。

        默认不加载：LoRA-SAM 会再加载一份完整 SAM3 + adapter，与 sam_client
        并存时 RAM 峰值翻倍（30G 机器 + IDE/ROS 常驻时易触发 OOM Killed）。
        需要时在 设置→通用 勾选「启动时加载 LoRA-SAM」。
        """
        settings = QSettings("ArmorLabel", "settings")
        settings.beginGroup("Parameters")
        enabled = bool(settings.value("LORA_ENABLED", False, type=bool))
        adapter_path = str(settings.value("LORA_ADAPTER_PATH", "") or "").strip()
        ckpt = str(settings.value("LORA_SAM3_CHECKPOINT", "models/sam3.pt") or "models/sam3.pt").strip()
        settings.endGroup()
        if not enabled:
            return
        if not adapter_path:
            return
        if not os.path.isabs(adapter_path):
            adapter_path = os.path.join(os.path.dirname(__file__), adapter_path)
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(os.path.dirname(__file__), ckpt)
        if not os.path.isdir(adapter_path):
            print(f"[LoRA-SAM] adapter 目录不存在，跳过: {adapter_path}")
            return

        self.lora_sam_client = LoraSamClient(self)
        self.lora_sam_client.model_status_changed.connect(self._on_lora_model_status)
        self.lora_sam_client.image_load_finished.connect(self._on_lora_image_loaded)
        self.lora_armor_auto = LoraArmorAuto(
            self.lora_sam_client, self.pose_config, self
        )
        self.lora_armor_auto.pose_shape_placed.connect(self._on_lora_pose_placed)
        self.lora_armor_auto.auto_finished.connect(self._on_lora_auto_finished)
        self.lora_armor_auto.status_message.connect(self._on_pose_panel_status)
        self.lora_armor_auto.progress_changed.connect(self._on_lora_progress)
        self.lora_sam_client.load_model_async(ckpt, adapter_path)
        print(f"[LoRA-SAM] 正在加载 adapter: {adapter_path}")

    def _on_lora_model_status(self, success, msg):
        if success:
            self._set_status(f"LoRA-SAM: {msg}", "msg_success")
            if self.pose_panel:
                self.pose_panel.set_lora_enabled(True)
            if self.current_image_path:
                self._start_lora_image_load(self.current_image_path)
        else:
            self._set_status(f"LoRA-SAM: {msg}", "msg_error")
            if self.pose_panel:
                self.pose_panel.set_lora_enabled(False)

    def _on_lora_image_loaded(self, success, msg):
        self.lora_sam_loading = False
        self._stop_lora_load_watchdog()
        self._sync_image_nav_slider()
        if not success:
            self._set_status(f"LoRA-SAM 特征提取失败: {msg}", "msg_error")

    def _on_lora_progress(self, done, total, cls, placed):
        if self.pose_panel:
            self.pose_panel.set_lora_progress(done, total, cls, placed)

    def _on_lora_manual_requested(self, category: str):
        if not self.lora_armor_auto:
            self._set_status("未启用 LoRA-SAM（请在设置中配置 Adapter 路径并重启）", "msg_error")
            if self.pose_panel:
                self.pose_panel.set_lora_result("LoRA 未启用", "error")
            return
        if self.lora_sam_loading or not self.lora_sam_client or not self.lora_sam_client.model:
            self._set_status("LoRA-SAM 尚未就绪", "msg_warning")
            if self.pose_panel:
                self.pose_panel.set_lora_result("LoRA-SAM 尚未就绪", "warning")
            return
        if not self.pose_mode_active:
            self._set_mode(CanvasMode.POSE)
        if self.pose_panel:
            self.pose_panel.set_auto_running(True, busy_text="自动中…", scope="lora")
            self.pose_panel.set_lora_progress(0, 1, category, 0)
        self.lora_armor_auto.request_manual(category)

    def _on_lora_auto_requested(self):
        if not self.lora_armor_auto:
            self._set_status("未启用 LoRA-SAM（请在设置中配置 Adapter 路径并重启）", "msg_error")
            if self.pose_panel:
                self.pose_panel.set_lora_result("LoRA 未启用", "error")
            return
        if self.lora_sam_loading or not self.lora_sam_client or not self.lora_sam_client.model:
            self._set_status("LoRA-SAM 尚未就绪", "msg_warning")
            if self.pose_panel:
                self.pose_panel.set_lora_result("LoRA-SAM 尚未就绪", "warning")
            return
        if not self.pose_mode_active:
            self._set_mode(CanvasMode.POSE)
        # 优先用 pose_config 中的类别；否则用默认 14 类
        names = None
        if self.pose_config and self.pose_config.get_category_names():
            names = self.pose_config.get_category_names()
        total = len(names) if names else len(LoraArmorAuto.CLASS_NAMES)
        if self.pose_panel:
            self.pose_panel.set_auto_running(True, busy_text=f"自动中 0/{total}", scope="lora")
            self.pose_panel.set_lora_progress(0, total, "", 0)
        self._async_bound_image_path = self.current_image_path
        self.lora_armor_auto.request_auto(names)

    def _on_lora_pose_placed(self, category, bbox, keypoints, score):
        """LoRA 检测结果 → PoseShape + 4 关键点。"""
        if (self._async_bound_image_path
                and self._async_bound_image_path != self.current_image_path):
            return
        if not self.pose_config or not self.pose_config.has_category(category):
            self._set_status(f"LoRA 检测到未知类别: {category}, 已跳过", "msg_warning")
            return
        x, y, w, h = bbox
        shape = PoseShape(QRectF(x, y, w, h), label="")
        shape.label = category
        cat = self.pose_config.get_category(category) if self.pose_config else None
        if cat:
            shape.kpt_names = cat["keypoints"]
            shape.kpt_count = cat["kpt_count"]
        else:
            # 自动补齐 pose 类别（4 角点）
            kpt_names = ["TL", "TR", "BR", "BL"]
            if self.pose_config and not self.pose_config.has_category(category):
                self.pose_config.add_category(category, kpt_names)
            shape.kpt_names = kpt_names
            shape.kpt_count = 4
        if hasattr(shape, "update_label_text"):
            shape.update_label_text(category)
        if hasattr(shape, "update_label_position"):
            shape.update_label_position(shape)
        self.scene.addItem(shape)
        for kx, ky in keypoints:
            shape.add_keypoint(QPointF(kx, ky))
            if shape.keypoint_items:
                kpt_item = shape.keypoint_items[-1]
                kpt_item.visibility = 2
                kpt_item.setBrush(QColor(*KPT_COLORS.get(2, KPT_COLORS[2])))
        if hasattr(shape, "update_label_text"):
            shape.update_label_text(shape.label)
        if hasattr(shape, "update_label_position"):
            shape.update_label_position(shape)
        self.update_all_instances()
        self.auto_save_annotation()
        self.push_state()

    def _on_lora_auto_finished(self):
        if self.pose_panel:
            self.pose_panel.set_auto_running(False, scope="lora")
            if self.lora_armor_auto:
                msg = self.lora_armor_auto.last_result_msg or "LoRA 完成"
                level = self.lora_armor_auto.last_result_level or "info"
                self.pose_panel.set_lora_result(msg, level)
                if not getattr(self, "_lora_suppress_finish_toast", False):
                    flags = {
                        "success": "success",
                        "warning": "warning",
                        "error": "danger",
                    }.get(level, "info")
                    DialogOver(self, msg, "LoRA 标注", flags)
        self._lora_suppress_finish_toast = False

    def _exit_hier_mode(self):
        """退出层级标注模式：关闭 canvas 自动放置、隐藏面板。"""
        self.hier_mode_active = False
        self.scene.set_hierarchical_active(False)
        if self.hier_panel:
            self.hier_panel.hide()

    def _connect_signals(self):
        self.actionOpen.triggered.connect(self.open_dir)

        self.actionSettings.triggered.connect(self.open_settings)

        self.actionDatasetTool.triggered.connect(self.open_dataset_tool)
        self.actionDatasetFusion.triggered.connect(self.open_dataset_fusion)
        self.actionVideoExtractor.triggered.connect(self.open_video_extractor)
        self.actionYoloAuto.triggered.connect(self.open_yolo_auto)
        self.actionYoloTrain.triggered.connect(self.open_yolo_train)
        self.actionYoloInference.triggered.connect(self.open_yolo_inference)
        self.actionMLPTrain.triggered.connect(self.open_mlp_train)
        self.actionMobileNetTrain.triggered.connect(self.open_mobilenet_train)
        self.actionMLPDemo.triggered.connect(self.open_mlp_demo)
        self.actionArmorExtractor.triggered.connect(self.open_armor_extractor)
        self.actionStatistics.triggered.connect(self.show_statistics)
        self.actionMergeClass.triggered.connect(self.merge_categories)
        self.actionBatchRename.triggered.connect(self.batch_rename)
        self.actionHierarchicalLabel.triggered.connect(self.open_hierarchical_label)
        self.actionCleaner.triggered.connect(self.clean_unlabeled)

        self.actionRect.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.RECT))
        self.actionPoly.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.POLY))
        self.actionPoint.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.POINT))
        self.actionRBox.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.RBOX))
        self.actionPose.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.POSE))
        self.actionMLP.triggered.connect(lambda checked=False: self._set_mode(CanvasMode.MLP))

        self.samSwitch.toggled.connect(self.on_sam_toggled)

        self.samPromptBtn.clicked.connect(self.trigger_sam_prompt)
        self.samPromptInput.returnPressed.connect(self.trigger_sam_prompt)

        self.listFiles.currentItemChanged.connect(self.on_file_selected)
        self.scene.mouse_moved.connect(self.update_coordinate_label)
        self.scene.shape_drawn.connect(self.on_shape_drawn)

        self.scene.shape_double_clicked.connect(self.edit_shape_label)

        self.scene.shape_deleted.connect(self.on_shape_deleted)

        self.scene.state_changed.connect(self.on_state_changed)

        self.listClasses.itemChanged.connect(self.on_list_item_changed)
        self.listClasses.customContextMenuRequested.connect(self.on_class_right_click)

        self.instanceListWidget.itemClicked.connect(self.on_instance_item_clicked)
        self.instanceListWidget.customContextMenuRequested.connect(self.on_instance_right_click)
        self.btnAddClass.clicked.connect(self._on_add_class_btn)
        self.btnRMMode.clicked.connect(self._on_rm_mode_btn)

        self.btnHelp.clicked.connect(self.show_help_dialog)

        # 光条外接开关
        self.rectLightFitSwitch.toggled.connect(self._on_rect_light_fit_toggled)
        self.scene.shape_drawn.connect(self._on_shape_drawn_for_light_fit)

        # 筛选控件
        self.filterCombo.currentIndexChanged.connect(self._on_filter_changed)
        self.filterCountSpin.valueChanged.connect(self._on_filter_value_changed)
        self.filterClassCombo.currentIndexChanged.connect(self._on_filter_class_changed)
        self.filterSortCombo.currentIndexChanged.connect(self._on_filter_sort_changed)
        self.chkAspectAnomaly.toggled.connect(self._on_aspect_anomaly_toggled)
        self.spinAspectThresh.valueChanged.connect(self._on_aspect_thresh_changed)
        self.imageNavSlider.valueChanged.connect(self._on_image_nav_slider_changed)

    def _set_status(self, text: str, severity: str = "msg") -> None:
        """统一设置底部状态条文本 + 颜色 (走 style.qss role).

        severity ∈ {"msg", "msg_success", "msg_warning", "msg_error", "msg_active"}.
        """
        self.helpLabel.setText(text)
        self.helpLabel.setProperty("role", severity)
        # 重新刷 QSS 让 dynamic property 生效
        self.helpLabel.style().unpolish(self.helpLabel)
        self.helpLabel.style().polish(self.helpLabel)
        self.helpLabel.update()

    def get_current_class(self):
        if not self.class_list:
            self.add_class_to_list("class_0")
            self.save_classes()
            self.listClasses.setCurrentRow(0)
        current = self.listClasses.currentItem()
        if current:
            return current.data(Qt.UserRole)
        if self.listClasses.count() > 0:
            self.listClasses.setCurrentRow(0)
            return self.listClasses.item(0).data(Qt.UserRole)
        return ""

    def _on_add_class_btn(self):
        cls_name, ok = QInputDialog.getItem(self, "新建类别", "请输入类别名称:", [], 0, True)
        if ok and cls_name:
            cls_name = cls_name.strip()
            if cls_name and cls_name not in self.class_list:
                self.add_class_to_list(cls_name)
                self.save_classes()
                # 选中新添加的类别
                for i in range(self.listClasses.count()):
                    if self.listClasses.item(i).data(Qt.UserRole) == cls_name:
                        self.listClasses.setCurrentRow(i)
                        break
                self.update_all_instances()

    def _on_rm_mode_btn(self):
        """RM-mode: 一键导入 RM 标准类别 (14 类兵种 + 2 类能量机关 RT/BT)。

        - 左侧 class_list: 16 类（0:BG..13:RB 兵种, 14:RT, 15:BT 能量机关）
        - pose_config: 位姿类别，装甲板各 4 角点，能量机关各 5 点
        """
        from core.constants import HIER_CLASS_NAMES, BUFF_CLASS_NAMES
        rm_all_classes = list(HIER_CLASS_NAMES) + list(BUFF_CLASS_NAMES)
        # 确认覆盖现有类别
        if self.class_list:
            reply = QMessageBox.question(
                self, "RM-mode",
                f"将用 RM 标准类别 (14 类兵种 + 2 类能量机关 RT/BT) 覆盖当前 {len(self.class_list)} 个类别, 继续?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        # 导入 RM 标准类别 (14 类兵种 + 2 类 BUFF)
        self.class_list = list(rm_all_classes)
        self.refresh_class_list_display()
        self.save_classes()
        if getattr(self, "_dataset_root", None):
            self._write_classes_file(self._dataset_root)
        print(f"[RM-mode] 已导入 {len(self.class_list)} 类: {self.class_list}")

        # 同步 pose_config: 装甲板 14 类为 4 角点，能量机关 RT/BT 为 5 点
        rm_kpt_names = ["TL", "TR", "BL", "BR"]
        buff_kpt_names = ["R_center", "target_inner", "bottom_side", "outer_axis", "top_side"]
        self.pose_config.categories = {}
        for name in HIER_CLASS_NAMES:
            self.pose_config.add_category(name, rm_kpt_names)
        self.pose_config.add_category("RT", buff_kpt_names)
        self.pose_config.add_category("BT", buff_kpt_names)
        # 保存 pose_config
        pose_cfg_path = os.path.join(os.path.dirname(__file__), "config", "pose_classes.json")
        self.pose_config.save(pose_cfg_path)
        if self.current_dir:
            self.pose_config.save(os.path.join(self.current_dir, "pose_classes.json"))
        if getattr(self, "_dataset_root", None) and self._dataset_root != self.current_dir:
            self.pose_config.save(os.path.join(self._dataset_root, "pose_classes.json"))
        print(f"[RM-mode] pose_config 已同步 {len(self.pose_config.get_category_names())} 个位姿类别")

        # 若位姿面板已存在 (pose 模式已激活), 刷新并自动选中第一个
        if self.pose_panel:
            self.pose_panel.refresh()
            first_name = self.pose_config.get_category_names()[0] if self.pose_config.get_category_names() else None
            if first_name:
                self.pose_panel._select_category(first_name)
            print(f"[RM-mode] pose_panel 已刷新, 预选: {self.pose_selected_label}")

        if self.hier_mode_active:
            # 层级标注已激活 → 预置 BO/RO 的 L2 子类
            if not self.hier_config:
                self._ensure_hier_config()
            if not self.hier_panel:
                self._ensure_hier_panel()
            if self.hier_panel:
                self.hier_panel.refresh()
                # 预选 BO (蓝前哨) 让 L2 子类区显示出来
                self.hier_panel._select_category("BO")
                print("[RM-mode] 层级标注已激活 → 预置 BO/RO 的 L2 子类 (up/mid/down), 已预选 BO")
            self._set_status("RM-mode: 标准类别已导入 (14 类兵种 + RT/BT 能量机关)", "msg_success")
        else:
            self._set_status("RM-mode: 标准类别已导入 (0:BG…13:RB 兵种, 14:RT, 15:BT 能量机关)", "msg_success")

    def add_class_to_list(self, cls_name):
        """列表项支持双击编辑，显示索引ID"""
        if cls_name not in self.class_list:
            idx = len(self.class_list)
            self.class_list.append(cls_name)
            display_text = f"[{idx}] {cls_name}"
            item = QListWidgetItem(display_text)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(Qt.UserRole, cls_name)
            self.listClasses.addItem(item)

    def refresh_class_list_display(self):
        """刷新类别列表显示（索引变化时调用）"""
        self.listClasses.blockSignals(True)
        self.listClasses.clear()
        for i, cls_name in enumerate(self.class_list):
            display_text = f"[{i}] {cls_name}"
            item = QListWidgetItem(display_text)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(Qt.UserRole, cls_name)
            self.listClasses.addItem(item)
        self.listClasses.blockSignals(False)
        self._refresh_filter_class_combo()

    def on_class_right_click(self, pos):
        """右键点击类别列表 - 删除类别"""
        item = self.listClasses.itemAt(pos)
        if not item:
            return

        cls_name = item.data(Qt.UserRole)
        if not cls_name:
            return

        menu = QMenu(self)
        # 沿用 ui/style.qss 中的 QMenu 编辑风样式

        delete_action = QAction(f"删除类别 '{cls_name}'", self)
        menu.addAction(delete_action)

        action = menu.exec(self.listClasses.mapToGlobal(pos))
        if action == delete_action:
            self._delete_class(cls_name)

    def _delete_class(self, cls_name):
        """删除类别 - 统计并删除所有标注了该类别的标签"""
        # 统计所有图片中标注了该类别的数量
        affected_images = set()
        total_count = 0

        idx = self.class_list.index(cls_name) if cls_name in self.class_list else -1

        if self.current_dir:
            for f in os.listdir(self.current_dir):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                img_path = os.path.join(self.current_dir, f)
                base_path = os.path.splitext(img_path)[0]

                # 检查 JSON
                json_path = base_path + ".json"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                        for shape in data.get("shapes", []):
                            if shape.get("label") == cls_name:
                                total_count += 1
                                affected_images.add(img_path)
                    except Exception:
                        pass

                # 检查 YOLO
                txt_path = base_path + ".txt"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as tf:
                            for line in tf:
                                parts = line.strip().split()
                                if parts:
                                    cid = int(parts[0])
                                    if cid < len(self.class_list) and self.class_list[cid] == cls_name:
                                        total_count += 1
                                        affected_images.add(img_path)
                    except Exception:
                        pass

        # 确认对话框
        msg = f"类别: {cls_name}\n"
        msg += f"标注了该类别的标签数: {total_count}\n"
        msg += f"涉及图片数: {len(affected_images)}\n\n"
        msg += "确定要删除所有标注了该类别的标签吗？（不会删除图片）"

        reply = QMessageBox.question(self, "确认删除类别", msg,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply != QMessageBox.Yes:
            return

        # 执行删除
        removed_count = 0
        if self.current_dir:
            for f in os.listdir(self.current_dir):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                img_path = os.path.join(self.current_dir, f)
                base_path = os.path.splitext(img_path)[0]

                # 处理 JSON
                json_path = base_path + ".json"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                        original_count = len(data.get("shapes", []))
                        data["shapes"] = [s for s in data.get("shapes", []) if s.get("label") != cls_name]
                        removed = original_count - len(data["shapes"])
                        if removed > 0:
                            removed_count += removed
                            with open(json_path, 'w', encoding='utf-8') as jf:
                                json.dump(data, jf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                # 处理 YOLO
                txt_path = base_path + ".txt"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as tf:
                            lines = tf.readlines()
                        new_lines = []
                        had_target_class = False
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                cid = int(parts[0])
                                if cid < len(self.class_list) and self.class_list[cid] == cls_name:
                                    removed_count += 1
                                    had_target_class = True
                                else:
                                    if cid > idx:
                                        cid -= 1
                                    new_lines.append(f"{cid} {' '.join(parts[1:])}")
                        with open(txt_path, 'w', encoding='utf-8') as tf:
                            tf.writelines(new_lines)
                    except Exception:
                        pass

        # 从类别列表中移除
        if cls_name in self.class_list:
            self.class_list.remove(cls_name)
            self.save_classes()
            self.refresh_class_list_display()

        # 如果当前画布上有该类别的标注，也删除
        for shape in list(self.scene.items()):
            if isinstance(shape, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                if getattr(shape, 'label', '') == cls_name:
                    self.scene.removeItem(shape)

        self.auto_save_annotation()
        self.update_all_instances()
        DialogOver(self, f"已删除类别 '{cls_name}'\n共移除 {removed_count} 个标签\n涉及 {len(affected_images)} 张图片", "删除完成", "success")

    def push_state(self):
        if not self.current_image_path: return
        current_state = Exporter.extract_shapes(self.scene)

        if self.undo_stack:
            last_state = self.undo_stack[-1]
            if json.dumps(last_state, sort_keys=True) == json.dumps(current_state, sort_keys=True):
                return

        self.undo_stack.append(current_state)
        if len(self.undo_stack) > self.max_history_steps:
            self.undo_stack.pop(0)

        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 1:
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)
            previous_state = self.undo_stack[-1]
            self.restore_state(previous_state)

    def redo(self):
        if self.redo_stack:
            next_state = self.redo_stack.pop()
            self.undo_stack.append(next_state)
            self.restore_state(next_state)

    def restore_state(self, state):
        self.scene.clear_shapes()
        from core.shape_registry import get as spec_get
        for shape_data in state:
            label = shape_data.get("label", "")
            shape_type = shape_data.get("type", "")
            points = shape_data.get("points", [])

            if label and label not in self.class_list:
                self.add_class_to_list(label)
                self.save_classes()

            spec = spec_get(shape_type)
            shape = spec.from_dict(shape_data) if spec else None

            if shape:
                self.scene.addItem(shape)
                if hasattr(shape, 'update_label_text'):
                    shape.update_label_text(label)
                if hasattr(shape, 'update_label_position'):
                    shape.update_label_position(shape)
                if hasattr(shape, 'update_label_visibility'):
                    shape.update_label_visibility(shape, is_selected=False, is_hovered=False)

        self.auto_save_annotation()

    # ---------------------------------------------------------
    # Tool window launcher — 统一复用, 减 8×10=80 行重复
    # ---------------------------------------------------------
    def _open_tool_window(self, attr_name: str, factory, *factory_args):
        """通用 tool 窗口启动: 单例复用 + show/raise/activate + 异常提示."""
        try:
            if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
                setattr(self, attr_name, factory(*factory_args))
            win = getattr(self, attr_name)
            win.show()
            win.raise_()
            win.activateWindow()
        except Exception as e:
            DialogOver(self, f"启动失败: {e}", "系统错误", "danger")

    def open_dataset_tool(self):
        self._open_tool_window("dataset_window", DatasetToolWindow)

    def open_video_extractor(self):
        self._open_tool_window("_video_extractor_window", VideoFrameExtractorWindow)

    def open_armor_extractor(self):
        self._open_tool_window("_armor_extractor_window", ArmorExtractorWindow)

    def open_yolo_auto(self):
        # 需要传 current_dir, 不能走通用工厂
        try:
            if not hasattr(self, "yolo_auto_window") or self.yolo_auto_window is None:
                self.yolo_auto_window = YoloAutoWindow(self.current_dir)
            else:
                self.yolo_auto_window.current_dir = self.current_dir
                self.yolo_auto_window.edt_dir.setText(self.current_dir)
            self.yolo_auto_window.show()
            self.yolo_auto_window.raise_()
            self.yolo_auto_window.activateWindow()
        except Exception as e:
            DialogOver(self, f"启动失败: {e}", "系统错误", "danger")

    def open_yolo_train(self):
        try:
            if not hasattr(self, "yolo_train_window") or self.yolo_train_window is None:
                self.yolo_train_window = YoloTrainWindow()
                self.yolo_train_window.worker_training_started.connect(self._on_yolo_training_started)
                self.yolo_train_window.worker_training_finished.connect(self._on_yolo_training_finished)
            self.yolo_train_window.show()
            self.yolo_train_window.raise_()
            self.yolo_train_window.activateWindow()
        except Exception as e:
            DialogOver(self, f"启动失败: {e}", "系统错误", "danger")

    def open_yolo_inference(self):
        self._open_tool_window("_yolo_inference_window", YoloInferenceWindow)

    def open_mlp_train(self):
        self._open_tool_window("_mlp_train_window", MLPTrainWindow, "number_classifier")

    def open_mobilenet_train(self):
        self._open_tool_window("_mobilenet_train_window", MLPTrainWindow, "mobilenet_v3_small")

    def open_mlp_demo(self):
        self._open_tool_window("_mlp_demo_window", MLPDemoWindow)

    def open_dataset_fusion(self):
        try:
            dlg = DatasetFusionDialog(self)
            dlg.exec()
        except Exception as e:
            DialogOver(self, f"启动失败: {e}", "系统错误", "danger")

    def _on_yolo_training_started(self):
        self.sam_loading = True
        self.samSwitch.setChecked(False)
        self.samSwitch.setEnabled(False)
        self.samPromptInput.setEnabled(False)
        self.samPromptBtn.setEnabled(False)
        self.rectLightFitSwitch.setEnabled(False)
        self.sam_client.unload_model()
        self._set_status("YOLO 训练中，SAM 已释放显存", "msg_warning")
    def _on_yolo_training_finished(self):
        self.sam_loading = False
        self.samSwitch.setEnabled(True)
        self.samPromptInput.setEnabled(True)
        self.samPromptBtn.setEnabled(True)
        self.rectLightFitSwitch.setEnabled(True)
        self._set_status("YOLO 训练完成，可手动开启 SAM", "msg_success")
    def trigger_sam_prompt(self):
        if self.scene.mode == CanvasMode.POINT:
            DialogOver(self, "点标注模式下无法使用 SAM 智能提取", "提示", "warning")
            return

        prompt = self.samPromptInput.text().strip()

        if (self.pose_mode_active and self.pose_panel
                and self.pose_panel.get_auto_mode() == "buff-merge"):
            self._on_buff_sam_refine_requested(prompt or None)
            return

        if not prompt:
            return

        self.samSwitch.setChecked(True)
        self._set_status(f"正在提取提示词: {prompt}...", "msg_warning")
        self.sam_client.request_text_inference(prompt)

    def handle_text_results(self, results, prompt_text):
        buff_active = (
            hasattr(self, 'rm_buff_merge_auto')
            and self.rm_buff_merge_auto.phase != "idle"
            and self.pose_panel
            and self.pose_panel.get_auto_mode() == "buff-merge"
        )

        if not results:
            if buff_active:
                return
            if self._rect_light_fit_active:
                self._handle_rect_light_fit_results([], prompt_text)
            else:
                self._set_status(f"提取完成: 未发现关于 '{prompt_text}' 的目标", "msg_error")
            return

        if self._rect_light_fit_active:
            self._handle_rect_light_fit_results(results, prompt_text)
            return

        if buff_active:
            return

        if hasattr(self, 'rm_armor_auto') and self.rm_armor_auto._phase != "idle":
            debug_log(f"handle_text_results blocked: phase={self.rm_armor_auto._phase}")
            self.helpLabel.setText(f"装甲板检测中 ({self.rm_armor_auto._phase}) ...")
            self.helpLabel.setProperty("role", "msg_warning"); \
            self.helpLabel.style().unpolish(self.helpLabel); self.helpLabel.style().polish(self.helpLabel); self.helpLabel.update()
            return

        self.helpLabel.setText(f"提取完成: 成功抓取 {len(results)} 个 '{prompt_text}' 目标")
        self.helpLabel.setProperty("role", "msg_success"); \
        self.helpLabel.style().unpolish(self.helpLabel); self.helpLabel.style().polish(self.helpLabel); self.helpLabel.update()

        if prompt_text not in self.class_list:
            self.add_class_to_list(prompt_text)
            self.save_classes()

        for res in results:
            if self.scene.mode == CanvasMode.RECT:
                x, y, w, h = res["rect"]
                shape = RectShape(QRectF(x, y, w, h), prompt_text)
            else:
                qpts = [QPointF(p[0], p[1]) for p in res["poly_pts"]]
                shape = PolyShape(QPolygonF(qpts), prompt_text)

            self.scene.addItem(shape)
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(prompt_text)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)
            if hasattr(shape, 'update_label_visibility'):
                shape.update_label_visibility(shape, is_selected=False, is_hovered=False)

        self.auto_save_annotation()
        self.push_state()
        self.update_all_instances()

    def _on_rect_light_fit_toggled(self, checked):
        self._rect_light_fit_active = checked
        if checked:
            self._set_status("状态: 光条外接已开启，绘制矩形后自动检测", "msg_success")
        else:
            self._set_status("状态: 就绪", "msg")
            self._rect_light_fit_pending_rect = None
            self.scene.sam_enabled = self.samSwitch.isChecked()

    def _on_shape_drawn_for_light_fit(self, shape):
        if not self._rect_light_fit_active:
            return
        if self.scene.mode != CanvasMode.RECT:
            return
        from core.shapes import RectShape
        if not isinstance(shape, RectShape):
            return
        if self.sam_loading or not self.scene.sam_client or not self.scene.sam_client.model:
            return

        self._rect_light_fit_pending_rect = shape.rect()
        self.scene.sam_enabled = False
        self._set_status("正在检测光条...", "msg_warning")
        self.scene.sam_client.request_text_inference("light")

    def _handle_rect_light_fit_results(self, results, prompt_text):
        self._rect_light_fit_active = self.rectLightFitSwitch.isChecked()
        self.scene.sam_enabled = self.samSwitch.isChecked()

        if not results or not self._rect_light_fit_pending_rect:
            self._set_status("光条外接: 未检测到光条", "msg_error")
            self._rect_light_fit_pending_rect = None
            return

        from PySide6.QtCore import QRectF
        roi_rect = self._rect_light_fit_pending_rect
        roi_x1, roi_y1 = roi_rect.left(), roi_rect.top()
        roi_x2, roi_y2 = roi_rect.right(), roi_rect.bottom()

        inside_lights = []
        for res in results:
            rx, ry, rw, rh = res["rect"]
            lx1, ly1, lx2, ly2 = rx, ry, rx + rw, ry + rh
            if lx1 >= roi_x1 and ly1 >= roi_y1 and lx2 <= roi_x2 and ly2 <= roi_y2:
                inside_lights.append((lx1, ly1, lx2, ly2))

        if not inside_lights:
            self._set_status("光条外接: 框内未包含完整光条", "msg_error")
            self._rect_light_fit_pending_rect = None
            return

        new_x1 = min(l[0] for l in inside_lights)
        new_y1 = min(l[1] for l in inside_lights)
        new_x2 = max(l[2] for l in inside_lights)
        new_y2 = max(l[3] for l in inside_lights)
        new_rect = QRectF(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)

        shape = None
        for item in self.scene.items():
            from core.shapes import RectShape
            if isinstance(item, RectShape) and not getattr(item, 'is_temp', False):
                if abs(item.rect().left() - roi_x1) < 2 and abs(item.rect().top() - roi_y1) < 2:
                    shape = item
                    break

        if shape is None:
            self._set_status("光条外接: 未能找到对应矩形框", "msg_error")
            self._rect_light_fit_pending_rect = None
            return

        shape.setRect(new_rect)
        if hasattr(shape, 'update_handles_pos'):
            shape.update_handles_pos()
        if hasattr(shape, 'update_label_position'):
            shape.update_label_position(shape)
        if hasattr(shape, 'update_label_visibility'):
            shape.update_label_visibility(shape, is_selected=True, is_hovered=False)

        self._rect_light_fit_pending_rect = None
        self.helpLabel.setText(f"光条外接: 已收缩至 {len(inside_lights)} 个光条")
        self.helpLabel.setProperty("role", "msg_success"); \
        self.helpLabel.style().unpolish(self.helpLabel); self.helpLabel.style().polish(self.helpLabel); self.helpLabel.update()
        self.push_state()
        self.auto_save_annotation()
        self.update_all_instances()

    def show_help_dialog(self):
        help_text = """
        <h3>【快捷键大全】</h3>
        <ul>
            <li><b>A / 左方向键</b>：上一张图片</li>
            <li><b>D / 右方向键</b>：下一张图片</li>
            <li><b style="color:#B5756A;">Ctrl + S</b>：保存当前标注</li>
            <li><b style="color:#B5756A;">Ctrl + Z</b>：撤销</li>
            <li><b style="color:#B5756A;">Ctrl + Y</b>：重做</li>
            <li><b style="color:#C9A36B;">Q</b>：开启/关闭 SAM 智能辅助</li>
            <li><b>R</b>：切换至 矩形标注</li>
            <li><b>P</b>：切换至 多边形标注</li>
            <li><b>T</b>：切换至 点标注</li>
            <li><b>O</b>：切换至 旋转框标注</li>
            <li><b>U</b>：切换至 位姿标注</li>
            <li><b>W</b>：上一个类别 / 位姿模式下切换实例类别</li>
            <li><b>S</b>：下一个类别 / 位姿模式下切换实例类别</li>
            <li><b>Space</b>：位姿模式下激活新实例</li>
            <li><b>Escape</b>：取消当前绘制</li>
            <li><b>Del / Backspace</b>：删除当前选中的标注框</li>
            <li><b>F1</b>：打开此帮助文档</li>
        </ul>
        <hr>
        <h3>【位姿标注操作】</h3>
        <ul>
            <li>先拖拽绘制 bbox 范围</li>
            <li>然后在 bbox 内依次点击放置关键点</li>
            <li>右键关键点切换可见性（绿=可见, 橙=遮挡, 灰=未标注）</li>
        </ul>
        <hr>
        <h3>【多边形绘制技巧】</h3>
        <ul>
            <li><b>左键点击</b>：添加顶点</li>
            <li><b>Ctrl + Z</b>：撤销上一个顶点</li>
            <li><b>双击 / Enter</b>：闭合多边形</li>
        </ul>
        <hr>
        <h3>【旋转框绘制快捷键】</h3>
        <ul>
            <li><b> Z / V</b>：每次向左/向右旋转 5°</li>
            <li><b>X / C</b>：每次向左/向右旋转 1°</li>
        </ul>
        <hr>
        <h3>【SAM 智能辅助】</h3>
        <ul>
            <li><b>鼠标点选</b>：开启开关后，鼠标悬停预览，点击直接确认生成高精度轮廓。</li>
            <li><b>提示词提取</b>：在右下角输入框输入目标名称（如: dog），按回车即可一键全图抓取并打好框！左侧选中的是"矩形"还是"多边形"格式。</li>
        </ul>
        """
        QMessageBox.about(self, "使用说明", help_text)

    def update_coordinate_label(self, x, y):
        self.coordLabel.setText(f"坐标: X: {x}, Y: {y}")

    def on_sam_toggled(self, checked):
        self.scene.set_sam_enabled(checked)
        self._update_help_text(self.scene.mode)
        if checked and self.sam_client.model is None:
            model_path = os.path.join(os.path.dirname(__file__), "models", "sam3.pt")
            self.sam_client.load_model_async(model_path)
            self._set_status("正在重新加载 SAM 模型...", "msg_warning")
    def _set_mode(self, mode):
        # 退出位姿模式时清理状态
        if self.scene.mode == CanvasMode.POSE and mode != CanvasMode.POSE:
            self._exit_pose_mode()
            if self.hier_mode_active:
                self._exit_hier_mode()

        self.scene.set_mode(mode)
        mode_name = CanvasMode.get_mode_name(mode)
        self.modeLabel.setText(f"模式: {mode_name}标注")
        self._update_help_text(mode)

        if mode == CanvasMode.RECT:
            self.actionRect.setChecked(True)
        elif mode == CanvasMode.POLY:
            self.actionPoly.setChecked(True)
        elif mode == CanvasMode.POINT:
            self.actionPoint.setChecked(True)
        elif mode == CanvasMode.RBOX:
            self.actionRBox.setChecked(True)
        elif mode == CanvasMode.POSE:
            self.actionPose.setChecked(True)
            self._enter_pose_mode()
        elif mode == CanvasMode.MLP:
            self.actionMLP.setChecked(True)
            self._current_mlp_label = None
            self._load_current_mlp_label()

        if mode != CanvasMode.RECT and self.rectLightFitSwitch.isChecked():
            self.rectLightFitSwitch.setChecked(False)

        if mode in (CanvasMode.POINT, CanvasMode.MLP):
            if self.samSwitch.isChecked():
                self.samSwitch.setChecked(False)
            self.samSwitch.setEnabled(False)
            self.samPromptInput.setEnabled(False)
            self.samPromptBtn.setEnabled(False)
            if mode == CanvasMode.POINT:
                self.samPromptInput.setPlaceholderText("点标注模式下 SAM 不可用")
            else:
                self.samPromptInput.setPlaceholderText("MLP模式下 SAM 不可用")
        else:
            self.samSwitch.setEnabled(True)
            self.samPromptInput.setEnabled(True)
            self.samPromptBtn.setEnabled(True)
            self.samPromptInput.setPlaceholderText("输入提示词提取 (如: dog)")

    def _enter_pose_mode(self):
        """进入位姿标注模式"""
        self.pose_mode_active = True
        self.scene.pose_active = True
        print(f"[POSE] _enter_pose_mode: activated, hier_mode={self.hier_mode_active}")

        # 检查是否有位姿类别定义
        if not self.pose_config.get_category_names():
            # 尝试加载
            pose_config_path = os.path.join(os.path.dirname(__file__), "config", "pose_classes.json")
            self.pose_config.load(pose_config_path)

            if not self.pose_config.get_category_names():
                # 弹出定义对话框
                if not self._open_pose_config_dialog():
                    # 用户取消，切回矩形模式
                    self._set_mode(CanvasMode.RECT)
                    return

        # 与左侧 class_list 对齐（旧 pose_classes.json 常以 B1 开头、BG 垫底）
        self._sync_pose_config_to_class_list()

        # 创建/显示位姿面板
        if not self.pose_panel:
            self.pose_panel = PosePanel(self.pose_config, self)
            self.pose_panel.category_changed.connect(self._on_pose_category_changed)
            self.pose_panel.category_delete_requested.connect(self._on_pose_category_delete)
            self.pose_panel.request_new_category.connect(self._on_pose_new_category)
            self.pose_panel.category_edit_requested.connect(self._on_pose_edit_category)
            self.pose_panel.instance_selected.connect(self.on_pose_instance_selected)
            self.pose_panel.instance_delete_requested.connect(self.on_pose_instance_delete)
            # 创建RM装甲板半自动标注器
            self.rm_armor_auto = RMArmorAuto(self.sam_client, self)
            self.rm_armor_auto.keypoints_placed.connect(self._on_rm_keypoints_placed)
            self.rm_armor_auto.error_message.connect(self._on_rm_auto_error)
            self.pose_panel.rm_auto_toggled.connect(self.rm_armor_auto.set_enabled)
            self.pose_panel.rm_auto_toggled.connect(self._on_rm_auto_toggled_from_panel)
            self.pose_panel.full_auto_requested.connect(self._on_full_auto_requested)
            self.pose_panel.batch_auto_requested.connect(self._on_batch_auto_requested)
            self.pose_panel.coverage_apply_requested.connect(self._on_coverage_apply_requested)
            self.pose_panel.lora_manual_requested.connect(self._on_lora_manual_requested)
            self.pose_panel.lora_auto_requested.connect(self._on_lora_auto_requested)
            self.pose_panel.status_message.connect(self._on_pose_panel_status)
            self.pose_panel.set_lora_enabled(
                bool(self.lora_sam_client and self.lora_sam_client.model)
            )
            self.rm_armor_auto.full_auto_step_placed.connect(self._on_full_auto_step_placed)
            self.rm_armor_auto.full_auto_regions_ready.connect(self._on_full_auto_regions_ready)
            self.rm_armor_auto.full_auto_finished.connect(self._on_batch_auto_finished)
            # 任何错误或完成都解锁面板的 全自动/批量 按钮
            # 批量模式下不能逐张解锁, 否则按钮会被重新启用导致重复进入 _on_batch_auto_requested,
            # 配合 skip 策略会把已标注图过滤掉, 使 _batch_total 越来越小 (图片数量越来越少).
            self.rm_armor_auto.error_message.connect(
                lambda _msg: None if getattr(self, "_batch_mode", False)
                else self.pose_panel.set_auto_running(False, scope="rm"))
            self.rm_armor_auto.full_auto_finished.connect(
                lambda: None if getattr(self, "_batch_mode", False)
                else self.pose_panel.set_auto_running(False, scope="rm"))

            self.rm_buff_merge_auto = RMBuffMergeAuto(self.sam_client, self)
            self.rm_buff_merge_auto.yolo_rough_done.connect(self._on_buff_yolo_rough_done)
            self.rm_buff_merge_auto.keypoints_refined.connect(self._on_buff_keypoints_refined)
            self.rm_buff_merge_auto.error_message.connect(self._on_buff_auto_error)
            self.rm_buff_merge_auto.status_message.connect(self._on_buff_status_message)
            self.pose_panel.auto_mode_changed.connect(self._on_pose_auto_mode_changed)
            self.pose_panel.buff_yolo_rough_requested.connect(self._on_buff_yolo_rough_requested)
            self.pose_panel.buff_sam_refine_requested.connect(self._on_buff_sam_refine_requested)
            self.pose_panel.buff_sam3_auto_requested.connect(self._on_buff_sam3_auto_requested)
            self.pose_panel.buff_weight_select_requested.connect(self._on_buff_weight_selected)
            self.pose_panel.kpt_only_toggled.connect(self._on_pose_kpt_only_toggled)

            # 仅在面板配置文件中记录 buff 权重路径, 不在进入位姿模式时立即加载
            # buff YOLO 模型按需在 auto_mode 切到 'buff-merge' 时再加载 (见 _on_pose_auto_mode_changed).
            default_buff_weight = RMBuffMergeAuto.default_weight_path()
            if default_buff_weight:
                self.pose_panel.set_buff_weight_path(default_buff_weight)
            # 将位姿面板添加到右侧 dock 中（替换普通类别列表区域）
            self.dockLayout.addWidget(self.pose_panel, 1)
            self.pose_panel.show()
        else:
            self.pose_panel.refresh()
            self.pose_panel.show()

        self._on_pose_kpt_only_toggled(self.pose_panel.is_kpt_only())

        if self.pose_panel.get_auto_mode() == "buff-merge":
            self._on_pose_auto_mode_changed("buff-merge")

        # 隐藏普通类别列表、实例列表和 SAM 区域（类别改由位姿/层级面板管理）
        self.listClasses.hide()
        self.labelClasses.hide()
        self.btnAddClass.hide()
        self.btnRMMode.hide()
        self.labelInstances.hide()
        self.instanceListWidget.hide()
        self.labelProgress.hide()
        if self.pose_panel and self.pose_panel.get_auto_mode() == "buff-merge":
            self.samTextGroup.show()
        else:
            self.samTextGroup.hide()

        # 从面板获取当前选中的类别; 若无选中则自动选第一个 (避免 pose_selected_label=None 导致无法标注)
        self.pose_selected_label = self.pose_panel.get_current_category()
        if not self.pose_selected_label:
            names = self.pose_config.get_category_names()
            if names:
                self.pose_panel._select_category(names[0])
                self.pose_selected_label = names[0]
                print(f"[POSE] 自动预选类别: {self.pose_selected_label}")

        if self.pose_selected_label:
            cat = self.pose_config.get_category(self.pose_selected_label)
            kpt_count = cat["kpt_count"] if cat else 0
            sam_on = self.samSwitch.isChecked()
            rm_on = self.pose_panel.rm_auto_btn.isChecked() if self.pose_panel else False
            if sam_on and rm_on:
                self.helpLabel.setText(f"位姿SAM+半自动 [{self.pose_selected_label}] - 点击bbox区域自动检测 → 收缩bbox → 放关键点")
            elif sam_on:
                self.helpLabel.setText(f"位姿SAM [{self.pose_selected_label}] - 悬停预览，点击确认bbox，拖拽新建 bbox")
            elif rm_on:
                self.helpLabel.setText(f"位姿半自动 [{self.pose_selected_label}] - 手动拖拽bbox → 自动检测数字+灯条 → 收缩bbox → 放关键点")
            elif self._pose_kpt_only_enabled():
                self._set_status(f"位姿仅标点 [{self.pose_selected_label}] - 点击空白处直接添加 {kpt_count} 个关键点", "msg_warning")
            else:
                self._set_status(f"位姿 [{self.pose_selected_label}] 直接拖拽 bbox → 点击添加 {kpt_count} 个关键点", "msg_warning")
        else:
            self._set_status("位姿标注模式 - 请先在右侧面板选择位姿类别", "msg_warning")
    def _pose_kpt_only_enabled(self):
        return bool(self.pose_panel and self.pose_panel.is_kpt_only())

    def _on_pose_kpt_only_toggled(self, checked):
        self.scene.pose_kpt_only = checked
        for item in self.scene.items():
            if isinstance(item, PoseShape) and not getattr(item, "is_temp", False):
                item.set_bbox_visible(not checked)
        self._update_pose_help_text()

    def _on_pose_category_changed(self, category_name):
        """位姿面板类别选中变化"""
        self.pose_selected_label = category_name
        if category_name:
            cat = self.pose_config.get_category(category_name)
            kpt_count = cat["kpt_count"] if cat else 0
            sam_on = self.samSwitch.isChecked()
            rm_on = self.pose_panel.rm_auto_btn.isChecked() if self.pose_panel else False
            if sam_on and rm_on:
                self.helpLabel.setText(f"位姿SAM+半自动 [{category_name}] - 点击bbox区域自动检测 → 收缩bbox → 放关键点")
            elif sam_on:
                self.helpLabel.setText(f"位姿SAM [{category_name}] - 悬停预览，点击确认bbox，拖拽新建 bbox")
            elif rm_on:
                self.helpLabel.setText(f"位姿半自动 [{category_name}] - 手动拖拽bbox → 自动检测数字+灯条 → 收缩bbox → 放关键点")
            elif self._pose_kpt_only_enabled():
                self._set_status(f"位姿仅标点 [{category_name}] - 点击空白处直接添加 {kpt_count} 个关键点", "msg_warning")
            else:
                self._set_status(f"位姿标注 [{category_name}] - 拖拽 bbox，再点 {kpt_count} 个关键点", "msg_warning")
        else:
            self._set_status("位姿标注模式 - 请选择位姿类别", "msg_warning")
    def _open_pose_config_dialog(self, preselect_name=None) -> bool:
        """打开位姿类别编辑对话框，保存后同步配置与目录标注。"""
        before = snapshot_categories(self.pose_config)
        dialog = PoseConfigDialog(self.pose_config, self, preselect_name=preselect_name)
        if not dialog.exec():
            self.pose_config.categories = before
            return False

        changes = dialog.get_changes()
        has_sync = any([
            changes.get("removed"),
            changes.get("renamed"),
            changes.get("trimmed"),
            changes.get("kpt_renamed"),
        ])

        summary = None
        if has_sync and self.current_dir:
            detail = describe_pose_config_changes(changes)
            reply = QMessageBox.question(
                self, "同步标注",
                f"位姿配置已变更，是否同步当前目录内全部标注？\n\n{detail}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                summary = apply_pose_config_changes_to_dir(
                    self.current_dir, self.class_list, changes,
                )
                self._apply_pose_config_to_canvas(changes)

        global_pose_path = os.path.join(os.path.dirname(__file__), "config", "pose_classes.json")
        self.pose_config.save(global_pose_path)
        if self.current_dir:
            self.pose_config.save(os.path.join(self.current_dir, "pose_classes.json"))

        if self.pose_panel:
            self.pose_panel.refresh()

        renamed = changes.get("renamed", {})
        if self.pose_selected_label in changes.get("removed", []):
            names = self.pose_config.get_category_names()
            self.pose_selected_label = names[0] if names else None
        elif self.pose_selected_label in renamed:
            self.pose_selected_label = renamed[self.pose_selected_label]

        if summary:
            self.update_all_instances()
            if self.current_image_path:
                self.load_annotations(self.current_image_path)
            self.auto_save_annotation()
            self.push_state()
            DialogOver(
                self,
                f"已同步 {len(summary['affected_files'])} 个文件\n"
                f"删除 {summary['removed_shapes']} · "
                f"裁剪 {summary['trimmed_shapes']} · "
                f"重命名 {summary['renamed_shapes']}",
                "同步完成", "success",
            )
        elif has_sync and self.current_image_path:
            self.load_annotations(self.current_image_path)
            self.update_all_instances()

        return True

    def _apply_pose_config_to_canvas(self, changes: dict):
        """将位姿配置变更应用到当前画布上的 PoseShape。"""
        removed = set(changes.get("removed", []))
        renamed = changes.get("renamed", {})
        trimmed = changes.get("trimmed", {})
        kpt_renamed = changes.get("kpt_renamed", {})

        for shape in list(self.scene.items()):
            if not isinstance(shape, PoseShape) or getattr(shape, "is_temp", False):
                continue
            label = getattr(shape, "label", "")
            if label in removed:
                self.scene.removeItem(shape)
                continue
            if label in renamed:
                shape.label = renamed[label]
                if hasattr(shape, "update_label_text"):
                    shape.update_label_text(renamed[label])
                label = renamed[label]
            if label in trimmed:
                info = trimmed[label]
                shape.trim_keypoints_to(info["new_count"], info["keypoints"])
            elif label in kpt_renamed:
                shape.set_kpt_names(kpt_renamed[label])

    def _on_pose_new_category(self):
        """位姿面板请求新建/管理类别"""
        self._open_pose_config_dialog()

    def _on_pose_edit_category(self, category_name):
        """位姿面板请求编辑指定类别"""
        self._open_pose_config_dialog(preselect_name=category_name)

    def _on_pose_category_delete(self, category_name):
        """位姿面板请求删除类别 — 扫描全部JSON文件删除对应位姿标签"""
        affected_images = set()
        total_count = 0

        if self.current_dir:
            for f in os.listdir(self.current_dir):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                img_path = os.path.join(self.current_dir, f)
                base_path = os.path.splitext(img_path)[0]

                json_path = base_path + ".json"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                        for shape in data.get("shapes", []):
                            if shape.get("label") == category_name and shape.get("shape_type") == "pose":
                                total_count += 1
                                affected_images.add(img_path)
                    except Exception:
                        pass

                txt_path = base_path + ".txt"
                idx = self.class_list.index(category_name) if category_name in self.class_list else -1
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as tf:
                            for line in tf:
                                parts = line.strip().split()
                                if parts and len(parts) > 5 and (len(parts) - 5) % 3 == 0:
                                    cid = int(parts[0])
                                    if idx >= 0 and cid < len(self.class_list) and self.class_list[cid] == category_name:
                                        total_count += 1
                                        affected_images.add(img_path)
                    except Exception:
                        pass

        msg = f"位姿类别: {category_name}\n"
        msg += f"标注了该类别的位姿标签数: {total_count}\n"
        msg += f"涉及图片数: {len(affected_images)}\n\n"
        msg += "确定要删除所有标注了该类别的位姿标签吗？"
        reply = QMessageBox.question(self, "确认删除类别", msg,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        removed_count = 0
        if self.current_dir:
            for f in os.listdir(self.current_dir):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                img_path = os.path.join(self.current_dir, f)
                base_path = os.path.splitext(img_path)[0]

                json_path = base_path + ".json"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                        original_count = len(data.get("shapes", []))
                        data["shapes"] = [s for s in data.get("shapes", [])
                                          if not (s.get("label") == category_name and s.get("shape_type") == "pose")]
                        removed = original_count - len(data["shapes"])
                        if removed > 0:
                            removed_count += removed
                            with open(json_path, 'w', encoding='utf-8') as jf:
                                json.dump(data, jf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                txt_path = base_path + ".txt"
                idx = self.class_list.index(category_name) if category_name in self.class_list else -1
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as tf:
                            lines = tf.readlines()
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                cid = int(parts[0])
                                is_pose_line = len(parts) > 5 and (len(parts) - 5) % 3 == 0
                                if idx >= 0 and cid < len(self.class_list) and self.class_list[cid] == category_name and is_pose_line:
                                    removed_count += 1
                                else:
                                    if idx >= 0 and cid > idx and not is_pose_line:
                                        cid -= 1
                                    if is_pose_line:
                                        new_lines.append(f"{cid} " + " ".join(parts[1:]) + "\n")
                                    else:
                                        new_lines.append(line)
                        with open(txt_path, 'w', encoding='utf-8') as tf:
                            tf.writelines(new_lines)
                    except Exception:
                        pass

        # 删除画布上该类别的 PoseShape
        for shape in list(self.scene.items()):
            if isinstance(shape, PoseShape) and getattr(shape, 'label', '') == category_name and not getattr(shape, 'is_temp', False):
                self.scene.removeItem(shape)

        # 从配置中移除
        self.pose_config.remove_category(category_name)
        pose_config_path = os.path.join(os.path.dirname(__file__), "config", "pose_classes.json")
        self.pose_config.save(pose_config_path)

        # 刷新面板
        if self.pose_panel:
            self.pose_panel.refresh()

        # 更新当前选中的位姿类别
        if self.pose_selected_label == category_name:
            names = self.pose_config.get_category_names()
            self.pose_selected_label = names[0] if names else None

        self.update_all_instances()
        self.auto_save_annotation()
        self.push_state()

        DialogOver(self, f"已删除位姿类别 '{category_name}'\n共移除 {removed_count} 个标签\n涉及 {len(affected_images)} 张图片", "删除完成", "success")

    def _on_rm_keypoints_placed(self, pose_shape, keypoints):
        if not pose_shape:
            return
        for kpt in keypoints:
            x, y, v = kpt
            pose_shape.add_keypoint(QPointF(x, y))
            if pose_shape.keypoint_items:
                kpt_item = pose_shape.keypoint_items[-1]
                kpt_item.visibility = v
                kpt_item.setBrush(QColor(*KPT_COLORS.get(v, KPT_COLORS[2])))
        if hasattr(pose_shape, 'update_label_text'):
            pose_shape.update_label_text(pose_shape.label)
        if hasattr(pose_shape, 'update_label_position'):
            pose_shape.update_label_position(pose_shape)
        self.update_all_instances()
        self.auto_save_annotation()
        self.push_state()

    def _on_rm_auto_toggled_from_panel(self, checked):
        self.scene.rm_auto_mode = checked

    def _on_rm_auto_error(self, msg):
        self._set_status(f"装甲板自动标注: {msg}", "msg_error")
    def _on_pose_auto_mode_changed(self, mode):
        if not self.pose_mode_active:
            return
        if mode == "buff-merge":
            self.rm_armor_auto.set_enabled(False)
            self.rm_buff_merge_auto.set_enabled(True)
            self.samTextGroup.show()
            self.samPromptInput.setPlaceholderText("red_target / blue_target")
            self.helpLabel.setText("Buff-Merge: 点击 YOLO 粗略标注 → 输入提示词 SAM 精修")
            # 按需加载 buff YOLO 模型: 仅在切到 buff-merge 时加载,
            # 切回 armor 时不主动卸载, 避免重复加载.
            if hasattr(self, 'rm_buff_merge_auto') and self.rm_buff_merge_auto._yolo_model is None:
                weight = self.pose_panel.buff_weight_edit.text().strip() if self.pose_panel else ""
                if weight and os.path.isfile(weight):
                    self.rm_buff_merge_auto.load_yolo_async(weight)
        else:
            self.rm_buff_merge_auto.set_enabled(False)
            self.samTextGroup.hide()
            self.samPromptInput.setPlaceholderText("输入提示词提取 (如: dog)")
            rm_on = self.pose_panel.rm_auto_btn.isChecked() if self.pose_panel else False
            self.rm_armor_auto.set_enabled(rm_on)
            self._update_pose_help_text()
        self.helpLabel.setProperty("role", "msg_warning"); \
        self.helpLabel.style().unpolish(self.helpLabel); self.helpLabel.style().polish(self.helpLabel); self.helpLabel.update()

    def _on_buff_weight_selected(self, path):
        from core.settings import Settings
        Settings.app().set("Parameters/BUFF_YOLO_WEIGHT", path)
        if hasattr(self, 'rm_buff_merge_auto'):
            self.rm_buff_merge_auto.load_yolo_async(path)

    def _on_buff_yolo_rough_requested(self, prompt=None):
        if not self.pose_panel or self.pose_panel.get_auto_mode() != "buff-merge":
            return
        if not self.current_image_path:
            self._set_status("请先打开图片", "msg_error")
            return
        self.rm_buff_merge_auto.set_enabled(True)
        weight = self.pose_panel.buff_weight_edit.text().strip()
        if weight:
            self.rm_buff_merge_auto.load_yolo_async(weight)
        self._async_bound_image_path = self.current_image_path
        self.rm_buff_merge_auto.request_yolo_rough(self.current_image_path)

    def _resolve_buff_sam_prompt(self, targets, prompt=None):
        """从输入框或实例 RT/BT 标签推断 SAM 提示词。"""
        if prompt and str(prompt).strip():
            return str(prompt).strip()
        text = self.samPromptInput.text().strip()
        if text:
            return text
        if not hasattr(self, 'rm_buff_merge_auto') or not targets:
            return ""
        for shape in targets:
            label = getattr(shape, 'label', '') or ''
            resolved = self.rm_buff_merge_auto.prompt_for_label(label)
            if resolved in ("red_target", "blue_target"):
                return resolved
        label = getattr(targets[0], 'label', '') or ''
        return self.rm_buff_merge_auto.prompt_for_label(label)

    def _on_buff_sam_refine_requested(self, prompt=None):
        if not self.pose_panel or self.pose_panel.get_auto_mode() != "buff-merge":
            return
        if hasattr(self, 'rm_buff_merge_auto') and self.rm_buff_merge_auto.phase != "idle":
            self._set_status("SAM 精修进行中，请稍候...", "msg_warning")
            return
        if not self.sam_client.model:
            self._set_status("SAM 模型未就绪", "msg_error")
            return
        if self.sam_loading:
            self._set_status("SAM 正在分析图片特征，请稍候...", "msg_warning")
            return
        if not self.sam_client.inference_worker.inference_state:
            self._set_status("SAM 图像特征未就绪，请等待「分析完成」", "msg_error")
            return
        selected = [s for s in self.scene.selectedItems() if isinstance(s, PoseShape)]
        if selected:
            targets = selected
        else:
            targets = [item for item in self.scene.items()
                       if isinstance(item, PoseShape) and not getattr(item, 'is_temp', False)]
        if not targets:
            self._set_status("没有可精修的位姿实例，请先 YOLO 粗略标注", "msg_error")
            return
        prompt = self._resolve_buff_sam_prompt(targets, prompt)
        if not prompt or prompt not in ("red_target", "blue_target"):
            self._set_status(
                "请输入提示词 red_target / blue_target，或确保实例类别为 RT / BT"
            , "msg_error")
            return
        self.samPromptInput.setText(prompt)
        self.rm_buff_merge_auto.set_enabled(True)
        self.helpLabel.setText(f"SAM 精修 {len(targets)} 个实例 ({prompt})...")
        self.helpLabel.setProperty("role", "msg_warning"); \
        self.helpLabel.style().unpolish(self.helpLabel); self.helpLabel.style().polish(self.helpLabel); self.helpLabel.update()
        self.rm_buff_merge_auto.request_sam_refine(targets, prompt)

    def _on_buff_yolo_rough_done(self, image_path, detections):
        if image_path and image_path != self.current_image_path:
            self._set_status("Buff YOLO 结果已过期（已切图），已丢弃", "msg_warning")
            return
        if self._async_bound_image_path and self._async_bound_image_path != self.current_image_path:
            self._set_status("Buff YOLO 结果绑定路径已失效（已切图），已丢弃", "msg_warning")
            return
        placed = 0
        for det in detections:
            label = self.rm_buff_merge_auto.resolve_pose_label(det.get("label", ""))
            data = RMBuffMergeAuto.detection_to_pose_data(det, label)
            if not data:
                continue
            roi_rect, pose_label, kpts = data
            buff_kpt_names = ["R_center", "target_inner", "bottom_side", "outer_axis", "top_side"]
            if pose_label in ("RT", "BT") and self.pose_config.get_kpt_count(pose_label) != 5:
                self.pose_config.add_category(pose_label, buff_kpt_names)
                if self.pose_panel:
                    self.pose_panel.refresh()
            if not self.pose_config.has_category(pose_label):
                self._set_status(f"Buff YOLO 检测到未知类别: {pose_label}, 已跳过", "msg_warning")
                continue
            shape = PoseShape(roi_rect, label=pose_label)
            cat = self.pose_config.get_category(pose_label)
            if cat:
                shape.kpt_names = cat["keypoints"]
                shape.kpt_count = cat["kpt_count"]
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(pose_label)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)
            self.scene.addItem(shape)
            for kpt in kpts:
                x, y, v = kpt[0], kpt[1], kpt[2] if len(kpt) > 2 else 2
                shape.add_keypoint(QPointF(x, y))
                if shape.keypoint_items:
                    kpt_item = shape.keypoint_items[-1]
                    kpt_item.visibility = v
                    kpt_item.setBrush(QColor(*KPT_COLORS.get(v, KPT_COLORS[2])))
            placed += 1

        self.update_all_instances()
        self.auto_save_annotation()
        self.push_state()
        self._set_status(f"YOLO 粗略标注完成: {placed} 个实例，可输入提示词 SAM 精修", "msg_success")
    def _on_buff_keypoints_refined(self, pose_shape, keypoints):
        if not pose_shape or not keypoints:
            return
        while len(pose_shape.keypoint_items) > 0:
            item = pose_shape.keypoint_items.pop()
            if item.scene():
                item.scene().removeItem(item)
        for i, kpt in enumerate(keypoints):
            x, y, v = kpt[0], kpt[1], kpt[2] if len(kpt) > 2 else 2
            pose_shape.add_keypoint(QPointF(x, y))
            if pose_shape.keypoint_items:
                kpt_item = pose_shape.keypoint_items[-1]
                kpt_item.visibility = v
                kpt_item.setBrush(QColor(*KPT_COLORS.get(v, KPT_COLORS[2])))
        if hasattr(pose_shape, 'update_label_text'):
            pose_shape.update_label_text(pose_shape.label)
        if hasattr(pose_shape, 'update_label_position'):
            pose_shape.update_label_position(pose_shape)
        self.update_all_instances()
        self.auto_save_annotation()
        self.push_state()

    def _on_buff_sam3_auto_requested(self):
        if not self.current_image_path or not os.path.isfile(self.current_image_path):
            self._set_status("未打开有效的图片，无法启动 BUFF SAM3 标注", "msg_warning")
            return
        if getattr(self, "sam_loading", False) or not self.sam_client.processor:
            self._set_status("SAM3 模型尚未就绪，请等待 SAM3 加载完成", "msg_warning")
            return

        if hasattr(self, "_buff_sam3_worker") and self._buff_sam3_worker.isRunning():
            self._set_status("BUFF SAM3 自动标注正在运行中...", "msg_warning")
            return

        self._async_bound_image_path = self.current_image_path
        self._set_status("正在使用 YOLO11n ONNX + SAM3 进行能量机关全自动标注...", "msg_info")
        if self.pose_panel:
            self.pose_panel.set_auto_running(True, "SAM3 解算中…")

        from buff_sam3_auto.paths import default_onnx_path
        onnx_path = str(default_onnx_path())
        self._buff_sam3_worker = BuffSam3Worker(
            self.sam_client.processor,
            self.current_image_path,
            onnx_path=onnx_path,
            parent=self,
        )
        self._buff_sam3_worker.finished.connect(self._on_buff_sam3_auto_finished)
        self._buff_sam3_worker.start()

    def _on_buff_sam3_auto_finished(self, image_path, annotations, msg):
        if self.pose_panel:
            self.pose_panel.set_auto_running(False)

        if image_path != self.current_image_path:
            self._set_status("BUFF SAM3 结果已过期（已切换图片），已丢弃", "msg_warning")
            return
        if self._async_bound_image_path and self._async_bound_image_path != self.current_image_path:
            self._set_status("BUFF SAM3 结果绑定路径已失效（已切换图片），已丢弃", "msg_warning")
            return

        if not annotations:
            self._set_status(f"BUFF SAM3 全自动标注: {msg}", "msg_warning")
            return

        placed = 0
        buff_kpt_names = ["R_center", "target_inner", "bottom_side", "outer_axis", "top_side"]
        for ann in annotations:
            pose_label = ann.get("label", "RT")
            if pose_label in ("RT", "BT") and self.pose_config.get_kpt_count(pose_label) != 5:
                self.pose_config.add_category(pose_label, buff_kpt_names)
                if self.pose_panel:
                    self.pose_panel.refresh()
            if not self.pose_config.has_category(pose_label):
                self._set_status(f"BUFF SAM3 检测到未知类别: {pose_label}, 已跳过", "msg_warning")
                continue
            pixel_kps = ann.get("pixel_kps", [])
            roi_rect_coords = ann.get("roi_rect")
            if not pixel_kps or not roi_rect_coords:
                continue

            ox, oy, rw, rh = roi_rect_coords
            roi_rect = QRectF(ox, oy, rw, rh)

            shape = PoseShape(roi_rect, label=pose_label)
            cat = self.pose_config.get_category(pose_label)
            if cat:
                shape.kpt_names = cat["keypoints"]
                shape.kpt_count = cat["kpt_count"]
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(pose_label)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)

            self.scene.addItem(shape)
            for kpt in pixel_kps:
                x, y, v = kpt[0], kpt[1], int(kpt[2]) if len(kpt) > 2 else 2
                shape.add_keypoint(QPointF(x, y))
                if shape.keypoint_items:
                    kpt_item = shape.keypoint_items[-1]
                    kpt_item.visibility = v
                    kpt_item.setBrush(QColor(*KPT_COLORS.get(v, KPT_COLORS[2])))
            placed += 1

        if placed > 0:
            self.update_all_instances()
            self.auto_save_annotation()
            self.push_state()
            self._set_status(f"BUFF SAM3 自动标注完成: 放置了 {placed} 个能量机关五点实例", "msg_success")
        else:
            self._set_status("BUFF SAM3 未能成功解算五点关键点", "msg_warning")

    def _on_buff_auto_error(self, msg):
        self._set_status(f"Buff-Merge: {msg}", "msg_error")
    def _on_buff_status_message(self, msg):
        self._set_status(f"Buff-Merge: {msg}", "msg_warning")

    # ------------------------------------------------------------------
    # Buff 半自动: 手动拉框 → SAM3 双轮廓 → 极轴五点
    # ------------------------------------------------------------------
    def _trigger_buff_semi_auto(self, shape):
        if not self.current_image_path:
            return
        if getattr(self, "sam_loading", False) or not self.sam_client.processor:
            self._set_status("SAM3 模型尚未就绪，无法半自动解算五点", "msg_warning")
            return
        if self._buff_semi_worker is not None and self._buff_semi_worker.isRunning():
            self._set_status("Buff 半自动解算进行中，请稍候...", "msg_warning")
            return

        rect = shape.rect()
        if rect.width() < 8 or rect.height() < 8:
            self._set_status("拉框过小，无法解算五点", "msg_warning")
            return

        self._buff_semi_pending_shape = shape
        self._async_bound_image_path = self.current_image_path
        self._set_status(f"Buff 半自动: SAM3 解算 [{shape.label}] 五点中...", "msg_warning")

        self._buff_semi_worker = BuffSam3SemiWorker(
            self.sam_client.processor,
            self.current_image_path,
            (rect.x(), rect.y(), rect.width(), rect.height()),
            shape.label,
            parent=self,
        )
        self._buff_semi_worker.finished.connect(self._on_buff_semi_auto_finished)
        self._buff_semi_worker.start()

    def _on_buff_semi_auto_finished(self, image_path, pixel_kps, msg):
        shape = self._buff_semi_pending_shape
        self._buff_semi_pending_shape = None

        if image_path != self.current_image_path:
            self._set_status("Buff 半自动结果已过期（已切图），已丢弃", "msg_warning")
            return
        if (self._async_bound_image_path
                and self._async_bound_image_path != self.current_image_path):
            self._set_status("Buff 半自动结果绑定路径已失效，已丢弃", "msg_warning")
            return
        if shape is None or shape.scene() is not self.scene:
            self._set_status("Buff 半自动: 目标实例已删除，结果已丢弃", "msg_warning")
            return
        if not pixel_kps:
            self._set_status(f"Buff 半自动: {msg}", "msg_warning")
            return

        cat = self.pose_config.get_category(shape.label) if self.pose_config else None
        if cat:
            shape.kpt_names = cat["keypoints"]
            shape.kpt_count = cat["kpt_count"]
        while shape.keypoint_items:
            item = shape.keypoint_items.pop()
            if item.scene():
                item.scene().removeItem(item)
        for kpt in pixel_kps:
            x, y, v = kpt[0], kpt[1], int(kpt[2]) if len(kpt) > 2 else 2
            shape.add_keypoint(QPointF(x, y))
            if shape.keypoint_items:
                kpt_item = shape.keypoint_items[-1]
                kpt_item.visibility = v
                kpt_item.setBrush(QColor(*KPT_COLORS.get(v, KPT_COLORS[2])))
        if hasattr(shape, 'update_label_text'):
            shape.update_label_text(shape.label)
        if hasattr(shape, 'update_label_position'):
            shape.update_label_position(shape)

        self.update_all_instances()
        self.auto_save_annotation()
        self.push_state()
        self._set_status(f"Buff 半自动: [{shape.label}] 五点已标注", "msg_success")

    def _on_full_auto_requested(self):
        if not self.pose_panel or not self.pose_selected_label:
            self._set_status("请先选择位姿类别", "msg_error")
            return
        if not self.sam_client.model:
            self._set_status("SAM模型未就绪", "msg_error")
            return
        # 防止重复点击: 已经在跑就直接返回 (request_full_auto 内部也会再 guard)
        if hasattr(self, "rm_armor_auto") and self.rm_armor_auto._phase != "idle":
            self._set_status(f"全自动标注进行中 ({self.rm_armor_auto._phase}), 请稍候", "msg_warning")
            return
        self.rm_overlay_timer.stop()
        self.scene.clear_rm_overlays()
        self.pose_panel.set_auto_running(True, busy_text="标注中…", scope="rm")
        self._set_status("正在全自动检测装甲板...", "msg_warning")
        self._async_bound_image_path = self.current_image_path
        self.rm_armor_auto.request_full_auto(self.pose_selected_label)

    def _on_pose_panel_status(self, msg: str, level: str):
        """pose_panel 内部状态消息 -> 底部 status bar."""
        sev = {
            "info":    "msg",
            "warning": "msg_warning",
            "error":   "msg_error",
            "success": "msg_success",
        }.get(level, "msg")
        self._set_status(msg, sev)

    def _has_real_annotation(self, json_path):
        if not os.path.exists(json_path):
            return False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return len(data.get("shapes", [])) > 0
        except Exception:
            return False

    def _on_batch_auto_requested(self):
        # 防止批量进行中重复点击: 否则 _batch_images 会被按当前已标注状态重建,
        # 配合 skip 策略导致 _batch_total 越来越小 (图片数量越来越少).
        if getattr(self, "_batch_mode", False):
            self._set_status("批量标注进行中, 请等待完成或取消", "msg_warning")
            return
        if not self.pose_panel or not self.pose_selected_label:
            self._set_status("请先选择位姿类别", "msg_error")
            return
        if not self.sam_client.model:
            self._set_status("SAM模型未就绪", "msg_error")
            return
        if self.listFiles.count() == 0:
            self._set_status("请先打开图片目录", "msg_error")
            return

        from PySide6.QtWidgets import QInputDialog
        item, ok = QInputDialog.getItem(self, "批量全标注",
            "选择处理模式：", ["覆盖全部", "跳过已标注", "补全添加"], 0, False)
        if not ok:
            return
        if item == "覆盖全部":
            self._batch_policy = "overwrite"
        elif item == "跳过已标注":
            self._batch_policy = "skip"
        else:
            self._batch_policy = "append"

        self._batch_images = []
        for i in range(self.listFiles.count()):
            path = self.listFiles.item(i).text()
            json_path = os.path.splitext(path)[0] + ".json"
            has_annotation = self._has_real_annotation(json_path)
            self._batch_images.append((path, has_annotation, i))

        if self._batch_policy == "skip":
            self._batch_images = [(p, h, r) for p, h, r in self._batch_images if not h]

        self._batch_total = len(self._batch_images)
        self._batch_index = 0
        self._batch_mode = True

        if self._batch_total == 0:
            self._set_status("没有需要标注的图片", "msg_warning")
            self._batch_mode = False
            return

        # 批量期间禁用 全自动/批量 按钮, 防止重复触发重建 _batch_images
        if self.pose_panel:
            self.pose_panel.set_auto_running(True, busy_text="批量中…", scope="rm")

        self._set_status(f"批量标注: 0/{self._batch_total} 开始...", "msg_warning")
        from PySide6.QtWidgets import QProgressDialog
        self._batch_progress = QProgressDialog("批量标注中...", "取消标注", 0, self._batch_total, self)
        self._batch_progress.setWindowTitle("批量全标注")
        self._batch_progress.setWindowModality(Qt.WindowModal)
        self._batch_progress.setMinimumDuration(0)
        self._batch_progress.setAutoClose(True)
        self._batch_progress.setAutoReset(True)
        self._batch_progress.canceled.connect(self._on_batch_canceled)
        self._batch_progress.setValue(0)
        self._batch_progress.setLabelText(f"正在标注 (0/{self._batch_total})...")

        self._batch_process_next()

    def _batch_process_next(self):
        if self._batch_index >= self._batch_total:
            self._batch_mode = False
            if self._batch_progress:
                self._batch_progress.close()
            if self.pose_panel:
                self.pose_panel.set_auto_running(False, scope="rm")
            self._set_status(f"批量标注完成！共 {self._batch_total} 张", "msg_success")
            return

        if self._batch_progress and self._batch_progress.wasCanceled():
            self._batch_mode = False
            self._batch_progress.close()
            if self.pose_panel:
                self.pose_panel.set_auto_running(False, scope="rm")
            self._set_status("批量标注已取消", "msg_warning")
            return

        path, _, original_row = self._batch_images[self._batch_index]

        if self._batch_policy == "overwrite":
            json_path = os.path.splitext(path)[0] + ".json"
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                except Exception:
                    pass

        self.listFiles.setCurrentRow(original_row)
        current_idx = self._batch_index + 1
        filename = os.path.basename(path)
        self.labelProgress.setText(f"<b>{current_idx}/{self._batch_total}</b><br><span style='font-size:11px; color:#5A5A62;'>{filename}</span>")
        if self._batch_progress:
            self._batch_progress.setValue(self._batch_index)
            self._batch_progress.setLabelText(f"正在标注 ({current_idx}/{self._batch_total}):\n{filename}")

    def _on_batch_auto_finished(self):
        if not self._batch_mode:
            return
        self._batch_index += 1
        if self._batch_progress:
            self._batch_progress.setValue(self._batch_index)
        self._batch_process_next()

    def _on_batch_canceled(self):
        self._batch_mode = False
        if self.pose_panel:
            self.pose_panel.set_auto_running(False, scope="rm")
        if hasattr(self, "rm_armor_auto") and self.rm_armor_auto:
            self.rm_armor_auto._reset_full_auto(emit_finished=False)
            self.rm_armor_auto._reset()
        if hasattr(self, "rm_buff_merge_auto") and self.rm_buff_merge_auto:
            self.rm_buff_merge_auto._reset()
        self._async_bound_image_path = None

    # ------------------------------------------------------------------
    # 等效覆盖: 把当前图 pose 标注复制覆盖到后续所有图
    # ------------------------------------------------------------------
    def _on_coverage_apply_requested(self):
        """把当前图片的 pose 标注 (bbox + 关键点位置) 复制覆盖到后续所有图片。

        适用于固定机位/目标位置不变的连续帧 (如视频抽帧序列)。
        坐标按 scene 像素坐标直接粘贴, 要求后续图与当前图同分辨率。
        """
        from core.shape_registry import get as spec_get
        from core.constants import SHAPE_POSE, SHAPE_HIERARCHY
        from PySide6.QtWidgets import QProgressDialog

        # 1. 快照当前图的 pose/hierarchy 标注
        all_shapes = Exporter.extract_shapes(self.scene)
        snapshot = [d for d in all_shapes if d.get("type") in (SHAPE_POSE, SHAPE_HIERARCHY)]
        if not snapshot:
            self._set_status("等效覆盖: 当前图没有 pose 标注, 无法复制", "msg_warning")
            return

        cur_row = self.listFiles.currentRow()
        total = self.listFiles.count()
        remaining = total - cur_row - 1
        if remaining <= 0:
            self._set_status("等效覆盖: 当前已是最后一张, 没有后续图片", "msg_warning")
            return

        # 2. 确认
        reply = QMessageBox.question(
            self, "等效覆盖",
            f"将当前图的 {len(snapshot)} 个 pose 标注 (bbox + 关键点位置)\n"
            f"覆盖到后续 {remaining} 张图片?\n\n"
            f"注意: 坐标按像素直接粘贴, 要求后续图与当前图同分辨率。\n"
            f"目标图原有的 pose 标注会被替换, 非 pose 标注保留。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # 3. 进度对话框
        progress = QProgressDialog("等效覆盖中...", "取消", 0, remaining, self)
        progress.setWindowTitle("等效覆盖")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        source_row = cur_row
        kpt_only = self.scene.pose_kpt_only
        applied = 0
        self._coverage_mode = True  # 跳过 SAM set_image_async, 避免阻塞主线程
        try:
            for i in range(cur_row + 1, total):
                if progress.wasCanceled():
                    break
                # 切图 → on_file_selected 会保存上一张 + 加载新图标注 (跳过 SAM)
                self.listFiles.setCurrentRow(i)
                # 移除新图已有的 pose 标注 (保留非 pose 标注)
                for item in list(self.scene.items()):
                    if isinstance(item, PoseShape) and not getattr(item, 'is_temp', False):
                        self.scene.removeItem(item)
                # 粘贴快照
                for d in snapshot:
                    spec = spec_get(d.get("type"))
                    if spec is None:
                        continue
                    shape = spec.from_dict(d)
                    if shape is None:
                        continue
                    self.scene.addItem(shape)
                    if hasattr(shape, 'set_bbox_visible'):
                        shape.set_bbox_visible(not kpt_only)
                # 保存
                self.auto_save_annotation()
                applied += 1
                progress.setValue(applied)
                progress.setLabelText(f"等效覆盖 {applied}/{remaining}")
                QApplication.processEvents()
        finally:
            self._coverage_mode = False

        progress.close()

        # 4. 回到源图, 让用户看到原始标注
        self.listFiles.setCurrentRow(source_row)

        if applied > 0:
            self._set_status(
                f"等效覆盖完成: 已覆盖 {applied}/{remaining} 张, pose 标注来自图 [{source_row+1}]",
                "msg_success")
        else:
            self._set_status("等效覆盖已取消", "msg_warning")

    def _on_full_auto_step_placed(self, roi_and_category, keypoints):
        if (self._async_bound_image_path
                and self._async_bound_image_path != self.current_image_path):
            return
        roi_rect, category = roi_and_category
        if not self.pose_config or not self.pose_config.has_category(category):
            self._set_status(f"RM 全自动检测到未知类别: {category}, 已跳过", "msg_warning")
            return
        shape = PoseShape(roi_rect, label="")
        shape.label = category
        cat = self.pose_config.get_category(category)
        if cat:
            shape.kpt_names = cat["keypoints"]
            shape.kpt_count = cat["kpt_count"]
        if hasattr(shape, 'update_label_text'):
            shape.update_label_text(category)
        if hasattr(shape, 'update_label_position'):
            shape.update_label_position(shape)
        self.scene.addItem(shape)
        debug_log(f"full_auto kpt_count={shape.kpt_count} kpt_names={shape.kpt_names}")
        debug_log(f"full_auto placing {len(keypoints)} keypoints for {category}")
        for kpt in keypoints:
            x, y, v = kpt
            ok = shape.add_keypoint(QPointF(x, y))
            if shape.keypoint_items:
                kpt_item = shape.keypoint_items[-1]
                kpt_item.visibility = v
                kpt_item.setBrush(QColor(*KPT_COLORS.get(v, KPT_COLORS[2])))
        if hasattr(shape, 'update_label_text'):
            shape.update_label_text(shape.label)
        if hasattr(shape, 'update_label_position'):
            shape.update_label_position(shape)
        self.update_all_instances()
        self.auto_save_annotation()
        self.push_state()

    def _on_full_auto_regions_ready(self, regions):
        self.scene.clear_rm_overlays()
        occluded_rgb = KPT_COLORS[1]   # 橙 — 搜索范围
        visible_rgb  = KPT_COLORS[2]   # 绿 — 数字框
        for cx, cy, bw, bh, search_radius in regions:
            circle = QGraphicsEllipseItem(cx - search_radius, cy - search_radius,
                                          2 * search_radius, 2 * search_radius)
            circle.setPen(QPen(QColor(*occluded_rgb), 2, Qt.DashLine))
            circle.setBrush(QBrush(QColor(*occluded_rgb, 30)))
            circle.setZValue(9990)
            self.scene.addItem(circle)
            self.scene.rm_overlay_items.append(circle)

            rect = QGraphicsRectItem(cx - bw / 2, cy - bh / 2, bw, bh)
            rect.setPen(QPen(QColor(*visible_rgb), 2, Qt.DashLine))
            rect.setBrush(QBrush(QColor(*visible_rgb, 50)))
            rect.setZValue(9991)
            self.scene.addItem(rect)
            self.scene.rm_overlay_items.append(rect)

            label = QGraphicsTextItem("数字")
            label.setDefaultTextColor(QColor(*occluded_rgb))
            label.setFont(QFont("Arial", 12, QFont.Bold))
            label.setPos(cx - bw / 2, cy - bh / 2 - 20)
            label.setZValue(9992)
            self.scene.addItem(label)
            self.scene.rm_overlay_items.append(label)

        self.rm_overlay_timer.start(self._rm_overlay_timeout)

    def _confirm_mlp_label(self):
        label = self.get_current_class()
        if not label:
            self._set_status("请先创建一个类别", "msg_error")
            return
        self._current_mlp_label = label
        self.auto_save_annotation()
        self._set_status(f"MLP标注: {label}", "msg_success")
        self._update_mlp_status_label()

    def _load_current_mlp_label(self):
        self._current_mlp_label = None
        if not self.current_image_path:
            self._update_mlp_status_label()
            return
        base_path = os.path.splitext(self.current_image_path)[0]
        json_path = base_path + ".json"
        if os.path.exists(json_path):
            label = Exporter.load_mlp_label(json_path)
            if label:
                self._current_mlp_label = label
        self._update_mlp_status_label()

    def _update_mlp_status_label(self):
        if self.scene.mode == CanvasMode.MLP:
            if self._current_mlp_label:
                self.labelMlpStatus.setText(f"✅ MLP: {self._current_mlp_label}")
                self.labelMlpStatus.setProperty("role", "status_online")
            else:
                self.labelMlpStatus.setText("⬜ MLP: 未标注 (Enter确认)")
                self.labelMlpStatus.setProperty("role", "status_warning")
        else:
            self.labelMlpStatus.setText("")

    def _exit_pose_mode(self):
        """退出位姿标注模式"""
        self.pose_mode_active = False
        self.pose_drawing_bbox = False
        self.pose_bbox_start = None
        self.pose_current_shape = None
        self.pose_selected_label = None
        if self.pose_temp_rect and self.scene:
            try:
                self.scene.removeItem(self.pose_temp_rect)
            except Exception:
                pass
            self.pose_temp_rect = None

        # 恢复普通类别列表和 SAM 区域
        self.listClasses.show()
        self.labelClasses.show()
        self.btnAddClass.show()
        self.btnRMMode.show()
        self.labelInstances.show()
        self.instanceListWidget.show()
        self.labelProgress.show()
        self.samTextGroup.show()
        # 退出位姿后刷新，确保 [0] BG 仍在侧栏
        self.refresh_class_list_display()

        # 隐藏位姿面板
        if self.pose_panel:
            self.pose_panel.hide()

        self.update_all_instances()

    def _update_help_text(self, mode):
        is_sam = self.samSwitch.isChecked()
        if mode == CanvasMode.RECT:
            if is_sam:
                self.helpLabel.setText("操作: 鼠标悬停实时预览外接矩形，左键点击直接确认生成矩形框")
            else:
                self.helpLabel.setText("操作: 拖动鼠标绘制常规矩形")
        elif mode == CanvasMode.POLY:
            if is_sam:
                self.helpLabel.setText("操作: 鼠标悬停实时预览轮廓节点，左键点击直接确认生成多边形")
            else:
                self.helpLabel.setText("操作: 点击添加顶点，双击闭合多边形")
        elif mode == CanvasMode.POINT:
            self.helpLabel.setText("操作: 点击添加点标注")
        elif mode == CanvasMode.RBOX:
            self.helpLabel.setText("操作: 拖动绘制旋转框，Z/X/C/V调整角度")
        elif mode == CanvasMode.POSE:
            if self.pose_selected_label:
                cat = self.pose_config.get_category(self.pose_selected_label)
                kpt_count = cat["kpt_count"] if cat else 0
                sam_on = self.samSwitch.isChecked()
                rm_on = self.pose_panel.rm_auto_btn.isChecked() if self.pose_panel else False
                if sam_on and rm_on:
                    self.helpLabel.setText(f"位姿SAM+半自动 [{self.pose_selected_label}] - 点击bbox区域自动检测 → 收缩bbox → 放关键点")
                elif sam_on:
                    self.helpLabel.setText(f"位姿SAM [{self.pose_selected_label}] - 悬停预览，点击确认bbox，拖拽新建 bbox")
                elif rm_on:
                    self.helpLabel.setText(f"位姿半自动 [{self.pose_selected_label}] - 手动拖拽bbox → 自动检测数字+灯条 → 收缩bbox → 放关键点")
                else:
                    self.helpLabel.setText(f"位姿标注 [{self.pose_selected_label}] - 拖拽 bbox，再点 {kpt_count} 个关键点")
            else:
                self.helpLabel.setText("位姿标注模式 - 请先在右侧面板选择位姿类别")

    def load_classes(self, dir_path):
        self.class_list.clear()
        self.listClasses.clear()
        class_file = os.path.join(dir_path, "classes.txt")
        if not os.path.exists(class_file):
            fallback = os.path.join(os.path.dirname(__file__), "config", "classes.txt")
            if os.path.exists(fallback):
                class_file = fallback
        if os.path.exists(class_file):
            with open(class_file, "r", encoding="utf-8") as f:
                for line in f:
                    cls_name = line.strip()
                    # 跳过 "0: red_armor" 这类 yaml 风格行，只取纯类名
                    if ":" in cls_name and cls_name.split(":", 1)[0].strip().isdigit():
                        cls_name = cls_name.split(":", 1)[1].strip()
                    if cls_name and cls_name not in self.class_list:
                        self.add_class_to_list(cls_name)

        # 自动从 JSON 标注文件中提取不在 classes.txt 中的类别名
        for f in os.listdir(dir_path):
            if not f.lower().endswith('.json'):
                continue
            json_path = os.path.join(dir_path, f)
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                for shape in data.get('shapes', []):
                    label = shape.get('label', '')
                    if label and label not in self.class_list:
                        self.add_class_to_list(label)
            except Exception:
                pass

        if self.class_list:
            self.save_classes()

    def save_classes(self):
        """写 classes.txt：优先数据集根，保证 YOLO 布局下 BG 等不会丢。"""
        targets = []
        if getattr(self, "_dataset_root", None):
            targets.append(self._dataset_root)
        if self.current_dir and self.current_dir not in targets:
            targets.append(self.current_dir)
        for directory in targets:
            if not directory or not os.path.isdir(directory):
                continue
            class_file = os.path.join(directory, "classes.txt")
            with open(class_file, "w", encoding="utf-8") as file_handle:
                for class_name in self.class_list:
                    file_handle.write(class_name + "\n")

    def _align_class_list_with_yolo_ids(self):
        """把 class_list 补齐到 YOLO txt 实际最大 class_id（占位名 "0"/"1"/…）。

        只读内存不写盘；class_list 已足够长时无操作。
        """
        max_id = -1
        for image_path, anno_format in self._anno_format.items():
            if anno_format != ANNO_FORMAT_YOLO:
                continue
            txt_path = self._anno_map.get(image_path)
            if not txt_path or not os.path.isfile(txt_path):
                continue
            try:
                with open(txt_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            continue
                        if class_id > max_id:
                            max_id = class_id
            except Exception:
                continue
        if max_id >= len(self.class_list):
            self.class_list.extend(str(i) for i in range(len(self.class_list), max_id + 1))
            self.refresh_class_list_display()

    def _sync_pose_config_to_class_list(self, write_global: bool = False):
        """按 class_list 顺序重排/补齐 pose_config，保证与 [0]=BG 的类索引一致。

        默认只写当前数据集的 pose_classes.json，不覆盖全局 config/pose_classes.json。
        （打开 red_armor/blue_armor 等旧数据集时写全局，会把 RM 14 类配置冲掉。）
        """
        if not self.class_list:
            return
        old = dict(self.pose_config.categories)
        default_kpts = ["TL", "TR", "BL", "BR"]
        buff_kpts = ["R_center", "target_inner", "bottom_side", "outer_axis", "top_side"]
        new = {}
        for name in self.class_list:
            if name in old:
                # 能量机关 RT/BT 必须纠正为 5 关键点，避免被旧 4 关键点配置冲掉
                if name in ("RT", "BT") and old[name].get("kpt_count") != 5:
                    new[name] = {"kpt_count": len(buff_kpts), "keypoints": list(buff_kpts)}
                else:
                    new[name] = old[name]
            elif name in ("RT", "BT"):
                new[name] = {"kpt_count": len(buff_kpts), "keypoints": list(buff_kpts)}
            else:
                new[name] = {"kpt_count": len(default_kpts), "keypoints": list(default_kpts)}
        # 仅保留 class_list 内的类别，避免把旧数据集的 red_armor 等垫在 14 类后面
        if list(new.keys()) == list(old.keys()) and set(new) == set(old):
            return
        self.pose_config.categories = new
        roots = []
        if getattr(self, "_dataset_root", None):
            roots.append(self._dataset_root)
        if self.current_dir and self.current_dir not in roots:
            roots.append(self.current_dir)
        for root in roots:
            if root and os.path.isdir(root):
                self.pose_config.save(os.path.join(root, "pose_classes.json"))
        if write_global:
            pose_cfg_path = os.path.join(os.path.dirname(__file__), "config", "pose_classes.json")
            self.pose_config.save(pose_cfg_path)
        if self.pose_panel:
            self.pose_panel.refresh()

    SHAPE_TYPE_NAMES = {
        "RectShape": "矩形",
        "PolyShape": "多边形",
        "PointShape": "点",
        "RotatedRectShape": "旋转框",
        "PoseShape": "位姿",
    }

    def update_all_instances(self):
        all_shapes = [item for item in self.scene.items()
                      if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape))
                      and not getattr(item, 'is_temp', False)]

        self.instanceListWidget.blockSignals(True)
        self.instanceListWidget.clear()
        for i, shape in enumerate(reversed(all_shapes)):
            label = getattr(shape, 'label', '') or '未命名'
            type_name = self.SHAPE_TYPE_NAMES.get(type(shape).__name__, type(shape).__name__)

            if isinstance(shape, PoseShape):
                kpt_done = len(shape.keypoint_items)
                kpt_total = getattr(shape, 'kpt_count', 0)
                display = f"#{i+1} {label} ({type_name} {kpt_done}/{kpt_total})"
            else:
                display = f"#{i+1} {label} ({type_name})"

            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, shape)
            self.instanceListWidget.addItem(item)
        self.instanceListWidget.blockSignals(False)

        if self.pose_panel:
            self.update_pose_instance_list()

    def on_instance_item_clicked(self, item):
        shape = item.data(Qt.UserRole)
        if shape:
            for sel in self.scene.selectedItems():
                sel.setSelected(False)
            shape.setSelected(True)

    def on_instance_right_click(self, pos):
        item = self.instanceListWidget.itemAt(pos)
        if item:
            shape = item.data(Qt.UserRole)
            if shape:
                from PySide6.QtWidgets import QMenu
                menu = QMenu(self)
                # 沿用 ui/style.qss 中的 QMenu 编辑风样式
                delete_action = menu.addAction("🗑️ 删除此标注")
                action = menu.exec(self.instanceListWidget.mapToGlobal(pos))
                if action == delete_action:
                    if shape in self.scene.items():
                        self.scene.removeItem(shape)
                        self.auto_save_annotation()
                        self.push_state()
                        self.update_all_instances()

    def handle_new_shape(self, shape):
        # 位姿模式下的形状 - 自动设置标签和关键点名称，不弹窗
        if isinstance(shape, PoseShape):
            print(f"[POSE] handle_new_shape: hier_mode={self.hier_mode_active} "
                  f"hier_label={self.hier_selected_label!r} pose_label={self.pose_selected_label!r}")
            # 层级标注优先：使用 hier_selected_label + hierarchical_classes.json 的 kpt 配置
            if self.hier_mode_active and self.hier_selected_label and self.hier_config:
                shape.label = self.hier_selected_label
                shape.is_hierarchical = True
                # 有 L2 的 BO/RO → 5 点；RG/B1/… → 仅 4 角点
                kpt_count, kpt_names = self.hier_config.get_kpt_spec_for_class(
                    self.hier_selected_label
                )
                shape.kpt_names = list(kpt_names)
                shape.kpt_count = kpt_count
                if hasattr(shape, 'update_label_text'):
                    shape.update_label_text(self.hier_selected_label)
                if hasattr(shape, 'update_label_position'):
                    shape.update_label_position(shape)
                self.scene.addItem(shape)
                print(f"[POSE]   → hierarchy shape added: label={shape.label} kpt_count={shape.kpt_count}")
                self.push_state()
                return
            if not self.pose_selected_label:
                print(f"[POSE]   → DROPPED: no pose category selected (pose_selected_label is None)")
                return
            shape.label = self.pose_selected_label
            cat = self.pose_config.get_category(self.pose_selected_label)
            if cat:
                shape.kpt_names = cat["keypoints"]
                shape.kpt_count = cat["kpt_count"]
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(self.pose_selected_label)
            if hasattr(shape, 'update_label_position'):
                shape.update_label_position(shape)
            self.scene.addItem(shape)
            print(f"[POSE]   → pose shape added: label={shape.label} kpt_count={shape.kpt_count}")
            self.push_state()
            if (hasattr(self, 'rm_armor_auto') and self.rm_armor_auto.enabled
                    and self.pose_panel and self.pose_panel.get_auto_mode() == "armor"
                    and not self.scene.pose_kpt_only):
                self.rm_armor_auto.request_auto_keypoints(shape)
            elif (self.pose_panel and self.pose_panel.get_auto_mode() == "buff-merge"
                    and self.pose_panel.is_buff_semi_auto()
                    and not self.scene.pose_kpt_only):
                self._trigger_buff_semi_auto(shape)
            return

        # 非位姿模式：使用当前选中的类别，不弹窗
        cls_name = self.get_current_class()
        if not cls_name:
            return

        self.scene.addItem(shape)

        shape.label = cls_name
        if hasattr(shape, 'update_label_text'):
            shape.update_label_text(cls_name)
        if hasattr(shape, 'update_label_position'):
            shape.update_label_position(shape)
        if hasattr(shape, 'update_label_visibility'):
            shape.update_label_visibility(shape, is_selected=True, is_hovered=False)
        for item in self.scene.selectedItems():
            item.setSelected(False)
        shape.setSelected(True)
        self.push_state()

    def edit_shape_label(self, shape):
        current_label = shape.label
        default_idx = self.class_list.index(current_label) if current_label in self.class_list else 0

        cls_name, ok = QInputDialog.getItem(self, "修改类别", "请重新选择或输入类别名称:", self.class_list, default_idx,
                                            True)

        if ok and cls_name:
            cls_name = cls_name.strip()
            if cls_name not in self.class_list:
                self.add_class_to_list(cls_name)
                self.save_classes()

            shape.label = cls_name
            if hasattr(shape, 'update_label_text'):
                shape.update_label_text(cls_name)

            self.auto_save_annotation()
            self.push_state()
            self.update_all_instances()

    def on_shape_deleted(self):
        self.auto_save_annotation()
        self.push_state()
        self.update_all_instances()

    def on_shape_drawn(self, shape):
        self.handle_new_shape(shape)
        self.auto_save_annotation()
        self.update_all_instances()

    def on_state_changed(self):
        self.push_state()
        self.update_all_instances()
        self._update_pose_help_text()
        # 关键点拖拽 / 右键改 vis（含分层 KP4）需落盘
        self.auto_save_annotation()

    def update_pose_instance_list(self):
        if not self.pose_panel and not (self.hier_panel and self.hier_mode_active):
            return
        pose_shapes = [item for item in self.scene.items()
                       if isinstance(item, PoseShape) and not getattr(item, 'is_temp', False)]
        instances = []
        for i, shape in enumerate(reversed(pose_shapes)):
            label = getattr(shape, 'label', '') or '未命名'
            kpt_done = len(shape.keypoint_items)
            kpt_total = getattr(shape, 'kpt_count', 0)
            instances.append({
                "index": i + 1,
                "label": label,
                "kpt_done": kpt_done,
                "kpt_total": kpt_total,
                "shape": shape,
            })
        self.pose_panel.update_instance_list(instances)
        if self.hier_panel and self.hier_mode_active:
            self.hier_panel.update_instance_list(instances)

    def _update_pose_help_text(self):
        if not self.pose_mode_active or not self.pose_selected_label:
            return
        cat = self.pose_config.get_category(self.pose_selected_label)
        kpt_count = cat["kpt_count"] if cat else 0
        incomplete = None
        for item in self.scene.items():
            if isinstance(item, PoseShape) and not getattr(item, 'is_temp', False):
                if not item.is_kpt_complete():
                    incomplete = item
                    break
        sam_on = self.samSwitch.isChecked()
        rm_on = (self.pose_panel.rm_auto_btn.isChecked()
                 if self.pose_panel and self.pose_panel.get_auto_mode() == "armor" else False)
        buff_on = self.pose_panel.get_auto_mode() == "buff-merge" if self.pose_panel else False
        if incomplete:
            done = len(incomplete.keypoint_items)
            if buff_on:
                self.helpLabel.setText(
                    f"Buff-Merge [{self.pose_selected_label}] YOLO标注/SAM精修 ({done}/{kpt_count})"
                )
            elif sam_on and rm_on:
                self.helpLabel.setText(f"位姿SAM+半自动 [{self.pose_selected_label}] 点击bbox添加关键点 ({done}/{kpt_count})")
            elif self._pose_kpt_only_enabled():
                self.helpLabel.setText(f"位姿仅标点 [{self.pose_selected_label}] 点击添加关键点 ({done}/{kpt_count})")
            else:
                self.helpLabel.setText(f"位姿 [{self.pose_selected_label}] 点击添加关键点 ({done}/{kpt_count})，拖拽新建实例")
        else:
            if self._pose_kpt_only_enabled():
                self._set_status(f"位姿仅标点 [{self.pose_selected_label}] 所有实例已完成，点击空白处新建", "msg_warning")
            else:
                self._set_status(f"位姿 [{self.pose_selected_label}] 所有实例已完成，拖拽新建实例", "msg_warning")
    def on_pose_instance_selected(self, shape):
        for item in self.scene.selectedItems():
            item.setSelected(False)
        shape.setSelected(True)

    def on_pose_instance_delete(self, shape):
        if shape in self.scene.items():
            self.scene.removeItem(shape)
            self.auto_save_annotation()
            self.push_state()
            self.update_all_instances()

    def _switch_pose_category(self, pose_shape, direction):
        category_names = self.pose_config.get_category_names()
        if not category_names:
            return
        current_label = getattr(pose_shape, 'label', '')
        current_kpt_count = pose_shape.kpt_count
        try:
            idx = list(category_names).index(current_label)
        except ValueError:
            idx = 0
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(category_names):
            return
        new_label = category_names[new_idx]
        new_cat = self.pose_config.get_category(new_label)
        new_kpt_count = new_cat['kpt_count'] if new_cat else 0
        if new_kpt_count != current_kpt_count:
            DialogOver(self, f"无法切换：当前实例有 {current_kpt_count} 个关键点，目标类别 '{new_label}' 有 {new_kpt_count} 个关键点", "关键点数量不同", "warning")
            return
        pose_shape.label = new_label
        pose_shape.kpt_names = new_cat['keypoints'] if new_cat else []
        if hasattr(pose_shape, 'update_label_text'):
            pose_shape.update_label_text(new_label)
        if hasattr(pose_shape, 'update_label_position'):
            pose_shape.update_label_position(pose_shape)
        self.auto_save_annotation()
        self.push_state()
        self.update_all_instances()

    def on_list_item_changed(self, item):
        new_name = item.text().strip()
        import re
        new_name = re.sub(r'^\[\d+\]\s*', '', new_name)
        old_name = item.data(Qt.UserRole)

        if not old_name or new_name == old_name:
            return

        self.listClasses.blockSignals(True)
        try:
            if not new_name:
                DialogOver(self, "类别名不能为空！", "名称错误", "warning")
                item.setText(old_name)
                return

            if new_name in self.class_list:
                DialogOver(self, f"类别名 '{new_name}' 已存在！", "名称冲突", "warning")
                item.setText(old_name)
                return

            idx = self.class_list.index(old_name)
            self.class_list[idx] = new_name
            item.setData(Qt.UserRole, new_name)

            changed = False
            for shape in self.scene.items():
                if isinstance(shape, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                    if getattr(shape, 'label', '') == old_name:
                        shape.label = new_name
                        if hasattr(shape, 'update_label_text'):
                            shape.update_label_text(new_name)
                        changed = True

            self.save_classes()
            self.refresh_class_list_display()
            if changed:
                self.auto_save_annotation()
                self.push_state()
                self.update_all_instances()

            DialogOver(self, f"已将所有的 '{old_name}' 批量变更为 '{new_name}'", "修改成功", "success")

        finally:
            self.listClasses.blockSignals(False)

    def open_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if not dir_path:
            return
        layout = resolve_open_layout(dir_path)
        self.current_dir = dir_path
        self._dataset_layout = layout.layout
        self._images_dir = layout.images_dir
        self._labels_dir = layout.labels_dir
        self._dataset_root = layout.dataset_root
        self._anno_map = dict(layout.annotation_map)
        self._anno_format = dict(layout.annotation_format)
        self._yolo_label_targets = dict(layout.yolo_label_targets)

        # 阻断 clear 触发的 currentItemChanged：否则会对上一数据集的当前图
        # 执行 _save_annotation_for_path，把旧图键写进新 _anno_map（陈旧键污染），
        # 且用新数据集的 class_list 误存旧图（YOLO 格式下会清空旧图标注）。
        self.listFiles.blockSignals(True)
        self.listFiles.clear()
        self.listFiles.blockSignals(False)
        class_root = layout.dataset_root
        if layout.layout == DATASET_LAYOUT_YOLO and layout.class_names:
            self.class_list = list(layout.class_names)
            self.refresh_class_list_display()
            self._write_classes_file(class_root)
        else:
            self.load_classes(class_root)

        # 无 classes.txt（或过短）的 YOLO txt 数据集：按 txt 实际最大 class_id
        # 补 "0"/"1"/… 占位类名，保证 id↔index 对齐；否则 _load_yolo 会整行丢弃，
        # 切图保存时把原 txt 静默清空，且标签数量/类标/有无标签筛选全部失效。
        self._align_class_list_with_yolo_ids()

        pose_config_path = os.path.join(class_root, "pose_classes.json")
        self.pose_config.load(pose_config_path)
        # 旧数据集常把 BG 漏掉或追加在末尾；按 class_list（[0]=BG）重排
        self._sync_pose_config_to_class_list()

        for image_path in layout.image_paths:
            self.listFiles.addItem(image_path)

        if self.listFiles.count() > 0:
            self.listFiles.setCurrentRow(0)

        layout_label = "YOLO" if layout.layout == DATASET_LAYOUT_YOLO else "JSON/扁平"
        self._set_status(
            f"已打开 {len(layout.image_paths)} 张 ({layout_label})",
            "msg_success",
        )
        self._compute_label_counts()
        self._refresh_filter_class_combo()
        self._update_filter_status()
        self._sync_image_nav_slider()

    def _write_classes_file(self, directory: str):
        if not directory or not self.class_list:
            return
        class_file = os.path.join(directory, "classes.txt")
        with open(class_file, "w", encoding="utf-8") as file_handle:
            for class_name in self.class_list:
                file_handle.write(class_name + "\n")

    def update_model_status(self, success, msg):
        self.helpLabel.setText(msg)
        if success:
            self.helpLabel.setProperty("role", "msg_success"); \
            self.helpLabel.style().unpolish(self.helpLabel); self.helpLabel.style().polish(self.helpLabel); self.helpLabel.update()
            if self.current_image_path:
                self._set_status("模型已就绪，正在自动分析当前图片特征...", "msg_warning")
                self._start_sam_image_load(self.current_image_path)
        else:
            self.helpLabel.setProperty("role", "msg_error"); \
            self.helpLabel.style().unpolish(self.helpLabel); self.helpLabel.style().polish(self.helpLabel); self.helpLabel.update()

    def _on_sam_image_loaded(self, success, msg):
        self.sam_loading = False
        self._stop_sam_load_watchdog()
        self._sync_image_nav_slider()
        if success:
            if self._batch_mode:
                if self._batch_progress and self._batch_progress.wasCanceled():
                    self._batch_mode = False
                    return
                self._set_status("批量标注中，分析完成，开始检测...", "msg_warning")
                self.rm_armor_auto.request_full_auto(self.pose_selected_label)
            else:
                self._set_status("分析完成，可以开始智能标注", "msg_success")
        else:
            self._set_status(f"图片特征提取失败: {msg}", "msg_error")
            if self._batch_mode:
                self._batch_index += 1
                self._batch_process_next()

    def on_file_selected(self, current, previous):
        if previous:
            self._save_annotation_for_path(previous.text())

        if current:
            # 切图时中断进行中的 LoRA / RM / Buff，避免旧图回调污染
            if self.lora_armor_auto and self.lora_armor_auto.is_busy:
                self._lora_suppress_finish_toast = True
                self.lora_armor_auto.cancel()
            if hasattr(self, "rm_armor_auto") and self.rm_armor_auto:
                self.rm_armor_auto._reset_full_auto(emit_finished=False)
                if getattr(self.rm_armor_auto, "_phase", "idle") != "idle":
                    self.rm_armor_auto._reset()
            if hasattr(self, "rm_buff_merge_auto") and self.rm_buff_merge_auto:
                self.rm_buff_merge_auto._reset()
            if hasattr(self, "_buff_sam3_worker") and self._buff_sam3_worker.isRunning():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        self._buff_sam3_worker.finished.disconnect(self._on_buff_sam3_auto_finished)
                except Exception:
                    pass
            if self._buff_semi_worker is not None and self._buff_semi_worker.isRunning():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        self._buff_semi_worker.finished.disconnect(self._on_buff_semi_auto_finished)
                except Exception:
                    pass
            self._buff_semi_pending_shape = None
            self._async_bound_image_path = None

            self.rm_overlay_timer.stop()
            path = current.text()
            self.current_image_path = path
            self.scene.load_image(path)
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.load_annotations(path)
            self._load_current_mlp_label()

            self.undo_stack.clear()
            self.redo_stack.clear()
            self.push_state()

            # 层级标注模式: 切图时重置 L2 子类为 none, 避免上一张的子类标签残留
            if self.hier_mode_active and self.hier_panel:
                self.hier_panel.reset_l2_for_new_image()

            # 刷新实例列表
            self.update_all_instances()

            # 更新进度显示
            self._update_current_label_count()
            self._sync_image_nav_slider()

            if self.sam_client.model and not self._coverage_mode:
                self._set_status("正在分析图片智能特征...", "msg_warning")
                self._start_sam_image_load(path)
            elif not self.sam_client.model:
                self._set_status("等待后台加载模型，稍后将自动分析图片...", "msg_warning")

            if (self.lora_sam_client and self.lora_sam_client.model
                    and not self._coverage_mode):
                self._start_lora_image_load(path)

    def _resolve_annotation_save_path(self, image_path: str) -> tuple[str, str]:
        json_path = sidecar_json_path(image_path)
        existing_format = self._anno_format.get(image_path)
        if existing_format == ANNO_FORMAT_JSON or os.path.isfile(json_path):
            return ANNO_FORMAT_JSON, json_path
        if self._dataset_layout == DATASET_LAYOUT_YOLO:
            yolo_path = self._yolo_label_targets.get(image_path)
            if not yolo_path and self._labels_dir and self._images_dir:
                yolo_path = yolo_label_path_for_image(
                    image_path, self._images_dir, self._labels_dir
                )
            if yolo_path:
                label_dir = os.path.dirname(yolo_path)
                if label_dir:
                    os.makedirs(label_dir, exist_ok=True)
                return ANNO_FORMAT_YOLO, yolo_path
        return ANNO_FORMAT_JSON, json_path

    def _save_annotation_for_path(self, image_path):
        """将当前 scene 标注保存到指定图片路径（仅当 path 仍是当前图时落盘）。"""
        if not image_path or not self.scene.img_item:
            return
        if image_path != self.current_image_path:
            return
        shapes_data = Exporter.extract_shapes(self.scene)
        img_rect = self.scene.img_item.pixmap().rect()

        # 自动补全 class_list：确保新画的或自动标注出的 label (如 RT/BT) 包含在 self.class_list 中，
        # 避免在 save_yolo 时因为 label 不在 self.class_list 中而被过滤掉导致 .txt 不写入
        class_added = False
        for s in shapes_data:
            lbl = s.get("label", "").strip()
            if lbl and lbl not in self.class_list:
                self.add_class_to_list(lbl)
                class_added = True
        if class_added:
            self.save_classes()

        save_format, out_path = self._resolve_annotation_save_path(image_path)
        try:
            if save_format == ANNO_FORMAT_YOLO:
                Exporter.save_yolo(
                    out_path, img_rect.width(), img_rect.height(),
                    shapes_data, self.class_list,
                )
            else:
                Exporter.save_json(
                    out_path, image_path, img_rect.width(), img_rect.height(),
                    shapes_data, mlp_label=self._current_mlp_label,
                )
        except Exception as error:
            debug_log(f"自动保存失败: {str(error)}")

        self._anno_map[image_path] = out_path
        self._anno_format[image_path] = save_format
        if save_format == ANNO_FORMAT_YOLO:
            self._yolo_label_targets[image_path] = out_path

        label_names = [
            shape.get("label", "").strip()
            for shape in shapes_data
            if shape.get("label", "").strip()
        ]
        self._image_label_counts[image_path] = len(label_names)
        self._image_label_sets[image_path] = set(label_names)
        self._image_geom[image_path] = self._geom_from_shapes_data(shapes_data)
        self._update_filter_status()
        self._update_current_label_count()

    def auto_save_annotation(self):
        if not self.current_image_path or not self.scene.img_item:
            return
        self._save_annotation_for_path(self.current_image_path)

    def load_annotations(self, image_path):
        if not self.scene.img_item:
            return
        img_rect = self.scene.img_item.pixmap().rect()
        image_width = img_rect.width()
        image_height = img_rect.height()

        json_path = sidecar_json_path(image_path)
        if os.path.isfile(json_path):
            self._load_json(json_path)
            return

        annotation_path = self._anno_map.get(image_path)
        annotation_format = self._anno_format.get(image_path)
        if annotation_format == ANNO_FORMAT_YOLO and annotation_path:
            self._load_yolo(annotation_path, image_width, image_height)
            return
        if annotation_path and os.path.isfile(annotation_path):
            if annotation_path.lower().endswith(".txt"):
                self._load_yolo(annotation_path, image_width, image_height)
            elif annotation_path.lower().endswith(".json"):
                self._load_json(annotation_path)

    def _add_shape_to_scene(self, shape, label):
        if label not in self.class_list:
            self.add_class_to_list(label)
            self.save_classes()
        self.scene.addItem(shape)

    def _load_json(self, json_path):
        if not os.path.exists(json_path): return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            from core.shape_registry import get as spec_get
            for shape_data in data.get("shapes", []):
                label = shape_data.get("label", "")
                shape_type = shape_data.get("shape_type", "") or self._infer_shape_type_from_points(shape_data.get("points", []))
                spec = spec_get(shape_type)
                shape = spec.from_dict(shape_data) if spec else None
                if shape:
                    self._add_shape_to_scene(shape, label)
                    if getattr(shape, 'update_label_text', None):
                        shape.update_label_text(label)
        except Exception as e:
            debug_log(f"加载 JSON 标注失败: {e}")

    @staticmethod
    def _infer_shape_type_from_points(points: list) -> str:
        n = len(points)
        if n == 1:
            return "point"
        if n == 2:
            return "rectangle"
        return "polygon"

    def _load_yolo(self, txt_path, img_w, img_h):
        if not os.path.exists(txt_path): return
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    class_id = int(parts[0])
                    if class_id >= len(self.class_list): continue
                    label = self.class_list[class_id]

                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (cx - w / 2) * img_w
                    y1 = (cy - h / 2) * img_h
                    x2 = (cx + w / 2) * img_w
                    y2 = (cy + h / 2) * img_h

                    # 判断是否为 YOLO-pose 格式（有额外的关键点数据）
                    if len(parts) > 5 and (len(parts) - 5) % 3 == 0:
                        kpt_count = (len(parts) - 5) // 3
                        kpts = []
                        kpt_names = []
                        for i in range(kpt_count):
                            kx = float(parts[5 + i * 3]) * img_w
                            ky = float(parts[6 + i * 3]) * img_h
                            kv = int(float(parts[7 + i * 3]))
                            kpts.append([kx, ky, kv])
                            kpt_names.append(f"kpt_{i}")
                        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                        shape = PoseShape(rect, label, kpt_names=kpt_names, keypoints=kpts)
                    else:
                        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                        shape = RectShape(rect, label)

                    self._add_shape_to_scene(shape, label)
        except Exception as e:
            debug_log(f"加载 YOLO 标注失败: {e}")

    def _load_xml(self, xml_path):
        if not os.path.exists(xml_path): return
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.find("name").text if obj.find("name") is not None else ""
                bndbox = obj.find("bndbox")
                if bndbox is None: continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # 检查是否有 keypoints
                kpts_node = obj.find("keypoints")
                if kpts_node is not None:
                    kpts = []
                    kpt_names = []
                    for kpt_elem in kpts_node.findall("keypoint"):
                        kx = float(kpt_elem.find("x").text)
                        ky = float(kpt_elem.find("y").text)
                        kv = int(kpt_elem.find("visibility").text) if kpt_elem.find("visibility") is not None else 2
                        kn = kpt_elem.find("name").text if kpt_elem.find("name") is not None else f"kpt_{len(kpts)}"
                        kpts.append([kx, ky, kv])
                        kpt_names.append(kn)
                    rect = QRectF(xmin, ymin, xmax - xmin, ymax - ymin)
                    shape = PoseShape(rect, name, kpt_names=kpt_names, keypoints=kpts)
                else:
                    rect = QRectF(xmin, ymin, xmax - xmin, ymax - ymin)
                    shape = RectShape(rect, name)

                self._add_shape_to_scene(shape, name)
        except Exception as e:
            debug_log(f"加载 XML 标注失败: {e}")

    def _load_settings(self):
        from core.settings import get_rm_params, get_editor_params
        from core.constants import RM_VERTICAL_ANGLE_THRESHOLD, RM_PARALLEL_THRESHOLD, RM_BAR_LENGTH_MIN_RATIO, RM_SEARCH_RADIUS_RATIO, RM_ROI_PADDING
        self.max_history_steps = get_editor_params()["max_history"]
        rm = get_rm_params()
        RMArmorAuto.VERTICAL_ANGLE_THRESHOLD = rm["vertical_angle"]
        RMArmorAuto.PARALLEL_THRESHOLD = rm["parallel"]
        RMArmorAuto.BAR_LENGTH_MIN_RATIO = rm["bar_length"]
        RMArmorAuto.SEARCH_RADIUS_RATIO = rm["search_radius"]
        RMArmorAuto.ROI_PADDING = rm["roi_padding"]
        from core.mlp_classifier import MLPClassifier
        MLPClassifier.load_roi_params_from_settings()

    def open_settings(self):
        from core.mlp_classifier import MLPClassifier
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.max_history_steps = dialog.get_max_history_steps()
            MLPClassifier.load_roi_params_from_settings()

    def show_statistics(self):
        if not self.current_dir and not self._anno_map and self.listFiles.count() == 0:
            DialogOver(self, "请先打开图片目录", "提示", "warning")
            return

        image_files = [
            self.listFiles.item(i).text()
            for i in range(self.listFiles.count())
            if self.listFiles.item(i)
        ]
        # 优先用 open_dir 建立的映射（YOLO: images/ ↔ labels/），避免扁平旁路扫描漏标
        if self._anno_map:
            report = DatasetStatistics.generate_report(
                self.current_dir or getattr(self, "_dataset_root", "") or "",
                self.class_list,
                annotation_map=self._anno_map,
                image_files=image_files or None,
            )
        else:
            report = DatasetStatistics.generate_report(self.current_dir, self.class_list)

        current_image_report = None
        if self.current_image_path:
            current_image_report = DatasetStatistics.generate_current_image_report(self.scene, self.class_list)
        current_row = self.listFiles.currentRow()
        total_count = self.listFiles.count()
        current_filename = os.path.basename(self.current_image_path) if self.current_image_path else ""
        dialog = StatisticsDialog(report, current_image_report, self,
                                 current_row=current_row,
                                 total_count=total_count,
                                 current_filename=current_filename,
                                 main_window=self)
        dialog.exec()

    def merge_categories(self):
        if not self.current_dir:
            DialogOver(self, "请先打开图片目录", "提示", "warning")
            return
        dialog = MergeClassDialog(self.class_list, self.pose_config, self)
        if dialog.exec():
            source_classes, target_class = dialog.get_merge_info()
            if not source_classes or not target_class:
                return
            self._do_merge(source_classes, target_class)

    def _do_merge(self, source_classes, target_class):
        renamed_count = 0
        for filename in os.listdir(self.current_dir):
            if filename.lower().endswith('.json'):
                anno_path = os.path.join(self.current_dir, filename)
                if self._merge_json(anno_path, source_classes, target_class):
                    renamed_count += 1
            elif filename.lower().endswith('.txt') and not filename.endswith('classes.txt'):
                anno_path = os.path.join(self.current_dir, filename)
                if self._merge_yolo(anno_path, source_classes, target_class):
                    renamed_count += 1
            elif filename.lower().endswith('.xml'):
                anno_path = os.path.join(self.current_dir, filename)
                if self._merge_xml(anno_path, source_classes, target_class):
                    renamed_count += 1

        for src_cls in source_classes:
            if src_cls in self.class_list:
                self.class_list.remove(src_cls)
            self.pose_config.remove_category(src_cls)

        self.save_classes()
        self.pose_config.save(os.path.join(self.current_dir, "pose_classes.json"))
        self.refresh_class_list_display()

        self._refresh_current_scene_labels(source_classes, target_class)

        self.listFiles.clear()
        for f in sorted(os.listdir(self.current_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.listFiles.addItem(os.path.join(self.current_dir, f))

        DialogOver(self, f"合并完成！\n已处理 {renamed_count} 个标注文件\n已移除类别: {', '.join(source_classes)}", "提示", "info")

        self._compute_label_counts()
        self._refresh_filter_class_combo()
        self._update_filter_status()

    def _merge_json(self, anno_path, source_classes, target_class):
        try:
            with open(anno_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            changed = False
            target_kpt_names = None
            if target_class in self.pose_config.categories:
                target_kpt_names = self.pose_config.categories[target_class].get("keypoints", [])

            for shape in data.get("shapes", []):
                if shape.get("label") in source_classes:
                    shape["label"] = target_class
                    changed = True
                    if "keypoints" in shape and target_kpt_names:
                        shape["kpt_names"] = target_kpt_names

            if changed:
                with open(anno_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return changed
        except Exception:
            return False

    def _merge_yolo(self, anno_path, source_classes, target_class):
        try:
            classes_path = os.path.join(self.current_dir, "classes.txt")
            if not os.path.exists(classes_path):
                return False

            with open(classes_path, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]

            if target_class not in classes:
                return False

            target_id = classes.index(target_class)
            source_ids = [classes.index(c) for c in source_classes if c in classes]

            with open(anno_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            changed = False
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in source_ids:
                        parts[0] = str(target_id)
                        changed = True
                    new_lines.append(' '.join(parts))

            if changed:
                with open(anno_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
            return changed
        except Exception:
            return False

    def _merge_xml(self, anno_path, source_classes, target_class):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(anno_path)
            root = tree.getroot()
            changed = False

            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is not None and name_elem.text in source_classes:
                    name_elem.text = target_class
                    changed = True

            if changed:
                tree.write(anno_path, encoding='utf-8', xml_declaration=True)
            return changed
        except Exception:
            return False

    def _refresh_current_scene_labels(self, source_classes, target_class):
        for shape in self.scene.items():
            if hasattr(shape, 'label') and shape.label in source_classes:
                shape.label = target_class
                if hasattr(shape, 'update_label_text'):
                    shape.update_label_text(target_class)
                if hasattr(shape, 'kpt_names') and target_class in self.pose_config.categories:
                    shape.kpt_names = self.pose_config.categories[target_class].get("keypoints", [])
        self.scene.update()

    def batch_rename(self):
        if not self.current_dir:
            DialogOver(self, "请先打开图片目录", "提示", "warning")
            return
        file_list = []
        for i in range(self.listFiles.count()):
            path = self.listFiles.item(i).text()
            base_name = os.path.splitext(os.path.basename(path))[0]
            file_list.append((path, base_name))
        dlg = BatchRenameDialog(self.current_dir, file_list, self)
        if dlg.exec():
            # dlg.exec() 返回时文件已改名；此处 auto_save 会把 scene 写回旧路径
            # （复活刚被 rename 的旧文件名），故不再调用。
            self._reload_after_fs_change()
            DialogOver(self, "重命名完成！", "提示", "info")

    def _reload_after_fs_change(self):
        """文件系统层面增删/重命名后，重建布局映射、文件列表与筛选缓存。

        旧 _anno_map 的键是旧文件名，直接 _compute_label_counts 会让缓存键
        全部失效（_image_matches_filter 默认 0）→ 数量/类标/有无标签筛选全错。
        """
        layout = resolve_open_layout(self.current_dir)
        self._dataset_layout = layout.layout
        self._images_dir = layout.images_dir
        self._labels_dir = layout.labels_dir
        self._dataset_root = layout.dataset_root
        self._anno_map = dict(layout.annotation_map)
        self._anno_format = dict(layout.annotation_format)
        self._yolo_label_targets = dict(layout.yolo_label_targets)
        # 旧路径已失效，置空防止 on_file_selected 把 scene 误存回旧文件名
        self.current_image_path = None
        self.listFiles.blockSignals(True)
        self.listFiles.clear()
        for image_path in layout.image_paths:
            self.listFiles.addItem(image_path)
        self.listFiles.blockSignals(False)
        if self.listFiles.count() > 0:
            self.listFiles.setCurrentRow(0)
        self._align_class_list_with_yolo_ids()
        self._compute_label_counts()
        self._refresh_filter_class_combo()
        self._update_filter_status()

    def clean_unlabeled(self):
        if not self.current_dir:
            DialogOver(self, "请先打开图片目录", "提示", "warning")
            return
        # 先查找无标签图片
        unlabeled_images = DatasetCleaner.find_unlabeled_images(self.current_dir)
        empty_label_images = DatasetCleaner.find_empty_label_files(self.current_dir)
        dialog = CleanerDialog(unlabeled_images, empty_label_images, self)
        if dialog.exec():
            # 执行删除
            selected = dialog.get_selected_files()
            to_trash = dialog.is_to_trash()
            if selected:
                DatasetCleaner.delete_images_with_annotations(selected, to_trash=to_trash)
            # 刷新文件列表与映射（旧 _anno_map 键已随删除失效）
            self._reload_after_fs_change()

    # ---------------------------------------------------------
    # 层级标注 (Hierarchical Annotation)
    # ---------------------------------------------------------
    def open_hierarchical_label(self):
        """工具菜单 → 层级标注：切换到 pose 模式 + 显示层级面板。"""
        if not self.hier_config:
            self._ensure_hier_config()
        if not self.hier_panel:
            self._ensure_hier_panel()
        # 切到 pose 模式
        self._set_mode(CanvasMode.POSE)
        # 切换面板可见性
        if self.pose_panel:
            self.pose_panel.hide()
        self.hier_panel.show()
        self.hier_panel.raise_()
        # 预选兵种，保证 hier_selected_label 有值
        if not self.hier_panel.get_current_category():
            names = [c["name"] for c in self.hier_config.get_annotatable_classes()]
            default = "BO" if "BO" in names else (names[0] if names else None)
            if default:
                self.hier_panel._select_category(default)
        # 激活 canvas 层级元数据自动放置 (仅 L2)
        l2 = self.hier_panel.get_l2_code() if self.hier_panel else 0
        self.scene.set_hierarchical_active(True, l2_code=l2)
        self.hier_mode_active = True
        self._set_status("层级标注已开启 — 选择兵种 + L2(仅 BO/RO)，拖拽 bbox 后点 4 角点", "msg_warning")

    def _ensure_hier_config(self):
        from core.hierarchical_config import HierarchicalConfig
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config", "hierarchical_classes.json")
        self.hier_config = HierarchicalConfig(config_path)
        self.hier_config.load()

    def _ensure_hier_panel(self):
        from ui.hierarchical_panel import HierarchicalPanel
        self.hier_panel = HierarchicalPanel(self.hier_config, self)
        self.hier_panel.category_changed.connect(self._on_hier_category_changed)
        self.hier_panel.l2_code_changed.connect(self._on_hier_l2_changed)
        self.hier_panel.instance_selected.connect(self.on_pose_instance_selected)
        self.hier_panel.instance_delete_requested.connect(self.on_pose_instance_delete)
        self.hier_panel.config_edit_requested.connect(self._on_hier_config_edit)
        self.hier_panel.status_message.connect(self._on_pose_panel_status)
        self.dockLayout.addWidget(self.hier_panel, 1)
        self.hier_panel.hide()

    def _on_hier_category_changed(self, name: str):
        self.hier_selected_label = name
        if name and self.hier_config:
            kpt_count, _ = self.hier_config.get_kpt_spec_for_class(name)
            has_l2 = self.hier_config.has_sublevels(name)
            l2_hint = " (有 L2 子类)" if has_l2 else " (仅4角点)"
            self._set_status(f"层级标注 [{name}] - {kpt_count} 关键点{l2_hint}", "msg_warning")

    def _on_hier_l2_changed(self, code: int):
        self.scene.hier_meta_l2_code = code
        if not self.hier_mode_active or not self.hier_config:
            return
        # 即时写回当前图的 BO/RO 实例（优先选中）
        targets = self._resolve_hier_l2_targets()
        if targets is None:
            return
        if not targets:
            DialogOver(self, "请先在实例列表选中一个 BO/RO 实例，或确保图上仅有一个哨兵实例",
                       "未确定目标", "info")
            return
        kpt_names = list(self.hier_config.kpt_names)
        kpt_count = self.hier_config.kpt_count
        updated = 0
        for shape in targets:
            if self.scene.ensure_hier_l2_keypoint(shape, code, kpt_names=kpt_names, kpt_count=kpt_count):
                updated += 1
        if updated:
            self.auto_save_annotation()
            self.push_state()
            self.update_all_instances()
            self._set_status(f"已将 L2={code} 写回 {updated} 个实例并保存", "msg_success")

    def _resolve_hier_l2_targets(self):
        """解析要点子类时应更新的 PoseShape 列表。

        - 有选中的 BO/RO → 只更新选中的
        - 无选中且恰好 1 个 BO/RO → 更新该实例
        - 多个且无选中 → 返回空列表（调用方提示）
        - 没有任何 BO/RO → 返回 None（静默，仅影响后续新标）
        """
        l2_labels = {"BO", "RO"}
        if self.hier_config:
            l2_labels = {c["name"] for c in self.hier_config.classes
                         if c.get("l2_levels")} or l2_labels

        def _is_boro(s):
            return (isinstance(s, PoseShape)
                    and not getattr(s, "is_temp", False)
                    and getattr(s, "label", "") in l2_labels)

        selected = [s for s in self.scene.selectedItems() if _is_boro(s)]
        if selected:
            return selected
        all_boro = [s for s in self.scene.items() if _is_boro(s)]
        if len(all_boro) == 1:
            return all_boro
        if len(all_boro) == 0:
            return None
        return []

    def _on_hier_config_edit(self):
        from ui.hierarchical_config_dialog import HierarchicalConfigDialog
        dlg = HierarchicalConfigDialog(self.hier_config, self,
                                       preselect_name=self.hier_selected_label)
        if dlg.exec():
            self.hier_config.save()
            if self.hier_panel:
                self.hier_panel.refresh()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        nav_keys = (Qt.Key_A, Qt.Key_D, Qt.Key_Left, Qt.Key_Right)

        # SAM / LoRA 加载中禁止切换图片
        if (self.sam_loading or self.lora_sam_loading) and key in nav_keys:
            self._set_status_debounced(
                "SAM/LoRA 正在加载图片特征中，请等待完成后再切换...", "msg_error"
            )
            return

        # 长按连切：允许 auto-repeat，但节流约 180ms，避免狂风暴雨切图
        if key in nav_keys and event.isAutoRepeat():
            import time
            if (time.monotonic() - self._nav_switch_ts) < 0.18:
                return

        if key == Qt.Key_A or key == Qt.Key_Left:
            self._navigate_image(-1)
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self._navigate_image(1)
        elif key == Qt.Key_Q:
            self.samSwitch.setChecked(not self.samSwitch.isChecked())
        elif key == Qt.Key_R:
            self._set_mode(CanvasMode.RECT)
        elif key == Qt.Key_P:
            self._set_mode(CanvasMode.POLY)
        elif key == Qt.Key_T:
            self._set_mode(CanvasMode.POINT)
        elif key == Qt.Key_O:
            self._set_mode(CanvasMode.RBOX)
        elif key == Qt.Key_U:
            self._set_mode(CanvasMode.POSE)
        elif key == Qt.Key_M:
            self._set_mode(CanvasMode.MLP)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if self.scene.mode == CanvasMode.MLP:
                self._confirm_mlp_label()
        elif key == Qt.Key_Space and self.pose_mode_active:
            self._set_status("位姿实例已就绪，可直接拖拽 bbox", "msg_success")
        elif key == Qt.Key_W:
            # W: 位姿模式且有选中PoseShape时切换实例类别，否则切换类别列表
            if self.hier_mode_active and self.hier_panel:
                self.hier_panel.select_prev()
            elif self.pose_mode_active and self.pose_panel:
                selected_poses = [item for item in self.scene.selectedItems()
                                  if isinstance(item, PoseShape) and not getattr(item, 'is_temp', False)]
                if selected_poses:
                    self._switch_pose_category(selected_poses[0], -1)
                else:
                    self.pose_panel.select_prev()
            else:
                # 所有模式下 W 切换类别列表
                row = self.listClasses.currentRow()
                if row > 0:
                    self.listClasses.setCurrentRow(row - 1)
        elif key == Qt.Key_S and not (modifiers & Qt.ControlModifier):
            # S: 位姿模式且有选中PoseShape时切换实例类别，否则切换类别列表（非Ctrl+S）
            if self.hier_mode_active and self.hier_panel:
                self.hier_panel.select_next()
            elif self.pose_mode_active and self.pose_panel:
                selected_poses = [item for item in self.scene.selectedItems()
                                  if isinstance(item, PoseShape) and not getattr(item, 'is_temp', False)]
                if selected_poses:
                    self._switch_pose_category(selected_poses[0], 1)
                else:
                    self.pose_panel.select_next()
            else:
                # 所有模式下 S 切换类别列表
                row = self.listClasses.currentRow()
                if row < self.listClasses.count() - 1:
                    self.listClasses.setCurrentRow(row + 1)
        elif key == Qt.Key_F1:
            self.show_help_dialog()
        elif key == Qt.Key_Z and modifiers == Qt.ControlModifier:
            self.undo()
        elif key == Qt.Key_Y and modifiers == Qt.ControlModifier:
            self.redo()
        elif key == Qt.Key_S and modifiers == Qt.ControlModifier:
            self.auto_save_annotation()
            DialogOver(self, "标注已保存", "保存成功", "success")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        debug_log("关闭 正在清理资源...")
        # 等待后台 BUFF 标注线程结束，避免 QThread 运行中被析构导致 abort
        for worker_attr in ("_buff_sam3_worker", "_buff_semi_worker"):
            worker = getattr(self, worker_attr, None)
            if worker is not None and worker.isRunning():
                worker.wait(10000)
        if hasattr(self, "sam_client") and self.sam_client is not None:
            self.sam_client.cleanup()
        if hasattr(self, "lora_sam_client") and self.lora_sam_client is not None:
            self.lora_sam_client.cleanup()
        event.accept()

    def _set_status_debounced(self, text: str, severity: str = "msg", interval_ms: int = 800):
        """避免长按 A/D 时反复 polish 状态栏导致卡顿。"""
        import time
        now = time.monotonic()
        if (now - self._nav_status_ts) * 1000 < interval_ms and self.helpLabel.text() == text:
            return
        self._nav_status_ts = now
        self._set_status(text, severity)

    def _start_sam_image_load(self, path: str):
        self.sam_loading = True
        self._sam_load_token += 1
        token = self._sam_load_token
        self._sync_image_nav_slider()
        self.sam_client.set_image_async(path, token=token)
        self._arm_sam_load_watchdog()

    def _start_lora_image_load(self, path: str):
        self.lora_sam_loading = True
        self._lora_load_token += 1
        token = self._lora_load_token
        self._sync_image_nav_slider()
        self.lora_sam_client.set_image_async(path, token=token)
        self._arm_lora_load_watchdog()

    def _arm_sam_load_watchdog(self):
        self._stop_sam_load_watchdog()
        self._sam_load_watchdog = QTimer(self)
        self._sam_load_watchdog.setSingleShot(True)
        self._sam_load_watchdog.timeout.connect(self._on_sam_load_timeout)
        self._sam_load_watchdog.start(30000)

    def _arm_lora_load_watchdog(self):
        self._stop_lora_load_watchdog()
        self._lora_load_watchdog = QTimer(self)
        self._lora_load_watchdog.setSingleShot(True)
        self._lora_load_watchdog.timeout.connect(self._on_lora_load_timeout)
        self._lora_load_watchdog.start(30000)

    def _stop_sam_load_watchdog(self):
        if self._sam_load_watchdog is not None:
            self._sam_load_watchdog.stop()
            self._sam_load_watchdog.deleteLater()
            self._sam_load_watchdog = None

    def _stop_lora_load_watchdog(self):
        if self._lora_load_watchdog is not None:
            self._lora_load_watchdog.stop()
            self._lora_load_watchdog.deleteLater()
            self._lora_load_watchdog = None

    def _on_sam_load_timeout(self):
        if not self.sam_loading:
            return
        self.sam_loading = False
        self._sync_image_nav_slider()
        self._set_status("SAM 特征提取超时，已解锁切图", "msg_error")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _on_lora_load_timeout(self):
        if not self.lora_sam_loading:
            return
        self.lora_sam_loading = False
        self._sync_image_nav_slider()
        self._set_status("LoRA-SAM 特征提取超时，已解锁切图", "msg_error")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _compute_label_counts(self):
        """扫描标注映射，缓存每张图片的标签数量、类别集合与检测几何。"""
        self._image_label_counts.clear()
        self._image_label_sets.clear()
        self._image_geom.clear()
        empty_geom = DatasetStatistics._empty_geom()

        def _fill(image_path, annotation_path):
            if annotation_path:
                label_names = DatasetStatistics.get_annotation_labels(
                    annotation_path, self.class_list
                )
                self._image_label_counts[image_path] = len(label_names)
                self._image_label_sets[image_path] = set(label_names)
                self._image_geom[image_path] = DatasetStatistics.get_detection_geom_stats(
                    annotation_path
                )
            else:
                self._image_label_counts[image_path] = 0
                self._image_label_sets[image_path] = set()
                self._image_geom[image_path] = dict(empty_geom)

        if self._anno_map:
            for image_path, annotation_path in self._anno_map.items():
                _fill(image_path, annotation_path)
            return
        if not self.current_dir:
            return
        try:
            image_files, annotation_map = DatasetStatistics.scan_directory(self.current_dir)
            for image_path in image_files:
                _fill(image_path, annotation_map.get(image_path))
        except Exception:
            pass

    def _geom_from_shapes_data(self, shapes_data):
        """从当前 scene 导出的 shapes 汇总检测框几何（仅 rectangle）。"""
        wh = []
        for shape in shapes_data or []:
            st = (shape.get("type") or "").lower()
            if st not in ("rectangle", "rect"):
                continue
            pts = shape.get("points") or []
            if len(pts) < 2:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            wh.append((max(xs) - min(xs), max(ys) - min(ys)))
        return DatasetStatistics._geom_from_wh(wh)

    def _rebuild_nav_rows(self):
        total = self.listFiles.count()
        rows = [
            row_index for row_index in range(total)
            if self.listFiles.item(row_index)
            and self._image_matches_filter(self.listFiles.item(row_index).text())
        ] if self._is_filter_active() else list(range(total))

        sort_key = self._filter_sort
        if sort_key and sort_key != "default":
            def _sort_key(row_index):
                path = self.listFiles.item(row_index).text()
                geom = self._image_geom.get(path) or DatasetStatistics._empty_geom()
                has = 0 if geom.get("has_rect") else 1  # 无几何排末尾
                if sort_key == "area_desc":
                    return (has, -float(geom.get("max_area") or 0.0))
                if sort_key == "area_asc":
                    return (has, float(geom.get("min_area") or 0.0) if geom.get("has_rect") else 1e18)
                if sort_key == "aspect_desc":
                    return (has, -float(geom.get("max_aspect") or 0.0))
                if sort_key == "aspect_asc":
                    return (has, float(geom.get("min_aspect") or 0.0) if geom.get("has_rect") else 1e18)
                return (has, row_index)
            rows.sort(key=_sort_key)
        self._nav_rows = rows

    def _sync_image_nav_slider(self):
        if not hasattr(self, "imageNavSlider"):
            return
        self._rebuild_nav_rows()
        nav_count = len(self._nav_rows)
        self._slider_syncing = True
        self.imageNavSlider.blockSignals(True)
        self.imageNavSlider.setEnabled(nav_count > 0 and not self.sam_loading and not self.lora_sam_loading)
        if nav_count <= 1:
            self.imageNavSlider.setMaximum(0)
            self.imageNavSlider.setValue(0)
        else:
            self.imageNavSlider.setMaximum(nav_count - 1)
            current_row = self.listFiles.currentRow()
            try:
                slider_index = self._nav_rows.index(current_row)
            except ValueError:
                slider_index = 0
            self.imageNavSlider.setValue(slider_index)
        self.imageNavSlider.blockSignals(False)
        self._slider_syncing = False

    def _on_image_nav_slider_changed(self, slider_value: int):
        if self._slider_syncing or self.sam_loading or self.lora_sam_loading:
            return
        if slider_value < 0 or slider_value >= len(self._nav_rows):
            return
        target_row = self._nav_rows[slider_value]
        if target_row != self.listFiles.currentRow():
            self.listFiles.setCurrentRow(target_row)

    def _is_filter_active(self):
        return (
            self._filter_mode != "all"
            or bool(self._filter_class)
            or (self._filter_sort and self._filter_sort != "default")
            or bool(self._filter_aspect_anomaly)
        )

    def _image_matches_filter(self, path):
        count = self._image_label_counts.get(path, 0)
        if self._filter_mode == "none":
            if count != 0:
                return False
        elif self._filter_mode == "any":
            if count <= 0:
                return False
        elif self._filter_mode == "ge":
            if count < self._filter_value:
                return False
        elif self._filter_mode == "le":
            if count > self._filter_value:
                return False
        elif self._filter_mode == "eq":
            if count != self._filter_value:
                return False
        # mode == "all"：不过滤数量
        if self._filter_class:
            labels = self._image_label_sets.get(path, set())
            if self._filter_class not in labels:
                return False
        if self._filter_aspect_anomaly:
            geom = self._image_geom.get(path) or {}
            if not geom.get("has_rect"):
                return False
            if float(geom.get("max_aspect") or 0.0) < float(self._filter_aspect_thresh):
                return False
        return True

    def _navigate_image(self, direction):
        import time
        if self.sam_loading or self.lora_sam_loading:
            self._set_status_debounced(
                "SAM/LoRA 正在加载图片特征中，请等待完成后再切换...", "msg_error"
            )
            return
        if not self._is_filter_active():
            current_row = self.listFiles.currentRow()
            new_row = current_row + direction
            if 0 <= new_row < self.listFiles.count():
                self._nav_switch_ts = time.monotonic()
                self.listFiles.setCurrentRow(new_row)
            else:
                msg = "已是最后一张图片" if direction > 0 else "已是第一张图片"
                self._set_status_debounced(msg, "msg_warning")
            return

        self._rebuild_nav_rows()
        if not self._nav_rows:
            self._set_status_debounced("没有图片匹配当前筛选条件", "msg_warning")
            return
        current_row = self.listFiles.currentRow()
        try:
            idx = self._nav_rows.index(current_row)
        except ValueError:
            # 当前不在匹配集：跳到首个或末个
            target = self._nav_rows[0] if direction >= 0 else self._nav_rows[-1]
            self._nav_switch_ts = time.monotonic()
            self.listFiles.setCurrentRow(target)
            return
        new_idx = idx + (1 if direction >= 0 else -1)
        if 0 <= new_idx < len(self._nav_rows):
            self._nav_switch_ts = time.monotonic()
            self.listFiles.setCurrentRow(self._nav_rows[new_idx])
        else:
            msg = "已是最后一张匹配筛选的图片" if direction > 0 else "已是第一张匹配筛选的图片"
            self._set_status_debounced(msg, "msg_warning")

    def _find_filtered_match(self, direction):
        """沿 _nav_rows 找下一张（与滑条顺序一致）。"""
        self._rebuild_nav_rows()
        if not self._nav_rows:
            return -1
        current = self.listFiles.currentRow()
        try:
            idx = self._nav_rows.index(current)
        except ValueError:
            return self._nav_rows[0] if direction >= 0 else self._nav_rows[-1]
        new_idx = idx + (1 if direction >= 0 else -1)
        if 0 <= new_idx < len(self._nav_rows):
            return self._nav_rows[new_idx]
        return -1

    def _find_first_filtered_match(self):
        """若当前已匹配返回 -2；否则返回 _nav_rows 首项。"""
        self._rebuild_nav_rows()
        if not self._nav_rows:
            return -1
        current = self.listFiles.currentRow()
        if current in self._nav_rows:
            return -2
        return self._nav_rows[0]

    def _on_filter_changed(self, idx):
        mode_map = {0: "all", 1: "none", 2: "any", 3: "ge", 4: "le", 5: "eq"}
        self._filter_mode = mode_map.get(idx, "all")
        need_spin = self._filter_mode in ("ge", "le", "eq")
        self.filterCountSpin.setEnabled(need_spin)
        if need_spin:
            self._filter_value = self.filterCountSpin.value()
        self._apply_filter()

    def _on_filter_value_changed(self, val):
        self._filter_value = val
        self._apply_filter()

    def _on_filter_class_changed(self, idx):
        data = self.filterClassCombo.itemData(idx)
        self._filter_class = data if data else ""
        self._apply_filter()

    def _on_filter_sort_changed(self, idx):
        data = self.filterSortCombo.itemData(idx)
        self._filter_sort = data if data else "default"
        self._apply_filter()

    def _on_aspect_anomaly_toggled(self, checked):
        self._filter_aspect_anomaly = bool(checked)
        self.spinAspectThresh.setEnabled(bool(checked))
        self._apply_filter()

    def _on_aspect_thresh_changed(self, val):
        self._filter_aspect_thresh = float(val)
        if self._filter_aspect_anomaly:
            self._apply_filter()

    def _refresh_filter_class_combo(self):
        """按 class_list + 缓存 orphan 刷新类别筛选下拉。"""
        if not hasattr(self, "filterClassCombo"):
            return
        prev = self._filter_class
        self.filterClassCombo.blockSignals(True)
        self.filterClassCombo.clear()
        self.filterClassCombo.addItem("全部类别", "")
        seen = set()
        for name in self.class_list:
            if name and name not in seen:
                self.filterClassCombo.addItem(name, name)
                seen.add(name)
        orphans = set()
        for labs in self._image_label_sets.values():
            orphans.update(labs)
        for name in sorted(orphans):
            if name and name not in seen:
                self.filterClassCombo.addItem(name, name)
                seen.add(name)
        # 恢复选中
        restore = 0
        if prev:
            for i in range(self.filterClassCombo.count()):
                if self.filterClassCombo.itemData(i) == prev:
                    restore = i
                    break
            else:
                self._filter_class = ""
        self.filterClassCombo.setCurrentIndex(restore)
        self.filterClassCombo.blockSignals(False)

    def _apply_filter(self):
        """更新筛选状态；仅在需要时跳到第一个匹配图。"""
        if (self.sam_loading or self.lora_sam_loading) and self._is_filter_active():
            self._update_filter_status()
            self._update_current_label_count()
            self._sync_image_nav_slider()
            self._set_status("SAM/LoRA 加载中，筛选条件已更新，请完成后再切图", "msg_warning")
            return
        self._update_filter_status()
        if self._is_filter_active():
            target = self._find_first_filtered_match()
            if target == -2:
                pass  # 当前已匹配，不跳
            elif target >= 0:
                self.listFiles.setCurrentRow(target)
            else:
                DialogOver(self, "没有图片匹配当前筛选条件", "无匹配", "info")
        self._update_current_label_count()
        self._sync_image_nav_slider()

    def _update_filter_status(self):
        if not self._is_filter_active():
            self.filterStatusLabel.setText("")
            return
        self._rebuild_nav_rows()
        total = self.listFiles.count()
        matched = len(self._nav_rows)
        bits = []
        if self._filter_class:
            bits.append(f"类别:{self._filter_class}")
        if self._filter_sort and self._filter_sort != "default":
            bits.append(f"排序:{self._filter_sort}")
        if self._filter_aspect_anomaly:
            bits.append(f"异比≥{self._filter_aspect_thresh:g}")
        hint = (" | " + " · ".join(bits)) if bits else ""
        self.filterStatusLabel.setText(f"筛选: {matched} / {total} 张匹配{hint}")

    def _update_current_label_count(self):
        """更新当前进度显示中的标签数量"""
        if self.current_image_path and self.scene.img_item:
            total = self.listFiles.count()
            current_idx = self.listFiles.currentRow() + 1
            filename = os.path.basename(self.current_image_path)
            count = self._image_label_counts.get(self.current_image_path, 0)
            if self._is_filter_active():
                self._rebuild_nav_rows()
                matched = len(self._nav_rows)
                try:
                    filt_idx = self._nav_rows.index(self.listFiles.currentRow()) + 1
                except ValueError:
                    filt_idx = "-"
                cls_hint = f" · {self._filter_class}" if self._filter_class else ""
                self.labelProgress.setText(
                    f"<b>筛选 {filt_idx}/{matched}</b> | 全库 {current_idx}/{total}"
                    f" 标签数: <span style='color:#8FA68F;'>{count}</span>{cls_hint}"
                    f"<br><span style='font-size:11px; color:#5A5A62;'>{filename}</span>")
            else:
                self.labelProgress.setText(
                    f"<b>{current_idx}/{total}</b> 标签数: <span style='color:#8FA68F;'>{count}</span>"
                    f"<br><span style='font-size:11px; color:#5A5A62;'>{filename}</span>")


def main():
    app = QApplication(sys.argv)
    # 调试模式初始化: 检测 --debug 命令行参数
    from core.debug import init_debug_mode, debug_log
    init_debug_mode()
    debug_log("ArmorLabel startup")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
