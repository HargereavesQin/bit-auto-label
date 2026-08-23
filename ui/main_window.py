"""
ui/main_window.py — ArmorLabel 主窗口 UI 装配.

所有视觉样式都从 ui/style.qss 加载 (在 main.py 启动时统一载入),
本文件只负责布局 / 控件创建 / objectName 与 role 属性设置,
不再写 inline setStyleSheet.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QDockWidget, QListWidget, QGraphicsView,
    QLabel, QLineEdit, QPushButton, QStatusBar, QMenu,
    QListWidgetItem, QMenuBar,     QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSizePolicy, QFrame, QSlider,
)
from PySide6.QtCore import Qt, Signal, QRect, QSize, QPoint
from PySide6.QtGui import (
    QAction, QActionGroup, QPainter, QColor, QFont, QBrush, QPen,
    QLinearGradient, QRadialGradient,
)

from .icons import Icon, Icons, C, set_action_icon


# =========================================================
# Tactical switch — 战术风拨片
# =========================================================
class SwitchControl(QWidget):
    toggled = Signal(bool)

    TRACK_W, TRACK_H = 44, 22
    KNOB = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.TRACK_W, self.TRACK_H)
        self.setCursor(Qt.PointingHandCursor)
        self._checked = False

    def isChecked(self):
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self.toggled.emit(checked)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setChecked(not self._checked)
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 轨道
        track_rect = QRect(0, 0, self.TRACK_W, self.TRACK_H)
        if self._checked:
            # 激活: 香槟金轨道 + 极轻 glow
            p.setBrush(QColor(C.AMBER))
            glow = QColor(C.AMBER)
            glow.setAlpha(40)
            p.setPen(QPen(glow, 2))
            p.drawRoundedRect(track_rect.adjusted(-1, -1, 1, 1), 11, 11)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C.AMBER))
        else:
            # 关闭: 灰轨道
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C.BG_ELEVATED))
        p.drawRoundedRect(track_rect, 11, 11)

        # 拨片 (knob)
        p.setBrush(QColor(C.TEXT if not self._checked else C.BG_APP))
        p.setPen(Qt.NoPen)
        if self._checked:
            knob_x = self.TRACK_W - self.KNOB - 2
        else:
            knob_x = 2
        p.drawEllipse(knob_x, 2, self.KNOB, self.KNOB)


# =========================================================
# AccentDivider — 状态条中间嵌入的细斜线分割
# 单色香槟金 → 暖灰, 不再走"蓝方/红方"战场暗示
# =========================================================
class DiagonalFieldLine(QWidget):
    """状态条中间嵌入的斜线分割. 宽 140, 高 28.

    改成单色香槟金, 装饰感而非战场暗示.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 28)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # 渐变笔刷: 香槟金 → 透明 → 暮霞 (低调双向)
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0,  QColor(C.AMBER))
        grad.setColorAt(0.4,  QColor(C.AMBER))
        grad.setColorAt(0.5,  QColor(140, 140, 150, 0))
        grad.setColorAt(0.6,  QColor(C.MUTED))
        grad.setColorAt(1.0,  QColor(C.MUTED))

        # 1. 外层 glow: 8px 香槟金光晕
        glow_pen = QPen()
        glow_pen.setWidth(8)
        glow_pen.setCapStyle(Qt.RoundCap)
        glow_grad = QLinearGradient(0, 0, w, h)
        glow_grad.setColorAt(0.0,  QColor(201, 163, 107, 70))
        glow_grad.setColorAt(0.5,  QColor(140, 140, 150, 0))
        glow_grad.setColorAt(1.0,  QColor(140, 140, 150, 70))
        glow_pen.setBrush(glow_grad)
        p.setPen(glow_pen)
        p.drawLine(4, h - 4, w - 4, 4)

        # 2. 主轴: 2px 单色渐变
        pen = QPen()
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setBrush(grad)
        p.setPen(pen)
        p.drawLine(4, h - 4, w - 4, 4)

        # 3. 双侧端点
        p.setPen(Qt.NoPen)

        # 左下端点 (香槟金)
        for r, alpha in [(6, 40), (4, 90)]:
            c = QColor(C.AMBER); c.setAlpha(alpha)
            p.setBrush(c)
            p.drawEllipse(QPoint(4, h - 4), r, r)
        p.setBrush(QColor(C.AMBER))
        p.drawEllipse(QPoint(4, h - 4), 3, 3)

        # 右上端点 (暗灰)
        for r, alpha in [(6, 40), (4, 90)]:
            c = QColor(C.MUTED); c.setAlpha(alpha)
            p.setBrush(c)
            p.drawEllipse(QPoint(w - 4, 4), r, r)
        p.setBrush(QColor(C.MUTED))
        p.drawEllipse(QPoint(w - 4, 4), 3, 3)

        # 4. 中央小菱形 (中立标记)
        p.setBrush(QColor(C.TEXT_MUTED))
        cx, cy = w / 2, h / 2
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPolygonF
        diamond = QPolygonF([
            QPointF(cx, cy - 3), QPointF(cx + 3, cy),
            QPointF(cx, cy + 3), QPointF(cx - 3, cy),
        ])
        p.drawPolygon(diamond)


# =========================================================
# Canvas view — 包装 QGraphicsView, 加滚轮缩放/中键平移
# 背景渐变由 ui/style.qss 中的 canvas role 控制
# =========================================================
class CanvasView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setDragMode(QGraphicsView.NoDrag)

        self._is_panning = False
        self._pan_start_pos = None

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(1.0 / zoom_in_factor, 1.0 / zoom_in_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start_pos = event.position().toPoint()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# =========================================================
# 主窗口装配
# =========================================================
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("ArmorLabel  /  DATASET ARMORY")
        MainWindow.resize(1380, 860)

        self._build_menubar(MainWindow)
        self._build_central(MainWindow)
        self._build_statusbar(MainWindow)
        self._build_left_toolbar(MainWindow)
        self._build_right_dock(MainWindow)
        self._build_canvas_actions(MainWindow)
        self._build_format_chip(MainWindow)

    # ---------------------------------------------------------
    def _build_menubar(self, MainWindow):
        self.menuBar = QMenuBar(MainWindow)

        self.menuTools = QMenu("工具", self.menuBar)

        self.actionDatasetTool     = QAction("数据集处理", MainWindow)
        self.actionDatasetFusion   = QAction("数据集融合", MainWindow)
        self.actionVideoExtractor  = QAction("视频抽帧", MainWindow)
        self.actionYoloAuto        = QAction("YOLO 自动标注", MainWindow)
        self.actionYoloTrain       = QAction("YOLO 训练", MainWindow)
        self.actionYoloInference   = QAction("YOLO 推理查看", MainWindow)
        self.actionMLPTrain        = QAction("MLP 分类器训练", MainWindow)
        self.actionMobileNetTrain  = QAction("MobileNet 分类器训练", MainWindow)
        self.actionMLPDemo         = QAction("MLP 分类推理演示", MainWindow)
        self.actionArmorExtractor  = QAction("装甲板提取", MainWindow)
        self.actionStatistics      = QAction("统计", MainWindow)
        self.actionCleaner         = QAction("清理无标签图片", MainWindow)
        self.actionMergeClass      = QAction("合并类别", MainWindow)
        self.actionBatchRename     = QAction("批量重命名", MainWindow)
        self.actionHierarchicalLabel = QAction("层级标注", MainWindow)

        set_action_icon(self.actionDatasetTool,    Icons.DATASET)
        set_action_icon(self.actionDatasetFusion,  Icons.FUSION)
        set_action_icon(self.actionVideoExtractor, Icons.VIDEO)
        set_action_icon(self.actionYoloAuto,       Icons.AUTO)
        set_action_icon(self.actionYoloTrain,      Icons.TRAIN)
        set_action_icon(self.actionYoloInference,  Icons.SCAN)
        set_action_icon(self.actionMLPTrain,       Icons.NEURAL)
        set_action_icon(self.actionMobileNetTrain, Icons.NEURAL)
        set_action_icon(self.actionMLPDemo,        Icons.PLAY)
        set_action_icon(self.actionArmorExtractor, Icons.ARMOR)
        set_action_icon(self.actionStatistics,     Icons.CHART)
        set_action_icon(self.actionCleaner,        Icons.BROOM)
        set_action_icon(self.actionMergeClass,     Icons.MERGE)
        set_action_icon(self.actionBatchRename,    Icons.RENAME)
        set_action_icon(self.actionHierarchicalLabel, Icons.POSE)

        self.menuTools.addAction(self.actionDatasetTool)
        self.menuTools.addAction(self.actionDatasetFusion)
        self.menuTools.addAction(self.actionVideoExtractor)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionYoloAuto)
        self.menuTools.addAction(self.actionYoloTrain)
        self.menuTools.addAction(self.actionYoloInference)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionMLPTrain)
        self.menuTools.addAction(self.actionMobileNetTrain)
        self.menuTools.addAction(self.actionMLPDemo)
        self.menuTools.addAction(self.actionArmorExtractor)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionHierarchicalLabel)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionStatistics)
        self.menuTools.addAction(self.actionMergeClass)
        self.menuTools.addAction(self.actionBatchRename)
        self.menuTools.addAction(self.actionCleaner)

        self.menuBar.addMenu(self.menuTools)

        self.menuSettings = QMenu("设置", self.menuBar)
        self.actionSettings = QAction("参数设置", MainWindow)
        set_action_icon(self.actionSettings, Icons.SLIDERS)
        self.menuSettings.addAction(self.actionSettings)
        self.menuBar.addMenu(self.menuSettings)

        MainWindow.setMenuBar(self.menuBar)

    # ---------------------------------------------------------
    def _build_central(self, MainWindow):
        self.centralWidget = QWidget(MainWindow)
        self.mainLayout = QHBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        self.view = CanvasView()
        self.view.setProperty("role", "canvas")   # 编辑风画布底色(纯色,无战场渐变)
        self.mainLayout.addWidget(self.view, 1)
        MainWindow.setCentralWidget(self.centralWidget)

    # ---------------------------------------------------------
    def _build_statusbar(self, MainWindow):
        self.statusBar = QStatusBar()
        # 左右分栏: mode 指示 + help 文字 + 对角线分隔条 + 坐标
        MainWindow.setStatusBar(self.statusBar)

        # 左侧: 模式指示 (mono 等宽)
        self.modeLabel = QLabel("[ RECT ]")
        self.modeLabel.setProperty("role", "mode")
        self.statusBar.addWidget(self.modeLabel)

        # 中间: 状态文字 (中性灰)
        self.helpLabel = QLabel("就绪")
        self.helpLabel.setProperty("role", "mono")
        self.statusBar.addWidget(self.helpLabel, 1)

        # 永久右侧: 斜线分割 + 坐标
        # 在 addPermanentWidget 之前先塞一个 stretch 让中右之间有空间放斜线
        self.fieldLine = DiagonalFieldLine()
        self.statusBar.addPermanentWidget(self.fieldLine)

        self.coordLabel = QLabel("X: 0000  Y: 0000")
        self.coordLabel.setProperty("role", "coord")  # 红色
        self.statusBar.addPermanentWidget(self.coordLabel)

    # ---------------------------------------------------------
    def _build_left_toolbar(self, MainWindow):
        """左侧工具栏: 打开 / 模式按钮 / SAM 拨片.

        v2: 加大图标到 24, 工具按钮加 tooltip 短键, 组分隔用蓝/红色细条,
        顶部加 ARMORY 标题, 底部加版本号徽章.
        """
        self.toolBar = QToolBar("工具栏")
        self.toolBar.setOrientation(Qt.Vertical)
        self.toolBar.setMovable(False)
        self.toolBar.setIconSize(QSize(24, 24))
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolBar.setObjectName("leftToolbar")
        MainWindow.addToolBar(Qt.LeftToolBarArea, self.toolBar)

        # ---- 顶部品牌头 ----
        brand = QLabel("DATASET\nARMORY")
        brand.setAlignment(Qt.AlignCenter)
        brand.setProperty("role", "brand")
        self.toolBar.addWidget(brand)
        self.toolBar.addSeparator()

        # 打开目录
        self.actionOpen = QAction("打开目录", MainWindow)
        set_action_icon(self.actionOpen, Icons.FOLDER, size=24)
        self.actionOpen.setToolTip("打开图片目录  (Ctrl+O)")
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addSeparator()

        # 格式 chip 占位
        self.formatLabel = QLabel("JSON")
        self.formatLabel.setProperty("role", "chip")
        self.formatLabel.setProperty("accent", "amber")
        self.formatLabel.setAlignment(Qt.AlignCenter)
        self.formatLabel.setMinimumWidth(64)
        self.toolBar.addWidget(self.formatLabel)
        self.toolBar.addSeparator()

        # ---- 标注模式分组小标题 ----
        mode_header = QLabel("MODES")
        mode_header.setAlignment(Qt.AlignCenter)
        mode_header.setProperty("role", "toolbar_section")
        self.toolBar.addWidget(mode_header)

        # 5 种标注模式 (canvas actions) 在 _build_canvas_actions 注入
        self.modeGroup = QActionGroup(MainWindow)
        self.modeGroup.setExclusive(True)
        # 占位 actions 真正创建在 _build_canvas_actions
        self.actionRect  = QAction("矩形", MainWindow)
        self.actionPoly  = QAction("多边形", MainWindow)
        self.actionPoint = QAction("点", MainWindow)
        self.actionRBox  = QAction("旋转框", MainWindow)
        self.actionPose  = QAction("位姿", MainWindow)
        self.actionMLP   = QAction("MLP", MainWindow)

        # 短键 tooltip
        self.actionRect.setShortcut("R")
        self.actionPoly.setShortcut("P")
        self.actionPoint.setShortcut("T")
        self.actionRBox.setShortcut("O")
        self.actionPose.setShortcut("U")
        self.actionMLP.setShortcut("M")

    # ---------------------------------------------------------
    def _build_canvas_actions(self, MainWindow):
        """5 个标注模式按钮 (图标 24px, 文字在下方, 激活时蓝色左边条 + glow)."""
        set_action_icon(self.actionRect,  Icons.RECT,  size=24)
        set_action_icon(self.actionPoly,  Icons.POLY,  size=24)
        set_action_icon(self.actionPoint, Icons.POINT, size=24)
        set_action_icon(self.actionRBox,  Icons.OBB,   size=24)
        set_action_icon(self.actionPose,  Icons.POSE,  size=24)
        set_action_icon(self.actionMLP,   Icons.NEURAL, size=24)

        for act in (self.actionRect, self.actionPoly, self.actionPoint,
                    self.actionRBox, self.actionPose, self.actionMLP):
            act.setCheckable(True)
            self.modeGroup.addAction(act)
            self.toolBar.addAction(act)

        self.actionRect.setChecked(True)
        self.toolBar.addSeparator()

        # ---- 开关分组小标题 ----
        switch_header = QLabel("TOGGLES")
        switch_header.setAlignment(Qt.AlignCenter)
        switch_header.setProperty("role", "toolbar_section")
        self.toolBar.addWidget(switch_header)

        # 光条外接拨片
        self.rectLightFitWidget = QWidget()
        rectLightFitLayout = QVBoxLayout(self.rectLightFitWidget)
        rectLightFitLayout.setContentsMargins(4, 8, 4, 4)
        rectLightFitLayout.setSpacing(2)
        rectLightFitLayout.setAlignment(Qt.AlignCenter)

        self.rectLightFitSwitch = SwitchControl()
        self.rectLightFitSwitch.setObjectName("rectLightFitSwitch")
        rectLightFitLayout.addWidget(self.rectLightFitSwitch, alignment=Qt.AlignCenter)

        self.rectLightFitLabel = QLabel("光条外接")
        self.rectLightFitLabel.setAlignment(Qt.AlignCenter)
        self.rectLightFitLabel.setProperty("role", "caption")
        rectLightFitLayout.addWidget(self.rectLightFitLabel)
        self.toolBar.addWidget(self.rectLightFitWidget)
        self.toolBar.addSeparator()

        # SAM 拨片
        self.samWidget = QWidget()
        samLayout = QVBoxLayout(self.samWidget)
        samLayout.setContentsMargins(4, 4, 4, 8)
        samLayout.setSpacing(2)
        samLayout.setAlignment(Qt.AlignCenter)

        self.samSwitch = SwitchControl()
        self.samSwitch.setObjectName("samSwitch")
        samLayout.addWidget(self.samSwitch, alignment=Qt.AlignCenter)

        self.samLabel = QLabel("SAM 智能辅助")
        self.samLabel.setAlignment(Qt.AlignCenter)
        self.samLabel.setProperty("role", "caption")
        samLayout.addWidget(self.samLabel)
        self.toolBar.addWidget(self.samWidget)

        # 底部 version 徽章
        self.toolBar.addSeparator()
        ver = QLabel("v0.5")
        ver.setAlignment(Qt.AlignCenter)
        ver.setProperty("role", "version_badge")
        self.toolBar.addWidget(ver)

    # ---------------------------------------------------------
    def _build_format_chip(self, _MainWindow):
        """No-op: format chip 已在 _build_left_toolbar 里创建."""
        pass

    # ---------------------------------------------------------
    def _build_right_dock(self, MainWindow):
        """右侧停靠: 类别 / 实例 / SAM 提示词."""
        self.dockRight = QDockWidget("标注管理", MainWindow)
        self.dockRight.setAllowedAreas(Qt.RightDockWidgetArea)
        self.dockRight.setMaximumWidth(360)
        self.dockRight.setMinimumWidth(300)
        MainWindow.addDockWidget(Qt.RightDockWidgetArea, self.dockRight)

        self.dockRightWidget = QWidget()
        self.dockLayout = QVBoxLayout(self.dockRightWidget)
        self.dockLayout.setContentsMargins(10, 10, 10, 10)
        self.dockLayout.setSpacing(10)

        # ----- 类别段 -----
        self._build_classes_section()

        # ----- 筛选栏 -----
        self._build_filter_section()

        # ----- 当前图片实例段 -----
        self._build_instances_section()

        # ----- 底部状态 (进度 / MLP) -----
        self.labelMlpStatus = QLabel("")
        self.labelMlpStatus.setAlignment(Qt.AlignCenter)
        self.labelMlpStatus.setProperty("role", "caption")
        self.dockLayout.addWidget(self.labelMlpStatus)

        self.labelProgress = QLabel("0 / 0")
        self.labelProgress.setAlignment(Qt.AlignCenter)
        self.labelProgress.setProperty("role", "mono")
        self.labelProgress.setProperty("accent", "amber")
        # 用动态属性换背景: 见 QSS 中 [role="mono"][accent="amber"]
        self.dockLayout.addWidget(self.labelProgress)

        self.imageNavSlider = QSlider(Qt.Horizontal)
        self.imageNavSlider.setMinimum(0)
        self.imageNavSlider.setMaximum(0)
        self.imageNavSlider.setProperty("role", "mono")
        self.imageNavSlider.setToolTip("拖动快速切换图片")
        self.dockLayout.addWidget(self.imageNavSlider)

        self.dockLayout.addStretch()

        # ----- SAM 提示词输入 -----
        self._build_sam_prompt_section()

        self.dockRight.setWidget(self.dockRightWidget)

    def _section_header(self, text: str, accent: str = "muted") -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("role", "section_header")
        lbl.setProperty("accent", accent)
        return lbl

    def _build_classes_section(self):
        self.labelClasses = self._section_header("CLASSES  /  类别", accent="blue")
        self.dockLayout.addWidget(self.labelClasses)

        self.listClasses = QListWidget()
        self.listClasses.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listClasses.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.dockLayout.addWidget(self.listClasses, 3)

        # 新建类别按钮 (主操作)
        self.btnAddClass = QPushButton("  新建类别")
        self.btnAddClass.setObjectName("btnAddClass")
        self.btnAddClass.setProperty("role", "primary")
        self.btnAddClass.setCursor(Qt.PointingHandCursor)
        self.btnAddClass.setIcon(Icon.icon(Icons.PLUS, size=14, color=C.BG_APP))
        self.btnAddClass.setIconSize(QSize(14, 14))
        self.dockLayout.addWidget(self.btnAddClass)

        # RM-mode 快速导入 14 类 (与 cpp 框架对齐)
        self.btnRMMode = QPushButton("  RM-mode (14 类)")
        self.btnRMMode.setObjectName("btnRMMode")
        self.btnRMMode.setProperty("role", "ghost")
        self.btnRMMode.setCursor(Qt.PointingHandCursor)
        self.dockLayout.addWidget(self.btnRMMode)

    def _build_filter_section(self):
        self.dockLayout.addWidget(self._section_header("FILTER  /  筛选", accent="muted"))

        self.filterBar = QWidget()
        filterLayout = QHBoxLayout(self.filterBar)
        filterLayout.setContentsMargins(0, 0, 0, 0)
        filterLayout.setSpacing(6)

        self.filterCombo = QComboBox()
        self.filterCombo.addItems(["全部", "无标签 (0)", "有标签 (>0)", "标签数 ≥", "标签数 ≤", "标签数 ="])
        self.filterCountSpin = QSpinBox()
        self.filterCountSpin.setRange(0, 9999)
        self.filterCountSpin.setValue(1)
        self.filterCountSpin.setEnabled(False)
        self.filterCountSpin.setFixedWidth(60)
        self.filterCountSpin.setProperty("role", "mono")

        filterLayout.addWidget(self.filterCombo, 1)
        filterLayout.addWidget(self.filterCountSpin)
        self.dockLayout.addWidget(self.filterBar)

        # 按类别筛选（与数量条件 AND）
        self.filterClassBar = QWidget()
        classLayout = QHBoxLayout(self.filterClassBar)
        classLayout.setContentsMargins(0, 0, 0, 0)
        classLayout.setSpacing(6)
        self.filterClassCombo = QComboBox()
        self.filterClassCombo.addItem("全部类别", "")
        self.filterClassCombo.setMinimumContentsLength(8)
        classLayout.addWidget(self.filterClassCombo, 1)
        self.dockLayout.addWidget(self.filterClassBar)

        # 检测框几何：面积/长宽比排序 + 异常长宽比
        self.filterGeomBar = QWidget()
        geomLayout = QHBoxLayout(self.filterGeomBar)
        geomLayout.setContentsMargins(0, 0, 0, 0)
        geomLayout.setSpacing(6)
        self.filterSortCombo = QComboBox()
        self.filterSortCombo.addItem("默认顺序", "default")
        self.filterSortCombo.addItem("面积↓(最大优先)", "area_desc")
        self.filterSortCombo.addItem("面积↑(最小优先)", "area_asc")
        self.filterSortCombo.addItem("长宽比↓", "aspect_desc")
        self.filterSortCombo.addItem("长宽比↑", "aspect_asc")
        self.chkAspectAnomaly = QCheckBox("异常长宽比")
        self.spinAspectThresh = QDoubleSpinBox()
        self.spinAspectThresh.setRange(1.5, 50.0)
        self.spinAspectThresh.setSingleStep(0.5)
        self.spinAspectThresh.setValue(4.0)
        self.spinAspectThresh.setDecimals(1)
        self.spinAspectThresh.setEnabled(False)
        self.spinAspectThresh.setFixedWidth(64)
        self.spinAspectThresh.setProperty("role", "mono")
        geomLayout.addWidget(self.filterSortCombo, 1)
        geomLayout.addWidget(self.chkAspectAnomaly)
        geomLayout.addWidget(self.spinAspectThresh)
        self.dockLayout.addWidget(self.filterGeomBar)

        self.filterStatusLabel = QLabel("")
        self.filterStatusLabel.setProperty("role", "caption")
        self.filterStatusLabel.setAlignment(Qt.AlignCenter)
        self.dockLayout.addWidget(self.filterStatusLabel)

    def _build_instances_section(self):
        self.labelInstances = self._section_header("INSTANCES  /  实例", accent="red")
        self.dockLayout.addWidget(self.labelInstances)

        self.instanceListWidget = QListWidget()
        self.instanceListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.instanceListWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.dockLayout.addWidget(self.instanceListWidget, 2)

        # 隐藏的 listFiles 占位, 保留 main.py 兼容性
        self.listFiles = QListWidget()
        self.listFiles.setVisible(False)

    def _build_sam_prompt_section(self):
        self.dockLayout.addWidget(self._section_header("SAM  /  提示词提取", accent="amber"))

        # 提示词输入与提交按钮装在 samTextGroup 容器里 (兼容 main.py 的 hide/show)
        self.samTextGroup = QWidget()
        sam_text_layout = QVBoxLayout(self.samTextGroup)
        sam_text_layout.setContentsMargins(0, 6, 0, 0)
        sam_text_layout.setSpacing(6)

        # 提示词输入 + 提交按钮一行
        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(6)

        self.samPromptInput = QLineEdit()
        self.samPromptInput.setPlaceholderText("输入提示词, 如: number / light / car")
        self.samPromptInput.setProperty("role", "search")
        prompt_row.addWidget(self.samPromptInput, 1)

        self.samPromptBtn = QPushButton("提交")
        self.samPromptBtn.setProperty("role", "primary")
        self.samPromptBtn.setCursor(Qt.PointingHandCursor)
        self.samPromptBtn.setIcon(Icon.icon(Icons.SPARK, size=14, color=C.BG_APP))
        self.samPromptBtn.setIconSize(QSize(14, 14))
        prompt_row.addWidget(self.samPromptBtn)
        sam_text_layout.addLayout(prompt_row)
        self.dockLayout.addWidget(self.samTextGroup)

        # 帮助按钮 (右下角小)
        help_row = QHBoxLayout()
        help_row.addStretch()
        self.btnHelp = QPushButton("?")
        self.btnHelp.setToolTip("使用说明 (F1)")
        self.btnHelp.setFixedSize(22, 22)
        self.btnHelp.setCursor(Qt.PointingHandCursor)
        self.btnHelp.setProperty("role", "ghost")
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        self.btnHelp.setFont(font)
        help_row.addWidget(self.btnHelp)
        self.dockLayout.addLayout(help_row)
