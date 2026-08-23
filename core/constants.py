"""
core/constants.py — 全项目默认值集中地.

所有"散落"的 magic number 在这里命名, 任何读 QSettings 的地方如果拿不到值,
都从这里的常量取. 改值只改这一处, 不再去翻 5 个文件.
"""
from __future__ import annotations


# ============== 数据集划分比例 ==============
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1

# ============== RM 装甲板半自动 ==============
RM_VERTICAL_ANGLE_THRESHOLD = 50.0   # 灯条垂直角度阈值 (°)
RM_PARALLEL_THRESHOLD = 45.0         # 灯条平行度阈值 (°)
RM_BAR_LENGTH_MIN_RATIO = 0.5        # 灯条最小长度系数
RM_SEARCH_RADIUS_RATIO = 1.5         # 灯条搜索半径系数
RM_ROI_PADDING = 10                  # ROI 边缘 padding (px)
RM_OVERLAY_TIMEOUT = 1000            # 全自动 overlay 显示时长 (ms)

# ============== MLP / 数字分类 ROI ==============
MLP_ROI_LIGHT_LEN = 12
MLP_ROI_PERSP_H = 28
MLP_ROI_PERSP_W = 32
MLP_ROI_CROP_W = 16
MLP_ROI_EXPAND_RATIO = 1.3
MLP_ROI_CENTER_RATIO = 0.5

# ============== YOLO 推理 ==============
YOLO_CONF = 0.25
YOLO_IOU = 0.7
YOLO_IMGSZ = 640
YOLO_DEVICE = "cuda"

# ============== SAM3 ==============
SAM_DEVICE = "cuda"
SAM_MODEL_FILENAME = "sam3.pt"

# ============== 编辑器 ==============
MAX_HISTORY_STEPS = 20
SNAP_THRESHOLD = 8   # 多边形顶点吸附距离 (px)

# ============== YOLO-Pose kpt 视觉阈值 ==============
KPT_VIS_THRESHOLD = 0.5   # >= 此值为 visible (2)
KPT_OCC_THRESHOLD = 0.1   # < 此值为 not visible (0), 中间为 occluded (1)

# ============== 关键点可见性颜色 (RGB 元组, 0=未标注/1=遮挡/2=可见) ==============
# main.py 中原本 4 处硬编码 {0: (100,100,100), 1: (255,165,0), 2: (0,255,0)}
# 统一到这里一处定义. 修改关键点配色只改这里.
KPT_COLOR_UNLABELED = (100, 100, 100)   # 灰
KPT_COLOR_OCCLUDED  = (255, 165, 0)     # 橙
KPT_COLOR_VISIBLE   = (0, 255, 0)       # 绿
KPT_COLORS = {0: KPT_COLOR_UNLABELED, 1: KPT_COLOR_OCCLUDED, 2: KPT_COLOR_VISIBLE}

# ============== 形状类型枚举 (用作 stringly-typed 的常量) ==============
SHAPE_RECTANGLE = "rectangle"
SHAPE_POLYGON = "polygon"
SHAPE_POINT = "point"
SHAPE_OBB = "obb"
SHAPE_POSE = "pose"
SHAPE_HIERARCHY = "hierarchy_pose"

SHAPE_TYPES = (SHAPE_RECTANGLE, SHAPE_POLYGON, SHAPE_POINT, SHAPE_OBB, SHAPE_POSE, SHAPE_HIERARCHY)

# ============== 层级标注 KP 索引 (Virtual Keypoint Multiplexing) ==============
# 14 类方案: 颜色并入 class_id, 关键点 5 个 (4 角 + L2 虚拟点, center 不单独存)
KPT_HIER_INDEX_CORNER_TL = 0
KPT_HIER_INDEX_CORNER_TR = 1
KPT_HIER_INDEX_CORNER_BL = 2
KPT_HIER_INDEX_CORNER_BR = 3
KPT_HIER_INDEX_L2_SUBLEVEL = 4
KPT_HIER_TOTAL = 5

# 14 类名 (索引即 class_id, 与 cpp 推理框架对齐)
# BG=蓝哨兵 / RG=红哨兵；BO/RO=前哨(含L2)；BB/RB=基地；B1-4/R1-4=编号装甲
HIER_CLASS_NAMES = [
    "BG", "B1", "B2", "B3", "B4", "BO", "BB",
    "RG", "R1", "R2", "R3", "R4", "RO", "RB",
]
HIER_NUM_CLASSES = 14
HIER_CLASS_ID_BG = 0  # 蓝方哨兵 (Blue Guard)，可标注 4 角点
HIER_BG_CLASS_ID = HIER_CLASS_ID_BG  # 旧名兼容

# BUFF 能量机关类别名及别名映射
BUFF_CLASS_NAMES = ["RT", "BT"]
BUFF_ALIAS_MAP = {
    "red_target": "RT",
    "blue_target": "BT",
    "redtarget": "RT",
    "bluetarget": "BT",
}

# ============== QSettings 命名空间 ==============
# 三个 namespace 统一在这里, 新代码请用 core.settings.Settings facade
NS_APP = "ArmorLabel"
NS_LEGACY = "LuoHuaLabel"

# ============== 持久化路径 ==============
DEFAULT_DATA_DIR = "origin_data"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_MLP_MODEL_DIR = "stored_weights/model/armor_tradition"
CONFIG_ROOT = "yolo_train_configs"

# ============== 数据集打开布局 ==============
DATASET_LAYOUT_FLAT = "flat"
DATASET_LAYOUT_YOLO = "yolo"
ANNO_FORMAT_JSON = "json"
ANNO_FORMAT_YOLO = "yolo"
IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
