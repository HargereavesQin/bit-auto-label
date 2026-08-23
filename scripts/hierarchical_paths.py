"""层级标注训练 — 路径与超参 (DecoupledEdgeYOLO + yolov10n)."""
from pathlib import Path

REPO = Path("/home/dreamchaser/Desktop/autolabel/armor-label")

# 数据集 — 层级标注输出目录 (YOLO pose 格式, 7 kpt)
DATASET_ROOT = REPO / "datasets/hierarchical_armor"
TRAIN_IMAGES = DATASET_ROOT / "images/train"
VAL_IMAGES = DATASET_ROOT / "images/val"
TRAIN_LABELS = DATASET_ROOT / "labels/train"
VAL_LABELS = DATASET_ROOT / "labels/val"
DATA_YAML = DATASET_ROOT / "data.yaml"

# 类别 (14 类, 与 cpp 推理框架对齐; 0:BG=蓝哨兵, 7:RG=红哨兵, 全部可标注)
CLASS_NAMES = ["BG", "B1", "B2", "B3", "B4", "BO", "BB",
               "RG", "R1", "R2", "R3", "R4", "RO", "RB"]
L2_NAMES = ["none", "up", "mid", "down"]
KPT_SHAPE = [5, 3]  # KP0-3 四角点, KP4 L2 子类 (虚拟点, 坐标复用 bbox 中心)

# 模型
BACKBONE_YAML = "yolov10n.yaml"  # vendored ultralytics, 无需 .pt
NUM_CLASSES = len(CLASS_NAMES)   # 14
NUM_L2 = len(L2_NAMES)           # 4

# 训练超参 (RTX 4090 24G · yolov10n · imgsz 640 · amp)
BATCH_SIZE = 32
TRAIN_IMGSZ = 640
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 5e-4
WORKERS = 8
DEVICE = "0"
AMP = True

# 损失权重
POSE_BOX_W = 7.5
POSE_KPT_W = 12.0
L2_W = 1.0

# 输出
RUNS_ROOT = REPO / "runs"
RUN_NAME = "hierarchical_v10n"
OUTPUT_DIR = RUNS_ROOT / RUN_NAME
WEIGHTS_DIR = OUTPUT_DIR / "weights"
LOG_FILE = OUTPUT_DIR / "train.log"

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
