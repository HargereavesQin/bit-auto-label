# -*- coding: utf-8 -*-
"""
@File ：main_dataset_tool.py
@Motto :学习新思想，争做新青年

数据集处理工具 — 编辑风 (Editorial Warm Dark).
- 自带 QSS, 不依赖 ui/style.qss (走完全独立的视觉语言)
- 暖灰底色 + 香槟金点缀, 留白克制
- 衬线标题 + Sans 正文 + Mono 数据
- 不写 emoji, 不写战术风装饰
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import random
import shutil
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
                               QTextEdit, QDialog, QSpinBox, QMessageBox, QFrame,
                               QCheckBox, QScrollArea, QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QTextCursor, QColor, QPainter, QPen, QFontDatabase

from utils.message import DialogOver


# =========================================================
# 加载编辑风共享样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


STYLE_QSS = _load_editorial_qss()




# =========================================================
# 模式定义
# =========================================================
MODES = [
    # (key,           显示名,                     描述,                                accent)
    ("YOLO_SPLIT",      "YOLO 纯划分",              "TXT → 7/2/1 打乱切分",                 "amber"),
    ("TO_YOLO_DET",     "JSON / XML  →  YOLO 检测", "rectangle + point",                   "amber"),
    ("TO_YOLO_SEG",     "JSON / XML  →  YOLO 分割", "rectangle + polygon",                 "blue"),
    ("TO_YOLO_OBB",     "JSON / XML  →  YOLO-OBB",  "rotated bbox",                       "blue"),
    ("TO_YOLO_POSE",    "JSON / XML  →  YOLO-Pose", "keypoints",                          "blue"),
    ("FROM_YOLO_JSON",  "YOLO  →  JSON",            "images+labels 自动识别 train/val/test", "green"),
    ("FILTER_JSON_KPTS","JSON 关键点筛选",           "位姿 JSON 批量保留指定序号关键点",      "green"),
    ("KPTS_TO_DET_JSON","四点 → 检测 JSON",         "位姿/YOLO 按所选4点生成外接 detect 框", "green"),
    ("UNET",            "JSON  →  U-Net 分割",      "instance masks",                     "green"),
    ("TO_MLP_CLASSIFY", "JSON  →  MLP 分类",        "mlp_label 导出",                      "amber"),
]


# =========================================================
# 关键点序号选择面板
# =========================================================
class KptFilterPanel(QFrame):
    """扫描结果展示 + 关键点序号多选。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "card")
        self._checkboxes: list[QCheckBox] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self.info_lbl = QLabel("点击「分析数据集」识别类别；「一键识别整理」同步 pose 配置并删除多余类别")
        self.info_lbl.setProperty("role", "caption")
        self.info_lbl.setWordWrap(True)
        layout.addWidget(self.info_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.setProperty("role", "icon")
        self.select_all_btn.setCursor(Qt.PointingHandCursor)
        self.select_all_btn.clicked.connect(self.select_all)
        btn_row.addWidget(self.select_all_btn)

        self.clear_all_btn = QPushButton("清空")
        self.clear_all_btn.setProperty("role", "icon")
        self.clear_all_btn.setCursor(Qt.PointingHandCursor)
        self.clear_all_btn.clicked.connect(self.clear_all)
        btn_row.addWidget(self.clear_all_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(88)
        scroll.setFrameShape(QFrame.NoFrame)
        self._kpt_host = QWidget()
        self._kpt_grid = QGridLayout(self._kpt_host)
        self._kpt_grid.setContentsMargins(0, 0, 0, 0)
        self._kpt_grid.setSpacing(6)
        scroll.setWidget(self._kpt_host)
        layout.addWidget(scroll)

    def _clear_checkboxes(self):
        for cb in self._checkboxes:
            cb.deleteLater()
        self._checkboxes.clear()
        while self._kpt_grid.count():
            item = self._kpt_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def apply_scan_result(self, scan: dict, default_first_n: int | None = None):
        task = scan.get("task", "detect")
        classes = scan.get("used_classes", [])
        kpt_counts = scan.get("kpt_counts", {})
        max_kpt = scan.get("max_kpt", 0)

        if task == "mixed":
            task_txt = "混合 (detect + pose)"
        elif task == "pose":
            task_txt = "位姿 Pose"
        else:
            task_txt = "检测 Detect"

        kpt_detail = ""
        if kpt_counts:
            kpt_detail = "；关键点数量分布: " + ", ".join(
                f"{n}点×{cnt}实例" for n, cnt in sorted(kpt_counts.items())
            )

        split_info = scan.get("splits")
        split_txt = ""
        if split_info:
            split_txt = "划分: " + ", ".join(f"{k}={v}张" for k, v in split_info.items()) + "  |  "

        src = scan.get("class_source", "file")
        if src == "inferred":
            class_txt = "类别(推断): " + "、".join(
                f"{c['name']}({c['count']})" for c in classes
            ) if classes else "无"
        else:
            class_txt = "类别: " + "、".join(
                f"{c['name']}({c['count']})" for c in classes
            ) if classes else "无"

        self.info_lbl.setText(
            f"{split_txt}任务类型: {task_txt}  |  {class_txt}{kpt_detail}"
        )

        class_stats = scan.get("class_stats", {})
        pose_lines_txt = []
        for name in sorted(class_stats.keys()):
            st = class_stats[name]
            if st["pose_count"] > 0:
                pose_lines_txt.append(
                    f"{name}: {st['max_kpt']}kpt×{st['pose_count']}"
                )
        if pose_lines_txt:
            self.info_lbl.setText(
                self.info_lbl.text()
                + "\n位姿类: " + "、".join(pose_lines_txt)
            )

        self._clear_checkboxes()
        max_kpt = scan.get("max_kpt", 0)
        if max_kpt <= 0 and class_stats:
            max_kpt = max(st["max_kpt"] for st in class_stats.values())
        if max_kpt <= 0:
            hint = QLabel("当前数据集无位姿关键点，无需筛选")
            hint.setProperty("role", "hint")
            self._kpt_grid.addWidget(hint, 0, 0)
            return

        cols = min(8, max(1, max_kpt))
        for i in range(max_kpt):
            cb = QCheckBox(f"#{i}")
            if default_first_n is not None:
                cb.setChecked(i < default_first_n)
            else:
                cb.setChecked(True)
            cb.setProperty("role", "mono")
            self._checkboxes.append(cb)
            self._kpt_grid.addWidget(cb, i // cols, i % cols)

    def select_all(self):
        for cb in self._checkboxes:
            cb.setChecked(True)

    def clear_all(self):
        for cb in self._checkboxes:
            cb.setChecked(False)

    def get_keep_indices(self) -> list[int]:
        return [i for i, cb in enumerate(self._checkboxes) if cb.isChecked()]


# =========================================================
# 比例条 — 极简单色 + 香槟金
# =========================================================
class RatioBar(QWidget):
    BAR_H = 4
    HEIGHT = 26

    def __init__(self, ratios, parent=None):
        super().__init__(parent)
        self.set_ratios(ratios)
        self.setFixedHeight(self.HEIGHT)

    def set_ratios(self, ratios):
        t, v, te = ratios
        self._train, self._val, self._test = int(round(t * 100)), int(round(v * 100)), int(round(te * 100))
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w, h = self.width(), self.height()
        bar_y = (h - self.BAR_H) // 2

        # 背景轨道
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1A1A20"))
        p.drawRoundedRect(0, bar_y, w, self.BAR_H, 2, 2)

        total = max(1, self._train + self._val + self._test)
        x = 0
        # 比例色 — 全部走香槟金的不同明度, 不用蓝/红
        segments = [
            (self._train, QColor("#C9A36B"), "train"),
            (self._val,   QColor("#8C8C95"), "val"),
            (self._test,  QColor("#5A5A62"), "test"),
        ]
        for ratio, color, _ in segments:
            if ratio <= 0:
                continue
            seg_w = int(w * ratio / total)
            p.setBrush(color)
            p.drawRect(x, bar_y, seg_w, self.BAR_H)
            x += seg_w


# =========================================================
# 状态点 — 标题栏右侧
# =========================================================
class StatusDot(QWidget):
    def __init__(self, state="idle", parent=None):
        super().__init__(parent)
        self.setProperty("state", state)
        self.setObjectName("statusDot")
        self.setFixedSize(8, 8)

    def set_state(self, state):
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


# =========================================================
# Worker 线程
# =========================================================
class DatasetWorker(QThread):
    log_signal = Signal(str)
    finish_signal = Signal(bool, str)

    def __init__(self, mode, img_dir, ann_dir, out_dir, ratios, keep_kpt_indices=None,
                 force_rm_14_classes=True):
        super().__init__()
        self.mode = mode
        self.img_dir = img_dir
        self.ann_dir = ann_dir if ann_dir else img_dir
        self.out_dir = out_dir
        self.train_r, self.val_r, self.test_r = ratios
        self.keep_kpt_indices = keep_kpt_indices
        self.force_rm_14_classes = force_rm_14_classes

    def log(self, msg):
        self.log_signal.emit(msg)

    def run(self):
        try:
            self.log(f"启动任务 · 模式 = {self.mode}")
            self.log(f"图片目录: {self.img_dir}")
            self.log(f"标签目录: {self.ann_dir}")
            self.log(f"输出目录: {self.out_dir}")
            self.log(f"划分比例: Train {int(self.train_r*100)}% / Val {int(self.val_r*100)}% / Test {int(self.test_r*100)}%")
            self.log("─" * 60)

            if self.mode == "UNET":
                self.process_unet()
            elif self.mode == "YOLO_SPLIT":
                self.process_yolo_split()
            elif self.mode == "TO_YOLO_DET":
                self.process_convert_to_yolo_det()
            elif self.mode == "TO_YOLO_SEG":
                self.process_convert_to_yolo_seg()
            elif self.mode == "TO_YOLO_OBB":
                self.process_convert_to_yolo_obb()
            elif self.mode == "TO_YOLO_POSE":
                self.process_convert_to_yolo_pose()
            elif self.mode == "FROM_YOLO_JSON":
                self.process_from_yolo_json()
            elif self.mode == "FILTER_JSON_KPTS":
                self.process_filter_json_kpts()
            elif self.mode == "KPTS_TO_DET_JSON":
                self.process_kpts_to_det_json()
            elif self.mode == "TO_MLP_CLASSIFY":
                self.process_convert_to_mlp_classify()

        except Exception as e:
            self.log(f"错误: {str(e)}")
            self.finish_signal.emit(False, str(e))

    def get_valid_pairs(self, exts, label_exts, use_json_xml_mix=False):
        valid_pairs = []
        for filename in os.listdir(self.img_dir):
            if filename.lower().endswith(exts):
                base_name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.img_dir, filename)

                if use_json_xml_mix:
                    json_path = os.path.join(self.ann_dir, f"{base_name}.json")
                    xml_path = os.path.join(self.ann_dir, f"{base_name}.xml")
                    if os.path.exists(json_path):
                        valid_pairs.append((img_path, json_path, 'json'))
                    elif os.path.exists(xml_path):
                        valid_pairs.append((img_path, xml_path, 'xml'))
                else:
                    for l_ext in label_exts:
                        l_path = os.path.join(self.ann_dir, f"{base_name}{l_ext}")
                        if os.path.exists(l_path):
                            valid_pairs.append((img_path, l_path))
                            break
        return valid_pairs

    def split_data(self, valid_pairs):
        random.shuffle(valid_pairs)
        total = len(valid_pairs)
        train_c = int(total * self.train_r)
        val_c = int(total * self.val_r)
        train_p = valid_pairs[:train_c]
        val_p = valid_pairs[train_c:train_c + val_c]
        test_p = valid_pairs[train_c + val_c:]
        return train_p, val_p, test_p

    def _ensure_clean_yolo_split_dirs(self):
        """若输出已有 images/{train,val,test}，先清空，避免二次导出同 stem 进多 split。"""
        images_root = os.path.join(self.out_dir, "images")
        labels_root = os.path.join(self.out_dir, "labels")
        existing = False
        for split_name in ("train", "val", "test"):
            idir = os.path.join(images_root, split_name)
            if os.path.isdir(idir) and any(os.scandir(idir)):
                existing = True
                break
        if not existing:
            return
        self.log("检测到已有 train/val/test 输出，正在清空后重新划分…")
        for root in (images_root, labels_root):
            if not os.path.isdir(root):
                continue
            for split_name in ("train", "val", "test"):
                d = os.path.join(root, split_name)
                if os.path.isdir(d):
                    shutil.rmtree(d)

    def _collect_unique_labels(self, valid_pairs, use_json_xml_mix=True):
        """从 valid_pairs 收集标注中出现的全部类名。"""
        unique_classes = set()
        for pair in valid_pairs:
            if use_json_xml_mix and len(pair) >= 3:
                _, l_path, l_type = pair[0], pair[1], pair[2]
                if l_type == 'json':
                    with open(l_path, 'r', encoding='utf-8') as f:
                        for s in json.load(f).get('shapes', []):
                            if s.get('label'):
                                unique_classes.add(s['label'])
                elif l_type == 'xml':
                    for obj in ET.parse(l_path).getroot().findall('object'):
                        if obj.find('name') is not None:
                            unique_classes.add(obj.find('name').text)
            else:
                _, j_path = pair[0], pair[1]
                with open(j_path, 'r', encoding='utf-8') as f:
                    for s in json.load(f).get('shapes', []):
                        if s.get('label'):
                            unique_classes.add(s['label'])
        return unique_classes

    def _build_class_map(self, unique_classes):
        """优先按 ann_dir/classes.txt 行序编号；缺失类字母序追加；无文件则全字母序。"""
        unique_classes = set(unique_classes)
        class_map = {}
        classes_txt = os.path.join(self.ann_dir, "classes.txt")
        if os.path.exists(classes_txt):
            with open(classes_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    if name and name not in class_map:
                        class_map[name] = len(class_map)
            missing = sorted(unique_classes - set(class_map.keys()))
            for name in missing:
                class_map[name] = len(class_map)
            if missing:
                self.log(f"classes.txt 未收录、已字母序追加: {missing}")
        else:
            for name in sorted(unique_classes):
                class_map[name] = len(class_map)
            self.log("未找到 classes.txt，按类名字母序编号")
        # 导出只用标注里出现过的类；保留 classes.txt 中未出现的类以稳定 ID
        return class_map

    # ------------------  转 U-Net ------------------
    def process_unet(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), ('.json',))
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON！")
            return

        self.log(f"找到 {len(valid_pairs)} 组有效数据。")
        class_mapping = {"background": 0}
        self.log("从 JSON 提取类别...")
        unique_classes = set()
        for _, j_path in valid_pairs:
            with open(j_path, 'r', encoding='utf-8') as f:
                for s in json.load(f).get('shapes', []):
                    if s.get('label'): unique_classes.add(s['label'])
        for c in sorted(list(unique_classes)):
            class_mapping[c] = len(class_mapping)
        self.log(f"类别映射: {class_mapping}")

        train_p, val_p, test_p = self.split_data(valid_pairs)
        self._ensure_clean_yolo_split_dirs()

        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'masks', split_name), exist_ok=True)
            self.log(f"生成 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, json_p in pairs:
                base_name = os.path.splitext(os.path.basename(img_p))[0]
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                with open(json_p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mask = np.zeros((data.get('imageHeight', 100), data.get('imageWidth', 100)), dtype=np.uint8)
                for shape in data.get('shapes', []):
                    cid = class_mapping.get(shape.get('label'))
                    if cid is None: continue
                    pts = np.array(shape.get('points', []), np.int32)
                    if shape.get('shape_type') == 'rectangle' and len(pts) == 2:
                        cv2.rectangle(mask, pts[0], pts[1], cid, -1)
                    elif len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], cid)
                    elif shape.get('shape_type') == 'point':
                        cv2.circle(mask, pts[0], 5, cid, -1)

                cv2.imencode('.png', mask)[1].tofile(
                    os.path.join(self.out_dir, 'masks', split_name, f"{base_name}.png"))

        self.log("")
        self.log("U-Net 格式数据集生成完毕")
        self.finish_signal.emit(True, "U-Net 数据集处理成功")

    # ------------------ MLP 分类 ------------------
    def process_convert_to_mlp_classify(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp', '.jpeg'), [], use_json_xml_mix=True)
        valid_pairs = [(img, lpath) for img, lpath, ltype in valid_pairs if ltype == 'json']
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON！")
            return

        self.log(f"找到 {len(valid_pairs)} 组有效数据，读取 mlp_label...")

        ann_dir = os.path.dirname(valid_pairs[0][1]) if valid_pairs else ''
        label_txt_path = os.path.join(ann_dir, 'label.txt') if ann_dir else ''
        name_to_folder = {}
        if label_txt_path and os.path.exists(label_txt_path):
            with open(label_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    name, folder = line.split(':', 1)
                    name = name.strip()
                    folder = folder.strip()
                    if name and folder:
                        name_to_folder[name] = folder
            self.log(f"找到 label.txt, 加载 {len(name_to_folder)} 个类别映射")

        next_idx = 0
        for img_path, json_path in valid_pairs:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mlp_label = data.get('mlp_label')
                if not mlp_label:
                    continue
                break
            except Exception:
                continue
            break

        folder_names_used = set(name_to_folder.values())
        next_idx = 0
        for name in list(folder_names_used):
            if name.startswith('idx_'):
                try:
                    idx = int(name.replace('idx_', ''))
                    next_idx = max(next_idx, idx + 1)
                except ValueError:
                    pass

        label_to_folder = {}
        for name, folder in name_to_folder.items():
            label_to_folder[name] = folder

        os.makedirs(self.out_dir, exist_ok=True)
        saved_count = 0

        for img_path, json_path in valid_pairs:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mlp_label = data.get('mlp_label')
                if not mlp_label:
                    continue

                if mlp_label not in label_to_folder:
                    folder_name = f'idx_{next_idx}'
                    label_to_folder[mlp_label] = folder_name
                    next_idx += 1
                else:
                    folder_name = label_to_folder[mlp_label]

                out_folder = os.path.join(self.out_dir, folder_name)
                os.makedirs(out_folder, exist_ok=True)

                img = cv2.imread(img_path)
                if img is None:
                    self.log(f"无法读取图片: {img_path}")
                    continue

                ext = os.path.splitext(img_path)[1]
                out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}{ext}"
                out_path = os.path.join(out_folder, out_name)
                cv2.imwrite(out_path, img)
                saved_count += 1

            except Exception as e:
                self.log(f"处理 {img_path} 出错: {e}")

        sorted_labels = sorted(label_to_folder.keys(), key=lambda x: label_to_folder.get(x, ''))
        label_lines = []
        for label in sorted_labels:
            fname = label_to_folder.get(label, '')
            label_lines.append(f"{fname}: {label}")
        label_path = os.path.join(self.out_dir, "label.txt")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(label_lines))

        self.log(f"类别映射: {label_path}")
        self.log(f"共 {len(sorted_labels)} 个类别, {saved_count} 张图片")
        self.log("")
        self.log("MLP 分类数据集导出完毕")
        self.finish_signal.emit(True, f"MLP 分类数据集处理成功, {saved_count} 张图片")

    # ------------------ 生成 data.yaml ------------------
    def generate_yaml(self, class_map, kpt_shape=None):
        yaml_path = os.path.join(self.out_dir, "data.yaml")
        abs_out_dir = os.path.abspath(self.out_dir).replace("\\", "/")

        nc = len(class_map)
        names_dict = {v: k for k, v in class_map.items()}
        yaml_content = []
        yaml_content.append(f"# 数据路径")
        yaml_content.append(f"path: {abs_out_dir}")
        yaml_content.append(f"train: images/train")
        yaml_content.append(f"val: images/val")
        yaml_content.append(f"test: images/test")

        yaml_content.append(f"\n# Number of classes")
        yaml_content.append(f"nc: {nc}")

        if kpt_shape is not None:
            yaml_content.append(f"\n# Keypoints (for pose training)")
            yaml_content.append(f"kpt_shape: {kpt_shape}")
            flip_idx = list(range(kpt_shape[0]))
            yaml_content.append(f"flip_idx: {flip_idx}")

        yaml_content.append(f"\n# Class names")
        yaml_content.append(f"names:")
        for i in range(nc):
            yaml_content.append(f"  {i}: {names_dict[i]}")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yaml_content))

        self.log(f"配置文件: {yaml_path}")

    # ------------------ YOLO 纯划分 ------------------
    def process_yolo_split(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), ('.txt',))
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和TXT！")
            return

        class_map = {}
        classes_txt = os.path.join(self.ann_dir, "classes.txt")
        if os.path.exists(classes_txt):
            with open(classes_txt, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip(): class_map[line.strip()] = i
        else:
            class_map = {"class_0": 0}

        self.log(f"找到 {len(valid_pairs)} 组有效数据。")
        train_p, val_p, test_p = self.split_data(valid_pairs)
        self._ensure_clean_yolo_split_dirs()

        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"拷贝 {split_name} 集 ({len(pairs)} 张)...")
            for img_p, txt_p in pairs:
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))
                shutil.copy(txt_p, os.path.join(self.out_dir, 'labels', split_name, os.path.basename(txt_p)))

        self.generate_yaml(class_map)
        self.log("")
        self.log("YOLO 数据集划分完毕")
        self.finish_signal.emit(True, "YOLO 划分处理成功")

    # ------------------ YOLO 检测 ------------------
    def process_convert_to_yolo_det(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), [], use_json_xml_mix=True)
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON/XML！")
            return

        self.log(f"找到 {len(valid_pairs)} 组有效数据，检查标注类型...")
        allowed_types = {'rectangle', 'point'}
        incompatible = {}
        for _, l_path, l_type in valid_pairs:
            if l_type == 'json':
                with open(l_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for s in data.get('shapes', []):
                        st = s.get('shape_type', 'polygon') if s.get('points') else 'unknown'
                        if st not in allowed_types and s.get('label'):
                            incompatible[st] = incompatible.get(st, 0) + 1
        if incompatible:
            msg = "YOLO检测格式只支持矩形和点标注！\n发现以下不兼容的类型：\n"
            for k, v in incompatible.items():
                type_name = {'obb': '旋转矩形(OBB)', 'polygon': '多边形', 'pose': '位姿'}.get(k, k)
                msg += f"  - {type_name}: {v}个\n"
            self.finish_signal.emit(False, msg.strip())
            return

        class_map = self._build_class_map(self._collect_unique_labels(valid_pairs))
        self.log(f"类别映射: {class_map}")

        train_p, val_p, test_p = self.split_data(valid_pairs)
        self._ensure_clean_yolo_split_dirs()
        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"处理 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, label_p, l_type in pairs:
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                lines = []
                if l_type == 'json':
                    with open(label_p, 'r', encoding='utf-8') as f:
                        for s in json.load(f).get('shapes', []):
                            cid = class_map.get(s.get('label'))
                            if cid is None or not s.get('points'): continue
                            pts = s['points']
                            shape_type = s.get('shape_type', 'polygon')
                            if shape_type == 'rectangle' and len(pts) == 2:
                                cx, cy = (pts[0][0] + pts[1][0]) / 2 / w, (pts[0][1] + pts[1][1]) / 2 / h
                                bw, bh = abs(pts[1][0] - pts[0][0]) / w, abs(pts[1][1] - pts[0][1]) / h
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                            elif shape_type == 'point' and len(pts) == 1:
                                cx, cy = pts[0][0] / w, pts[0][1] / h
                                pw, ph = 0.02, 0.02
                                cx = max(pw / 2, min(1.0 - pw / 2, cx))
                                cy = max(ph / 2, min(1.0 - pw / 2, cy))
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {pw:.6f} {ph:.6f}")
                elif l_type == 'xml':
                    root = ET.parse(label_p).getroot()
                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        cid = class_map.get(label)
                        if cid is None: continue
                        bndbox = obj.find('bndbox')
                        if bndbox is not None:
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            cx, cy = (xmin + xmax) / 2.0 / w, (ymin + ymax) / 2.0 / h
                            bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                txt_out = os.path.join(self.out_dir, 'labels', split_name,
                                       os.path.splitext(os.path.basename(img_p))[0] + ".txt")
                with open(txt_out, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

        with open(os.path.join(self.out_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for c in sorted(class_map.keys(), key=lambda k: class_map[k]): f.write(f"{c}\n")
        self.generate_yaml(class_map)
        self.log("")
        self.log("YOLO 检测格式导出完成")
        self.finish_signal.emit(True, "YOLO 检测格式导出成功")

    # ------------------ YOLO 分割 ------------------
    def process_convert_to_yolo_seg(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), [], use_json_xml_mix=True)
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON/XML！")
            return

        self.log(f"找到 {len(valid_pairs)} 组有效数据，检查标注类型...")
        allowed_types = {'rectangle', 'polygon'}
        incompatible = {}
        for _, l_path, l_type in valid_pairs:
            if l_type == 'json':
                with open(l_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for s in data.get('shapes', []):
                        st = s.get('shape_type', 'polygon') if s.get('points') else 'unknown'
                        if st not in allowed_types and s.get('label'):
                            incompatible[st] = incompatible.get(st, 0) + 1
        if incompatible:
            msg = "YOLO分割格式只支持矩形和多边形标注！\n发现以下不兼容的类型：\n"
            for k, v in incompatible.items():
                type_name = {'obb': '旋转矩形(OBB)', 'point': '点', 'pose': '位姿'}.get(k, k)
                msg += f"  - {type_name}: {v}个\n"
            self.finish_signal.emit(False, msg.strip())
            return

        class_map = self._build_class_map(self._collect_unique_labels(valid_pairs))
        self.log(f"类别映射: {class_map}")

        train_p, val_p, test_p = self.split_data(valid_pairs)
        self._ensure_clean_yolo_split_dirs()
        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"处理 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, label_p, l_type in pairs:
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                lines = []
                if l_type == 'json':
                    with open(label_p, 'r', encoding='utf-8') as f:
                        for s in json.load(f).get('shapes', []):
                            cid = class_map.get(s.get('label'))
                            if cid is None or not s.get('points'): continue
                            pts = s['points']
                            shape_type = s.get('shape_type', 'polygon')
                            if shape_type == 'rectangle' and len(pts) == 2:
                                x1, y1 = pts[0]
                                x2, y2 = pts[1]
                                lines.append(f"{cid} {x1/w:.6f} {y1/h:.6f} {x2/w:.6f} {y1/h:.6f} {x2/w:.6f} {y2/h:.6f} {x1/w:.6f} {y2/h:.6f}")
                            elif shape_type == 'polygon':
                                pts_normalized = []
                                for pt in pts:
                                    pts_normalized.extend([f"{pt[0] / w:.6f}", f"{pt[1] / h:.6f}"])
                                lines.append(f"{cid} " + " ".join(pts_normalized))
                elif l_type == 'xml':
                    root = ET.parse(label_p).getroot()
                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        cid = class_map.get(label)
                        if cid is None: continue
                        bndbox = obj.find('bndbox')
                        polygon = obj.find('polygon')
                        if bndbox is not None:
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            lines.append(f"{cid} {xmin/w:.6f} {ymin/h:.6f} {xmax/w:.6f} {ymin/h:.6f} {xmax/w:.6f} {ymax/h:.6f} {xmin/w:.6f} {ymax/h:.6f}")
                        elif polygon is not None:
                            pts_normalized = []
                            for pt in polygon.findall('pt'):
                                px = float(pt.find('x').text) / w
                                py = float(pt.find('y').text) / h
                                pts_normalized.extend([f"{px:.6f}", f"{py:.6f}"])
                            if pts_normalized:
                                lines.append(f"{cid} " + " ".join(pts_normalized))

                txt_out = os.path.join(self.out_dir, 'labels', split_name,
                                       os.path.splitext(os.path.basename(img_p))[0] + ".txt")
                with open(txt_out, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

        with open(os.path.join(self.out_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for c in sorted(class_map.keys(), key=lambda k: class_map[k]): f.write(f"{c}\n")
        self.generate_yaml(class_map)
        self.log("")
        self.log("YOLO 分割格式导出完成")
        self.finish_signal.emit(True, "YOLO 分割格式导出成功")

    # ------------------ YOLO-OBB ------------------
    def process_convert_to_yolo_obb(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), [], use_json_xml_mix=True)
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON/XML！")
            return

        self.log(f"找到 {len(valid_pairs)} 组有效数据，检查标注类型...")
        allowed_types = {'obb'}
        incompatible = {}
        for _, l_path, l_type in valid_pairs:
            if l_type == 'json':
                with open(l_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for s in data.get('shapes', []):
                        st = s.get('shape_type', 'polygon') if s.get('points') else 'unknown'
                        if st not in allowed_types and s.get('label'):
                            incompatible[st] = incompatible.get(st, 0) + 1
        if incompatible:
            msg = "YOLO-OBB格式只支持旋转矩形标注！\n发现以下不兼容的类型：\n"
            for k, v in incompatible.items():
                type_name = {'rectangle': '矩形', 'polygon': '多边形', 'point': '点', 'pose': '位姿'}.get(k, k)
                msg += f"  - {type_name}: {v}个\n"
            self.finish_signal.emit(False, msg.strip())
            return

        class_map = self._build_class_map(self._collect_unique_labels(valid_pairs))
        self.log(f"类别映射: {class_map}")

        train_p, val_p, test_p = self.split_data(valid_pairs)
        self._ensure_clean_yolo_split_dirs()
        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"处理 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, label_p, l_type in pairs:
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                lines = []
                if l_type == 'json':
                    with open(label_p, 'r', encoding='utf-8') as f:
                        for s in json.load(f).get('shapes', []):
                            cid = class_map.get(s.get('label'))
                            if cid is None or not s.get('points'): continue
                            pts = s['points']
                            shape_type = s.get('shape_type', 'polygon')
                            if shape_type == 'obb' and len(pts) >= 4:
                                obb_pts = pts[:4]
                                pts_normalized = []
                                for pt in obb_pts:
                                    pts_normalized.extend([f"{pt[0] / w:.6f}", f"{pt[1] / h:.6f}"])
                                lines.append(f"{cid} " + " ".join(pts_normalized))
                elif l_type == 'xml':
                    root = ET.parse(label_p).getroot()
                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        cid = class_map.get(label)
                        if cid is None: continue
                        robndbox = obj.find('robndbox')
                        if robndbox is not None:
                            cx = float(robndbox.find('cx').text) / w
                            cy = float(robndbox.find('cy').text) / h
                            bw = float(robndbox.find('w').text) / w
                            bh = float(robndbox.find('h').text) / h
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                txt_out = os.path.join(self.out_dir, 'labels', split_name,
                                       os.path.splitext(os.path.basename(img_p))[0] + ".txt")
                with open(txt_out, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

        with open(os.path.join(self.out_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for c in sorted(class_map.keys(), key=lambda k: class_map[k]): f.write(f"{c}\n")
        self.generate_yaml(class_map)
        self.log("")
        self.log("YOLO-OBB 格式导出完成")
        self.finish_signal.emit(True, "YOLO-OBB 格式导出成功")

    # ------------------ YOLO-Pose ------------------
    def process_convert_to_yolo_pose(self):
        valid_pairs = self.get_valid_pairs(('.jpg', '.png', '.bmp'), [], use_json_xml_mix=True)
        if not valid_pairs:
            self.finish_signal.emit(False, "未找到成对的图片和JSON/XML！")
            return

        self.log(f"找到 {len(valid_pairs)} 组有效数据，检查标注类型...")
        allowed_types = {'pose', 'hierarchy_pose'}
        incompatible = {}
        for _, l_path, l_type in valid_pairs:
            if l_type == 'json':
                with open(l_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for s in data.get('shapes', []):
                        st = s.get('shape_type', 'polygon') if s.get('points') else 'unknown'
                        if st not in allowed_types and s.get('label'):
                            incompatible[st] = incompatible.get(st, 0) + 1
        if incompatible:
            msg = "YOLO-Pose格式只支持位姿/层级标注！\n发现以下不兼容的类型：\n"
            for k, v in incompatible.items():
                type_name = {'rectangle': '矩形', 'polygon': '多边形', 'point': '点', 'obb': '旋转矩形'}.get(k, k)
                msg += f"  - {type_name}: {v}个\n"
            self.finish_signal.emit(False, msg.strip())
            return

        from core.constants import HIER_CLASS_NAMES, BUFF_CLASS_NAMES, BUFF_ALIAS_MAP
        hier_set = set(HIER_CLASS_NAMES)
        buff_set = set(BUFF_CLASS_NAMES)
        rm_valid_set = hier_set | buff_set | set(BUFF_ALIAS_MAP.keys())

        unique_classes = set()
        max_kpt_count = 0
        # label -> count / sample path (for non-RM warnings)
        non_rm_counts = {}
        non_rm_samples = {}
        hier_instance_count = 0
        buff_instance_count = 0
        for _, l_path, l_type in valid_pairs:
            if l_type == 'json':
                with open(l_path, 'r', encoding='utf-8') as f:
                    for s in json.load(f).get('shapes', []):
                        label = s.get('label')
                        if label:
                            norm_label = BUFF_ALIAS_MAP.get(label, label)
                            unique_classes.add(norm_label)
                            st = s.get('shape_type', 'polygon') if s.get('points') else 'unknown'
                            if st in allowed_types:
                                if norm_label in hier_set:
                                    hier_instance_count += 1
                                elif norm_label in buff_set:
                                    buff_instance_count += 1
                                else:
                                    non_rm_counts[label] = non_rm_counts.get(label, 0) + 1
                                    if label not in non_rm_samples:
                                        non_rm_samples[label] = os.path.basename(l_path)
                        st = s.get('shape_type', 'polygon') if s.get('points') else 'unknown'
                        if st in ('pose', 'hierarchy_pose') and s.get('keypoints'):
                            max_kpt_count = max(max_kpt_count, len(s['keypoints']))
            elif l_type == 'xml':
                for obj in ET.parse(l_path).getroot().findall('object'):
                    if obj.find('name') is not None:
                        raw_name = obj.find('name').text
                        norm_name = BUFF_ALIAS_MAP.get(raw_name, raw_name)
                        unique_classes.add(norm_name)

        force_rm = bool(getattr(self, "force_rm_14_classes", True))
        if force_rm:
            total_valid_rm = hier_instance_count + buff_instance_count
            if total_valid_rm == 0:
                msg = (
                    "强制 RM 标准类失败：源标注中没有任何 BG/B1/…/RB 或 RT/BT 类名。\n"
                    f"扫描到的类: {sorted(unique_classes)}\n"
                    "请改选已用 RM-mode / BUFF 标好的目录，或取消勾选「强制 RM 标准类」。"
                )
                self.log(msg)
                self.finish_signal.emit(False, msg)
                return

            if hier_instance_count > 0 and buff_instance_count == 0:
                class_map = {name: i for i, name in enumerate(HIER_CLASS_NAMES)}
                self.log(f"强制 RM 兵种 14 类映射 (BG=0): {list(HIER_CLASS_NAMES)}")
            elif hier_instance_count == 0 and buff_instance_count > 0:
                class_map = {"RT": 0, "BT": 1}
                self.log("检测到能量机关 (BUFF) 位姿标签，使用 BUFF 2 类映射 (RT=0, BT=1)")
            else:
                class_map = {name: i for i, name in enumerate(HIER_CLASS_NAMES)}
                class_map["RT"] = 14
                class_map["BT"] = 15
                self.log("检测到装甲板 + BUFF 混合位姿标签，使用 RM 16 类映射 (0..13 为装甲板, 14:RT, 15:BT)")

            if non_rm_counts:
                self.log("⚠ 以下非标准类名将被跳过（无法映射到 BG…RB 或 RT/BT）：")
                for name, cnt in sorted(non_rm_counts.items(), key=lambda x: -x[1]):
                    sample = non_rm_samples.get(name, "?")
                    self.log(f"  - {name}: {cnt} 个实例 (例: {sample})")
                legacy = {"red_armor", "blue_armor", "redarmor", "bluearmor"}
                if set(non_rm_counts).issubset(legacy) or any(n in legacy for n in non_rm_counts):
                    self.log(
                        "提示: 源目录仍是 red_armor/blue_armor 旧标注时，"
                        "请改选已用 RM-mode 标好的目录（如 datasets/pose_4_shenzhen_yolo / demo-pose）"
                    )
        else:
            if unique_classes and unique_classes.issubset(buff_set):
                class_map = {"RT": 0, "BT": 1}
                self.log("检测到 BUFF 能量机关标签，使用 BUFF 2 类映射 (RT=0, BT=1)")
            elif unique_classes and unique_classes.issubset(hier_set):
                class_map = {name: i for i, name in enumerate(HIER_CLASS_NAMES)}
                self.log(f"检测到 RM 兵种标签，使用固定 14 类映射 (BG=0): {list(HIER_CLASS_NAMES)}")
            else:
                class_map = self._build_class_map(unique_classes)
                self.log("未强制 RM 标准类，优先 classes.txt 行序，否则字母序")

        self.log(f"类别映射: {class_map}")
        self.log(f"最大关键点数量: {max_kpt_count}")

        written_instances = 0
        skipped_non_hier = 0
        train_p, val_p, test_p = self.split_data(valid_pairs)
        self._ensure_clean_yolo_split_dirs()
        for split_name, pairs in [('train', train_p), ('val', val_p), ('test', test_p)]:
            if not pairs: continue
            os.makedirs(os.path.join(self.out_dir, 'images', split_name), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'labels', split_name), exist_ok=True)
            self.log(f"处理 {split_name} 集 ({len(pairs)} 张)...")

            for img_p, label_p, l_type in pairs:
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]
                shutil.copy(img_p, os.path.join(self.out_dir, 'images', split_name, os.path.basename(img_p)))

                lines = []
                if l_type == 'json':
                    with open(label_p, 'r', encoding='utf-8') as f:
                        for s in json.load(f).get('shapes', []):
                            raw_label = s.get('label')
                            norm_label = BUFF_ALIAS_MAP.get(raw_label, raw_label) if raw_label else None
                            cid = class_map.get(norm_label)
                            if cid is None and raw_label:
                                cid = class_map.get(raw_label)
                            if cid is None:
                                if force_rm and raw_label and norm_label not in rm_valid_set:
                                    skipped_non_hier += 1
                                continue
                            if not s.get('points'):
                                continue
                            pts = s['points']
                            shape_type = s.get('shape_type', 'polygon')
                            if shape_type in ('pose', 'hierarchy_pose') and len(pts) == 2:
                                x1, y1 = pts[0]
                                x2, y2 = pts[1]
                                cx = (x1 + x2) / 2 / w
                                cy = (y1 + y2) / 2 / h
                                bw = abs(x2 - x1) / w
                                bh = abs(y2 - y1) / h
                                kpts = s.get('keypoints', [])
                                parts = [f"{cid}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
                                for kpt in kpts:
                                    kx = kpt[0] / w
                                    ky = kpt[1] / h
                                    kv = kpt[2] if len(kpt) > 2 else 2
                                    parts.append(f"{kx:.6f} {ky:.6f} {int(kv)}")
                                lines.append(" ".join(parts))
                                written_instances += 1

                txt_out = os.path.join(self.out_dir, 'labels', split_name,
                                       os.path.splitext(os.path.basename(img_p))[0] + ".txt")
                with open(txt_out, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

        if force_rm and written_instances == 0:
            msg = (
                "强制 RM 标准类失败：没有任何实例被写入。\n"
                "源目录可能全是 red_armor/blue_armor 等旧类名。\n"
                "请改选已用 RM-mode / BUFF 标好的目录，或取消勾选「强制 RM 标准类」。"
            )
            self.log(msg)
            self.finish_signal.emit(False, msg)
            return

        with open(os.path.join(self.out_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for c in sorted(class_map.keys(), key=lambda k: class_map[k]):
                f.write(f"{c}\n")
        kpt_shape = [max_kpt_count, 3] if max_kpt_count > 0 else None
        self.generate_yaml(class_map, kpt_shape=kpt_shape)

        # 便于主程序再次打开 YOLO 数据集时同步位姿定义
        try:
            buff_kpt_names = ["R_center", "target_inner", "bottom_side", "outer_axis", "top_side"]
            armor_kpt_names = ["TL", "TR", "BL", "BR"]
            categories = {}
            for name in class_map.keys():
                if name in ("RT", "BT"):
                    categories[name] = {"kpt_count": 5, "keypoints": list(buff_kpt_names)}
                else:
                    categories[name] = {"kpt_count": 4, "keypoints": list(armor_kpt_names)}
            pose_cfg = {"categories": categories}
            pose_path = os.path.join(self.out_dir, "pose_classes.json")
            with open(pose_path, "w", encoding="utf-8") as pf:
                json.dump(pose_cfg, pf, ensure_ascii=False, indent=2)
            self.log(f"已写入 pose_classes.json → {pose_path}")
        except Exception as exc:
            self.log(f"写 pose_classes.json 失败: {exc}")

        self.log("")
        self.log(f"写入实例: {written_instances}" + (f"，跳过非标准: {skipped_non_hier}" if skipped_non_hier else ""))
        self.log("YOLO-Pose 格式导出完成")
        self.finish_signal.emit(True, "YOLO-Pose 格式导出成功")

    # ------------------ YOLO → JSON ------------------
    def process_from_yolo_json(self):
        from core.yolo_to_json import (
            collect_yolo_split_pairs,
            normalize_yolo_dataset_dirs,
            resolve_yolo_class_names,
            iter_txt_paths_from_splits,
            yolo_txt_to_shapes,
            save_labelme_json,
            scan_class_pose_stats,
            load_yaml_kpt_shape,
            organize_dataset_categories,
            build_output_pose_config,
            write_dataset_meta_files,
        )
        from core.pose_config import PoseConfig

        global_pose_path = os.path.join(_ROOT, "config", "pose_classes.json")
        img_dir, label_dir, dataset_root = normalize_yolo_dataset_dirs(
            self.img_dir, self.ann_dir,
        )
        self.log(f"数据集根目录: {dataset_root}")
        self.log(f"images: {img_dir}")
        self.log(f"labels: {label_dir}")

        split_groups = collect_yolo_split_pairs(img_dir, label_dir)
        if not split_groups:
            self.finish_signal.emit(
                False,
                "未找到 YOLO 图片/标签配对！\n"
                "请选择 images 与 labels 目录（含 train/val/test 子目录），"
                "或选择对应的 split 子目录。",
            )
            return

        txt_paths = iter_txt_paths_from_splits(split_groups)
        class_names, name_source, meta_file = resolve_yolo_class_names(
            img_dir, label_dir, txt_paths,
        )
        if not class_names:
            self.finish_signal.emit(False, "标签文件中未找到有效的 class_id")
            return

        class_stats = scan_class_pose_stats(txt_paths, class_names)
        yaml_kpt = load_yaml_kpt_shape(meta_file)

        pose_config = PoseConfig(global_pose_path)
        pose_config.load()
        org = organize_dataset_categories(
            pose_config, class_stats, class_names,
            [dataset_root, label_dir],
            yaml_kpt_shape=yaml_kpt,
            save_global_path=global_pose_path,
        )
        class_names = org["class_names"]
        if org["added"]:
            self.log(f"新增位姿类别: {org['added']}")
        if org["removed"]:
            self.log(f"删除多余位姿类别: {org['removed']}")
        if org["updated"]:
            self.log(f"更新关键点数: {org['updated']}")

        output_pose = build_output_pose_config(pose_config, class_stats)

        for name, st in class_stats.items():
            if st["pose_count"] > 0:
                kpt_n = pose_config.get_kpt_count(name)
                self.log(
                    f"  位姿类 [{name}]: {st['pose_count']} 实例, "
                    f"{kpt_n} 关键点"
                )
            elif st["det_count"] > 0:
                self.log(f"  检测类 [{name}]: {st['det_count']} 实例")

        keep = self.keep_kpt_indices
        if keep is not None:
            self.log(f"保留关键点序号: {keep}")

        if name_source == "file":
            meta_hint = f" ({os.path.basename(meta_file)})" if meta_file else ""
            self.log(f"类别 ({len(class_names)}，来自 data.yaml/classes.txt{meta_hint}): {class_names}")
        else:
            self.log(
                f"未找到 classes.txt / data.yaml，已从标签推断类别 ({len(class_names)}): "
                f"{class_names}"
            )
            self.log("提示: 转换后可在 JSON 中批量替换 class_0 等为真实类别名")
        split_summary = ", ".join(f"{n}={len(p)}" for n, p in split_groups)
        self.log(f"识别划分: {split_summary}")
        total_pairs = sum(len(p) for _, p in split_groups)
        self.log(f"共 {total_pairs} 组图片+标签")

        total_saved = 0
        os.makedirs(self.out_dir, exist_ok=True)
        for split_name, pairs in split_groups:
            self.log(f"处理 {split_name} ({len(pairs)} 张)...")

            for img_p, txt_p in pairs:
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None:
                    self.log(f"无法读取: {img_p}")
                    continue
                h, w = img.shape[:2]
                base = os.path.basename(img_p)
                out_img = os.path.join(self.out_dir, base)
                shutil.copy(img_p, out_img)

                shapes = yolo_txt_to_shapes(
                    txt_p, w, h, class_names,
                    keep_kpt_indices=keep,
                    pose_config=pose_config,
                )
                json_path = os.path.join(
                    self.out_dir, os.path.splitext(base)[0] + ".json"
                )
                save_labelme_json(json_path, out_img, w, h, shapes)
                total_saved += 1

        split_names = [n for n, _ in split_groups]
        write_dataset_meta_files(
            self.out_dir, split_names, class_names, output_pose,
        )
        self.log(f"已写入 classes.txt / pose_classes.json → {self.out_dir}")

        self.log("")
        self.log(f"YOLO → JSON 完成，共 {total_saved} 张")
        self.finish_signal.emit(True, f"YOLO → JSON 成功，{total_saved} 张")

    # ------------------ JSON 关键点筛选 ------------------
    def process_filter_json_kpts(self):
        from core.yolo_to_json import filter_pose_json_shapes, collect_json_split_pairs

        if self.keep_kpt_indices is None:
            self.finish_signal.emit(False, "请先分析数据集并选择要保留的关键点序号")
            return

        split_groups = collect_json_split_pairs(self.img_dir, self.ann_dir or self.img_dir)
        if not split_groups:
            self.finish_signal.emit(False, "未找到 JSON 文件！")
            return

        keep = self.keep_kpt_indices
        self.log(f"保留关键点序号: {keep}")

        if len(split_groups) == 1 and split_groups[0][0] == "all":
            pairs = split_groups[0][1]
            self.log(f"找到 {len(pairs)} 个 JSON（扁平目录，按比例划分）")
            train_p, val_p, test_p = self.split_data(pairs)
            split_groups = [
                ("train", train_p), ("val", val_p), ("test", test_p),
            ]
            split_groups = [(n, p) for n, p in split_groups if p]
        else:
            total = sum(len(p) for _, p in split_groups)
            split_summary = ", ".join(f"{n}={len(p)}" for n, p in split_groups)
            self.log(f"识别划分: {split_summary}，共 {total} 个 JSON")

        total_saved = 0
        os.makedirs(self.out_dir, exist_ok=True)

        for split_name, pairs in split_groups:
            if not pairs:
                continue
            self.log(f"处理 {split_name} ({len(pairs)} 张)...")

            for img_p, json_p in pairs:
                json_base = os.path.basename(json_p)
                out_json = os.path.join(self.out_dir, json_base)

                if img_p and os.path.isfile(img_p):
                    img_data = np.fromfile(img_p, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    h, w = (img.shape[:2] if img is not None else (0, 0))
                    base_img = os.path.basename(img_p)
                    shutil.copy(img_p, os.path.join(self.out_dir, base_img))
                    image_path = base_img
                else:
                    with open(json_p, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    h = int(existing.get('imageHeight', 0))
                    w = int(existing.get('imageWidth', 0))
                    image_path = existing.get('imagePath', json_base.replace('.json', '.jpg'))

                with open(json_p, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                shapes = filter_pose_json_shapes(data.get('shapes', []), keep)
                out_data = dict(data)
                out_data['shapes'] = shapes
                out_data['imagePath'] = image_path
                out_data['imageWidth'] = int(w)
                out_data['imageHeight'] = int(h)

                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(out_data, f, ensure_ascii=False, indent=2)
                total_saved += 1

        self.log("")
        self.log(f"JSON 关键点筛选完成，共 {total_saved} 张")
        self.finish_signal.emit(True, f"关键点筛选成功，{total_saved} 张")

    # ------------------ 四点外接框 → 检测 JSON ------------------
    def process_kpts_to_det_json(self):
        from core.yolo_to_json import (
            collect_yolo_split_pairs,
            normalize_yolo_dataset_dirs,
            resolve_yolo_class_names,
            iter_txt_paths_from_splits,
            yolo_txt_to_shapes,
            save_labelme_json,
            collect_json_split_pairs,
            convert_pose_shapes_to_detect_only,
            write_classes_txt,
            _is_pose_shape,
        )
        from core.pose_config import PoseConfig

        if self.keep_kpt_indices is None or len(self.keep_kpt_indices) != 4:
            self.finish_signal.emit(False, "请选择恰好 4 个关键点序号")
            return

        indices = self.keep_kpt_indices
        self.log(f"外接框关键点序号: {indices}")

        img_dir, label_dir, dataset_root = normalize_yolo_dataset_dirs(
            self.img_dir, self.ann_dir,
        )
        split_groups = collect_yolo_split_pairs(img_dir, label_dir)
        if split_groups:
            self.log(f"输入类型: YOLO-Pose（{dataset_root}）")
            self._kpts_to_det_from_yolo(split_groups, img_dir, label_dir, indices)
            return

        ann_dir = self.ann_dir or self.img_dir
        split_groups = collect_json_split_pairs(self.img_dir, ann_dir)
        if split_groups:
            self.log("输入类型: 位姿 JSON")
            self._kpts_to_det_from_json(split_groups, indices)
            return

        self.finish_signal.emit(
            False,
            "未找到 YOLO 或 JSON 标注配对！\n"
            "YOLO: 选择含 images/ 与 labels/ 的数据集根目录\n"
            "JSON: 选择含 .json 标注的目录",
        )

    def _kpts_to_det_from_yolo(
        self,
        split_groups: list,
        img_dir: str,
        label_dir: str,
        indices: list[int],
    ) -> None:
        from core.yolo_to_json import (
            iter_txt_paths_from_splits,
            resolve_yolo_class_names,
            yolo_txt_to_shapes,
            save_labelme_json,
            convert_pose_shapes_to_detect_only,
            write_classes_txt,
            _is_pose_shape,
        )
        from core.pose_config import PoseConfig

        global_pose_path = os.path.join(_ROOT, "config", "pose_classes.json")
        pose_config = PoseConfig(global_pose_path)
        pose_config.load()

        txt_paths = iter_txt_paths_from_splits(split_groups)
        class_names, name_source, meta_file = resolve_yolo_class_names(
            img_dir, label_dir, txt_paths,
        )
        if not class_names:
            self.finish_signal.emit(False, "标签文件中未找到有效的 class_id")
            return

        if name_source == "file":
            meta_hint = f" ({os.path.basename(meta_file)})" if meta_file else ""
            self.log(f"类别 ({len(class_names)}，来自 data.yaml/classes.txt{meta_hint}): {class_names}")
        else:
            self.log(f"类别 (推断, {len(class_names)}): {class_names}")

        split_summary = ", ".join(f"{n}={len(p)}" for n, p in split_groups)
        self.log(f"识别划分: {split_summary}")

        total_saved = 0
        converted = 0
        skipped_non_pose = 0
        skipped_no_valid = 0
        os.makedirs(self.out_dir, exist_ok=True)

        for split_name, pairs in split_groups:
            self.log(f"处理 {split_name} ({len(pairs)} 张)...")
            for img_p, txt_p in pairs:
                img_data = np.fromfile(img_p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None:
                    self.log(f"无法读取: {img_p}")
                    continue
                h, w = img.shape[:2]
                base = os.path.basename(img_p)
                out_img = os.path.join(self.out_dir, base)
                shutil.copy(img_p, out_img)

                shapes = yolo_txt_to_shapes(
                    txt_p, w, h, class_names, pose_config=pose_config,
                )
                pose_n = sum(1 for s in shapes if _is_pose_shape(s))
                skipped_non_pose += len(shapes) - pose_n
                det_shapes = convert_pose_shapes_to_detect_only(shapes, indices)
                skipped_no_valid += pose_n - len(det_shapes)
                converted += len(det_shapes)

                json_path = os.path.join(
                    self.out_dir, os.path.splitext(base)[0] + ".json",
                )
                save_labelme_json(json_path, out_img, w, h, det_shapes)
                total_saved += 1

        write_classes_txt(self.out_dir, class_names)
        self.log(f"已写入 classes.txt → {self.out_dir}")
        self.log("")
        self.log(
            f"转换完成: {total_saved} 张, "
            f"生成 {converted} 个 detect 框, "
            f"丢弃非位姿 {skipped_non_pose}, "
            f"跳过无有效点 {skipped_no_valid}"
        )
        self.finish_signal.emit(
            True, f"四点→检测 JSON 成功，{total_saved} 张，{converted} 框",
        )

    def _kpts_to_det_from_json(
        self,
        split_groups: list,
        indices: list[int],
    ) -> None:
        from core.yolo_to_json import (
            save_labelme_json,
            convert_pose_shapes_to_detect_only,
            write_classes_txt,
            _is_pose_shape,
        )

        if len(split_groups) == 1 and split_groups[0][0] == "all":
            pairs = split_groups[0][1]
            self.log(f"找到 {len(pairs)} 个 JSON（扁平目录，按比例划分）")
            train_p, val_p, test_p = self.split_data(pairs)
            split_groups = [
                ("train", train_p), ("val", val_p), ("test", test_p),
            ]
            split_groups = [(n, p) for n, p in split_groups if p]
        else:
            total = sum(len(p) for _, p in split_groups)
            split_summary = ", ".join(f"{n}={len(p)}" for n, p in split_groups)
            self.log(f"识别划分: {split_summary}，共 {total} 个 JSON")

        total_saved = 0
        converted = 0
        skipped_non_pose = 0
        skipped_no_valid = 0
        class_names: set[str] = set()
        os.makedirs(self.out_dir, exist_ok=True)

        for split_name, pairs in split_groups:
            if not pairs:
                continue
            self.log(f"处理 {split_name} ({len(pairs)} 张)...")
            for img_p, json_p in pairs:
                with open(json_p, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if img_p and os.path.isfile(img_p):
                    img_data = np.fromfile(img_p, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    h, w = (img.shape[:2] if img is not None else (0, 0))
                    base_img = os.path.basename(img_p)
                    shutil.copy(img_p, os.path.join(self.out_dir, base_img))
                    image_path = base_img
                else:
                    h = int(data.get("imageHeight", 0))
                    w = int(data.get("imageWidth", 0))
                    image_path = data.get(
                        "imagePath", os.path.basename(json_p).replace(".json", ".jpg"),
                    )

                shapes = data.get("shapes", [])
                pose_n = sum(1 for s in shapes if _is_pose_shape(s))
                skipped_non_pose += len(shapes) - pose_n
                det_shapes = convert_pose_shapes_to_detect_only(shapes, indices)
                skipped_no_valid += pose_n - len(det_shapes)
                converted += len(det_shapes)
                for s in det_shapes:
                    if s.get("label"):
                        class_names.add(s["label"])

                json_base = os.path.basename(json_p)
                out_json = os.path.join(self.out_dir, json_base)
                save_labelme_json(out_json, image_path, w, h, det_shapes)
                total_saved += 1

        if class_names:
            write_classes_txt(self.out_dir, sorted(class_names))
            self.log(f"已写入 classes.txt → {self.out_dir}")

        self.log("")
        self.log(
            f"转换完成: {total_saved} 张, "
            f"生成 {converted} 个 detect 框, "
            f"丢弃非位姿 {skipped_non_pose}, "
            f"跳过无有效点 {skipped_no_valid}"
        )
        self.finish_signal.emit(
            True, f"四点→检测 JSON 成功，{total_saved} 张，{converted} 框",
        )


# =========================================================
# 比例设置对话框
# =========================================================
class RatioDialog(QDialog):
    def __init__(self, current_ratios, parent=None):
        super().__init__(parent)
        self.setWindowTitle("划分比例")
        self.setFixedSize(420, 320)
        self.setStyleSheet(STYLE_QSS)
        self.ratios = current_ratios

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("headerBar")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(24, 18, 24, 14)
        h_layout.setSpacing(2)

        title = QLabel("划分比例")
        title.setProperty("role", "serif_title")
        h_layout.addWidget(title)

        sub = QLabel("SPLIT RATIO")
        sub.setProperty("role", "serif_subtitle")
        h_layout.addWidget(sub)

        outer.addWidget(header)

        # Body
        body = QVBoxLayout()
        body.setContentsMargins(24, 18, 24, 18)
        body.setSpacing(12)
        outer.addLayout(body, 1)

        self.train_box = self._make_spin_row(body, "训练集  /  Train", int(current_ratios[0] * 100))
        self.val_box   = self._make_spin_row(body, "验证集  /  Val",   int(current_ratios[1] * 100))
        self.test_box  = self._make_spin_row(body, "测试集  /  Test",  int(current_ratios[2] * 100))

        body.addSpacing(4)
        self._ratio_bar = RatioBar(current_ratios, self)
        body.addWidget(self._ratio_bar)

        self.sum_lbl = QLabel("")
        self.sum_lbl.setProperty("role", "hint")
        body.addWidget(self.sum_lbl)

        self.train_box.valueChanged.connect(self._refresh)
        self.val_box.valueChanged.connect(self._refresh)
        self.test_box.valueChanged.connect(self._refresh)
        self._refresh()

        # Footer
        footer = QFrame()
        footer.setObjectName("footerBar")
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(24, 12, 24, 14)
        f_layout.setSpacing(8)
        f_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setProperty("role", "ghost")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        f_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("应用")
        ok_btn.setProperty("role", "primary")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setDefault(True)
        ok_btn.setAutoDefault(True)
        ok_btn.clicked.connect(self.check_and_accept)
        f_layout.addWidget(ok_btn)

        outer.addWidget(footer)

    def _make_spin_row(self, parent_layout, text, val):
        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel(text)
        lbl.setProperty("role", "form_label")
        lbl.setMinimumWidth(140)
        row.addWidget(lbl)

        box = QSpinBox()
        box.setRange(0, 100)
        box.setValue(val)
        box.setSuffix(" %")
        box.setProperty("role", "mono")
        box.setFixedWidth(120)
        row.addWidget(box)
        row.addStretch()

        parent_layout.addLayout(row)
        return box

    def _refresh(self):
        t, v, te = self.train_box.value(), self.val_box.value(), self.test_box.value()
        s = t + v + te
        self._ratio_bar.set_ratios((t/100, v/100, te/100))
        if s == 100:
            self.sum_lbl.setText(f"合计 {s}%   ·   校验通过")
            self.sum_lbl.setProperty("role", "status_success")
        else:
            self.sum_lbl.setText(f"合计 {s}%   ·   需 = 100%")
            self.sum_lbl.setProperty("role", "status_error")

    def check_and_accept(self):
        t, v, te = self.train_box.value(), self.val_box.value(), self.test_box.value()
        if t + v + te != 100:
            QMessageBox.warning(self, "错误", "三个比例之和必须等于 100%！")
            return
        self.ratios = (t / 100.0, v / 100.0, te / 100.0)
        self.accept()


# =========================================================
# 主窗口
# =========================================================
class DatasetToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据集处理  ·  Dataset Processing")
        self.resize(920, 700)
        self.setStyleSheet(STYLE_QSS)

        self.ratios = (0.7, 0.2, 0.1)
        self._load_config()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Header ----
        header = QWidget()
        header.setObjectName("headerBar")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(28, 18, 28, 16)
        h_layout.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title = QLabel("数据集处理")
        title.setProperty("role", "serif_title")
        text_col.addWidget(title)

        sub = QLabel("DATASET  ·  PROCESSING")
        sub.setProperty("role", "serif_subtitle")
        text_col.addWidget(sub)

        h_layout.addLayout(text_col, 1)

        # 右侧: 状态点 + 状态文字
        status_col = QHBoxLayout()
        status_col.setSpacing(8)
        self.status_dot = StatusDot("idle", self)
        status_col.addWidget(self.status_dot, 0, Qt.AlignVCenter)

        self.status_lbl = QLabel("就绪")
        self.status_lbl.setProperty("role", "status_info")
        status_col.addWidget(self.status_lbl, 0, Qt.AlignVCenter)

        h_layout.addLayout(status_col, 0)
        outer.addWidget(header)

        # ---- Body ----
        body = QVBoxLayout()
        body.setContentsMargins(28, 24, 28, 18)
        body.setSpacing(20)
        outer.addLayout(body, 1)

        # ----- 数据源 -----
        src_section = QLabel("数据源  ·  SOURCE")
        src_section.setProperty("role", "section")
        body.addWidget(src_section)

        src_box = QFrame()
        src_box.setProperty("role", "card")
        src_layout = QVBoxLayout(src_box)
        src_layout.setContentsMargins(18, 16, 18, 16)
        src_layout.setSpacing(12)

        self.img_input = self._make_path_row(src_layout, "图片目录", "请选择原图所在文件夹...")
        self.ann_input = self._make_path_row(src_layout, "标签目录", "（可选）留空表示与图片同目录")
        self.out_input = self._make_path_row(src_layout, "输出目录", "请选择处理后数据集的存放目录...")

        body.addWidget(src_box)

        # ----- 任务配置 -----
        cfg_section = QLabel("任务配置  ·  CONFIG")
        cfg_section.setProperty("role", "section")
        body.addWidget(cfg_section)

        cfg_box = QFrame()
        cfg_box.setProperty("role", "card")
        cfg_layout = QVBoxLayout(cfg_box)
        cfg_layout.setContentsMargins(18, 16, 18, 16)
        cfg_layout.setSpacing(14)

        # 模式
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        lbl = QLabel("处理模式")
        lbl.setProperty("role", "form_label")
        lbl.setMinimumWidth(64)
        mode_row.addWidget(lbl)

        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumHeight(34)
        for key, short, sub_desc, accent in MODES:
            self.mode_combo.addItem(short)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        cfg_layout.addLayout(mode_row)

        self.mode_desc_lbl = QLabel("")
        self.mode_desc_lbl.setProperty("role", "caption")
        cfg_layout.addWidget(self.mode_desc_lbl)

        self.force_rm14_chk = QCheckBox("强制 RM 标准类 (BG=0 / RT=0) — 自动映射兵种及能量机关 (RT/BT)")
        self.force_rm14_chk.setChecked(True)
        self.force_rm14_chk.setCursor(Qt.PointingHandCursor)
        self.force_rm14_chk.setToolTip(
            "勾选后自动支持 RM 14 类兵种 (BG…RB) 及 BUFF 能量机关 (RT/BT)；跳过 red_armor 等非标准旧名称。"
        )
        cfg_layout.addWidget(self.force_rm14_chk)

        # 关键点筛选（YOLO→JSON / JSON 筛选模式）
        self.kpt_section = QLabel("关键点筛选  ·  KEYPOINTS")
        self.kpt_section.setProperty("role", "section")
        cfg_layout.addWidget(self.kpt_section)

        kpt_tool_row = QHBoxLayout()
        kpt_tool_row.setSpacing(8)
        self.analyze_btn = QPushButton("分析数据集")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.setProperty("role", "icon")
        self.analyze_btn.clicked.connect(self.analyze_dataset)
        kpt_tool_row.addWidget(self.analyze_btn)

        self.organize_btn = QPushButton("一键识别整理")
        self.organize_btn.setCursor(Qt.PointingHandCursor)
        self.organize_btn.setProperty("role", "primary")
        self.organize_btn.clicked.connect(self.organize_dataset_categories)
        kpt_tool_row.addWidget(self.organize_btn)

        self.purge_rm14_btn = QPushButton("仅保留14类")
        self.purge_rm14_btn.setCursor(Qt.PointingHandCursor)
        self.purge_rm14_btn.setProperty("role", "icon")
        self.purge_rm14_btn.setToolTip(
            "删除 JSON 中非 RM 14 类 (BG/B1..BB/RG/R1..RB) 的全部标注，原地改写"
        )
        self.purge_rm14_btn.clicked.connect(self.purge_non_rm14_labels)
        kpt_tool_row.addWidget(self.purge_rm14_btn)

        self.purge_non4_btn = QPushButton("删除非4点pose")
        self.purge_non4_btn.setCursor(Qt.PointingHandCursor)
        self.purge_non4_btn.setProperty("role", "icon")
        self.purge_non4_btn.setToolTip(
            "删除关键点数 ≠ 4 的 pose/hierarchy_pose 标注（如 5 点 BT/RT），原地改写"
        )
        self.purge_non4_btn.clicked.connect(self.purge_non_4kpt_poses)
        kpt_tool_row.addWidget(self.purge_non4_btn)

        kpt_tool_row.addStretch()
        cfg_layout.addLayout(kpt_tool_row)

        self.kpt_panel = KptFilterPanel(self)
        cfg_layout.addWidget(self.kpt_panel)

        # 比例（YOLO→JSON 模式保留原 train/val/test，不使用随机划分）
        self._ratio_sep = QFrame()
        self._ratio_sep.setProperty("role", "hairline")
        cfg_layout.addWidget(self._ratio_sep)

        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(12)
        self._ratio_lbl = QLabel("划分比例")
        self._ratio_lbl.setProperty("role", "form_label")
        self._ratio_lbl.setMinimumWidth(64)
        ratio_row.addWidget(self._ratio_lbl)

        self.ratio_bar = RatioBar(self.ratios, self)
        self.ratio_bar.setMinimumWidth(360)
        ratio_row.addWidget(self.ratio_bar, 1)

        self.ratio_btn = QPushButton("调整")
        self.ratio_btn.setCursor(Qt.PointingHandCursor)
        self.ratio_btn.setProperty("role", "icon")
        self.ratio_btn.clicked.connect(self.open_ratio_dialog)
        ratio_row.addWidget(self.ratio_btn)
        cfg_layout.addLayout(ratio_row)

        self.ratio_text_lbl = QLabel("")
        self.ratio_text_lbl.setProperty("role", "mono")
        cfg_layout.addWidget(self.ratio_text_lbl)

        body.addWidget(cfg_box)

        # ----- 执行按钮 -----
        exec_row = QHBoxLayout()
        exec_row.setSpacing(8)
        exec_row.addStretch()

        self.start_btn = QPushButton("开始执行转换")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setProperty("role", "primary")
        self.start_btn.setMinimumWidth(180)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_processing)
        exec_row.addWidget(self.start_btn)

        body.addLayout(exec_row)

        # ----- 日志 -----
        log_section = QLabel("处理日志  ·  CONSOLE")
        log_section.setProperty("role", "section")
        body.addWidget(log_section)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setProperty("role", "log")
        self.console.setMinimumHeight(160)
        body.addWidget(self.console, 1)

        self._on_mode_changed(0)

    # ---------------------------------------------------------
    # 控件工厂
    # ---------------------------------------------------------
    def _make_path_row(self, parent_layout, label_text, placeholder):
        row = QHBoxLayout()
        row.setSpacing(12)

        lbl = QLabel(label_text)
        lbl.setProperty("role", "form_label")
        lbl.setMinimumWidth(64)
        row.addWidget(lbl)

        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setProperty("role", "mono")
        row.addWidget(edit, 1)

        btn = QPushButton("选择")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setProperty("role", "icon")
        btn.clicked.connect(lambda: self.select_dir(edit))
        row.addWidget(btn)

        parent_layout.addLayout(row)
        return edit

    # ---------------------------------------------------------
    # 状态切换
    # ---------------------------------------------------------
    def _on_mode_changed(self, idx):
        if idx < 0 or idx >= len(MODES):
            return
        key, _, sub_desc, _ = MODES[idx]
        self.mode_desc_lbl.setText(f"→  {sub_desc}")
        show_kpt = key in ("FROM_YOLO_JSON", "FILTER_JSON_KPTS", "KPTS_TO_DET_JSON")
        show_purge = key in (
            "FROM_YOLO_JSON", "FILTER_JSON_KPTS", "KPTS_TO_DET_JSON", "TO_YOLO_POSE",
        )
        self.kpt_section.setVisible(show_kpt or show_purge)
        self.analyze_btn.setVisible(show_kpt)
        self.organize_btn.setVisible(show_kpt)
        self.purge_rm14_btn.setVisible(show_purge)
        self.purge_non4_btn.setVisible(show_purge)
        self.kpt_panel.setVisible(show_kpt)
        self.force_rm14_chk.setVisible(key == "TO_YOLO_POSE")
        if key == "KPTS_TO_DET_JSON":
            self.kpt_panel.info_lbl.setText(
                "点击「分析数据集」识别类别；请勾选恰好 4 个关键点作为外接框顶点"
            )
        else:
            self.kpt_panel.info_lbl.setText(
                "点击「分析数据集」识别类别；「一键识别整理」同步 pose 配置并删除多余类别"
            )

        use_yolo_layout = key in ("FROM_YOLO_JSON", "KPTS_TO_DET_JSON")
        show_ratio = not use_yolo_layout
        self._ratio_sep.setVisible(show_ratio)
        self._ratio_lbl.setVisible(show_ratio)
        self.ratio_bar.setVisible(show_ratio)
        self.ratio_btn.setVisible(show_ratio)
        self.ratio_text_lbl.setVisible(show_ratio)

        if use_yolo_layout:
            self.img_input.setPlaceholderText(
                "数据集根目录，或 images 目录（含 train/val/test）"
            )
            self.ann_input.setPlaceholderText(
                "labels 目录（可选；选根目录时自动识别 labels/）"
            )
        else:
            self.img_input.setPlaceholderText("请选择原图所在文件夹...")
            self.ann_input.setPlaceholderText("（可选）留空表示与图片同目录")

    def analyze_dataset(self):
        from core.yolo_to_json import (
            collect_yolo_split_pairs,
            normalize_yolo_dataset_dirs,
            resolve_yolo_class_names,
            scan_yolo_dataset,
            iter_txt_paths_from_splits,
            scan_json_pose_dataset,
            load_yaml_kpt_shape,
        )

        img_dir = self.img_input.text().strip()
        ann_dir = self.ann_input.text().strip()
        if not img_dir:
            self.trigger_message("请先选择图片目录", "提示", "warning")
            return

        idx = self.mode_combo.currentIndex()
        mode_key = MODES[idx][0]

        if mode_key == "FROM_YOLO_JSON":
            img_dir, label_dir, dataset_root = normalize_yolo_dataset_dirs(img_dir, ann_dir)
            split_groups = collect_yolo_split_pairs(img_dir, label_dir)
            if not split_groups:
                self.trigger_message(
                    "未找到 YOLO 配对，请确认目录含 images/ 与 labels/",
                    "分析失败", "warning",
                )
                return
            txt_paths = iter_txt_paths_from_splits(split_groups)
            class_names, name_source, meta_file = resolve_yolo_class_names(
                img_dir, label_dir, txt_paths,
            )
            if not class_names:
                self.trigger_message("标签中未找到有效 class_id", "分析失败", "warning")
                return
            scan = scan_yolo_dataset(txt_paths, class_names)
            scan["splits"] = {n: len(p) for n, p in split_groups}
            scan["class_source"] = name_source
            scan["dataset_root"] = dataset_root
            scan["meta_file"] = meta_file
            self.append_log(f"数据集根目录: {dataset_root}")
            if meta_file:
                self.append_log(f"类别配置: {meta_file}")
            class_stats = scan.get("class_stats", {})
            from core.pose_config import PoseConfig
            preview_cfg = PoseConfig(os.path.join(_ROOT, "config", "pose_classes.json"))
            preview_cfg.load()
            yaml_kpt = load_yaml_kpt_shape(meta_file)
            new_pose = [
                n for n, st in class_stats.items()
                if st["pose_count"] > 0 and not preview_cfg.has_category(n)
            ]
            if new_pose:
                self.append_log(f"将自动定义位姿类别: {new_pose}")
            for name, st in class_stats.items():
                if st["pose_count"] > 0:
                    kpt = st["max_kpt"] or yaml_kpt or 0
                    self.append_log(
                        f"  [{name}] pose×{st['pose_count']}, 关键点={kpt}"
                    )
        elif mode_key == "FILTER_JSON_KPTS":
            ann_dir = ann_dir or img_dir
            from core.yolo_to_json import collect_json_paths
            json_paths = collect_json_paths(img_dir, ann_dir)
            if not json_paths:
                self.trigger_message("未找到 JSON 文件", "分析失败", "warning")
                return
            scan = scan_json_pose_dataset(json_paths)
            kpt_default_n = None
        elif mode_key == "KPTS_TO_DET_JSON":
            img_dir_n, label_dir, dataset_root = normalize_yolo_dataset_dirs(img_dir, ann_dir)
            split_groups = collect_yolo_split_pairs(img_dir_n, label_dir)
            kpt_default_n = 4
            if split_groups:
                txt_paths = iter_txt_paths_from_splits(split_groups)
                class_names, name_source, meta_file = resolve_yolo_class_names(
                    img_dir_n, label_dir, txt_paths,
                )
                if not class_names:
                    self.trigger_message("标签中未找到有效 class_id", "分析失败", "warning")
                    return
                scan = scan_yolo_dataset(txt_paths, class_names)
                scan["splits"] = {n: len(p) for n, p in split_groups}
                scan["class_source"] = name_source
                self.append_log(f"输入类型: YOLO-Pose · 根目录 {dataset_root}")
            else:
                ann_dir_e = ann_dir or img_dir
                from core.yolo_to_json import collect_json_paths
                json_paths = collect_json_paths(img_dir, ann_dir_e)
                if not json_paths:
                    self.trigger_message(
                        "未找到 YOLO 或 JSON 标注，请确认目录结构",
                        "分析失败", "warning",
                    )
                    return
                scan = scan_json_pose_dataset(json_paths)
                self.append_log("输入类型: 位姿 JSON")
        else:
            return

        if mode_key == "FROM_YOLO_JSON":
            kpt_default_n = None
        elif mode_key not in ("FILTER_JSON_KPTS", "KPTS_TO_DET_JSON"):
            return

        self.kpt_panel.apply_scan_result(scan, default_first_n=kpt_default_n)
        self.append_log("─" * 60)
        split_info = scan.get("splits")
        if split_info:
            self.append_log(
                "识别划分: " + ", ".join(f"{k}={v}" for k, v in split_info.items())
            )
        self.append_log(f"分析完成 · 任务={scan.get('task')} · 最大关键点={scan.get('max_kpt', 0)}")
        for c in scan.get("used_classes", []):
            self.append_log(f"  类别 {c['name']}: {c['count']} 实例")

    def _resolve_json_paths_for_purge(self) -> list[str] | None:
        """从当前图片/标注目录收集 JSON，供清理按钮使用。"""
        from core.yolo_to_json import collect_json_paths

        img_dir = self.img_input.text().strip()
        ann_dir = self.ann_input.text().strip()
        if not img_dir and not ann_dir:
            self.trigger_message("请先选择图片或标注目录", "提示", "warning")
            return None
        base_img = img_dir or ann_dir
        base_ann = ann_dir or img_dir
        paths = collect_json_paths(base_img, base_ann)
        if not paths:
            self.trigger_message("未找到 JSON 文件", "提示", "warning")
            return None
        return paths

    def purge_non_rm14_labels(self):
        """一键删除非 RM 14 类标注（原地改写 JSON）。"""
        from core.constants import HIER_CLASS_NAMES
        from core.yolo_to_json import purge_json_annotations

        paths = self._resolve_json_paths_for_purge()
        if paths is None:
            return
        keep = set(HIER_CLASS_NAMES)
        reply = QMessageBox.question(
            self, "确认清理",
            f"将在 {len(paths)} 个 JSON 中删除非 14 类标注：\n"
            f"{', '.join(HIER_CLASS_NAMES)}\n\n"
            "此操作原地改写，不可撤销。继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        stats = purge_json_annotations(paths, keep_labels=keep)
        self.append_log("─" * 60)
        self.append_log("仅保留14类 · 清理完成")
        self.append_log(
            f"  改写文件 {stats['files_touched']} / {len(paths)}，"
            f"标注 {stats['shapes_before']} → {stats['shapes_after']}，"
            f"按类名删除 {stats['removed_by_label']}"
        )
        if stats["removed_labels"]:
            top = sorted(stats["removed_labels"].items(), key=lambda x: -x[1])[:12]
            self.append_log("  被删类别: " + ", ".join(f"{n}×{c}" for n, c in top))
        self.trigger_message(
            f"已删除 {stats['removed_by_label']} 个非14类标注，"
            f"改写 {stats['files_touched']} 个文件",
            "清理完成", "info",
        )

    def purge_non_4kpt_poses(self):
        """一键删除关键点数 ≠ 4 的 pose 标注（原地改写 JSON）。"""
        from core.yolo_to_json import purge_json_annotations

        paths = self._resolve_json_paths_for_purge()
        if paths is None:
            return
        reply = QMessageBox.question(
            self, "确认清理",
            f"将在 {len(paths)} 个 JSON 中删除关键点数 ≠ 4 的 pose 标注\n"
            "（如 5 点 BT/RT；矩形等非 pose 不受影响）。\n\n"
            "此操作原地改写，不可撤销。继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        stats = purge_json_annotations(paths, keep_pose_kpt_count=4)
        self.append_log("─" * 60)
        self.append_log("删除非4点pose · 清理完成")
        self.append_log(
            f"  改写文件 {stats['files_touched']} / {len(paths)}，"
            f"标注 {stats['shapes_before']} → {stats['shapes_after']}，"
            f"按关键点数删除 {stats['removed_by_kpt']}"
        )
        self.trigger_message(
            f"已删除 {stats['removed_by_kpt']} 个非4点 pose，"
            f"改写 {stats['files_touched']} 个文件",
            "清理完成", "info",
        )

    def organize_dataset_categories(self):
        from core.yolo_to_json import (
            collect_yolo_split_pairs,
            normalize_yolo_dataset_dirs,
            resolve_yolo_class_names,
            scan_yolo_dataset,
            iter_txt_paths_from_splits,
            scan_json_pose_dataset,
            collect_json_paths,
            load_yaml_kpt_shape,
            organize_dataset_categories as organize_cats,
        )
        from core.pose_config import PoseConfig

        img_dir = self.img_input.text().strip()
        ann_dir = self.ann_input.text().strip()
        out_dir = self.out_input.text().strip()
        if not img_dir:
            self.trigger_message("请先选择图片目录", "提示", "warning")
            return

        idx = self.mode_combo.currentIndex()
        mode_key = MODES[idx][0]
        if mode_key not in ("FROM_YOLO_JSON", "FILTER_JSON_KPTS"):
            self.trigger_message("当前模式不支持类别整理", "提示", "warning")
            return

        global_pose_path = os.path.join(_ROOT, "config", "pose_classes.json")
        pose_config = PoseConfig(global_pose_path)
        pose_config.load()

        target_dirs: set[str] = set()
        meta_file = None
        yaml_kpt = None
        scan = None

        if mode_key == "FROM_YOLO_JSON":
            img_dir_n, label_dir, dataset_root = normalize_yolo_dataset_dirs(img_dir, ann_dir)
            split_groups = collect_yolo_split_pairs(img_dir_n, label_dir)
            if not split_groups:
                self.trigger_message(
                    "未找到 YOLO 配对，请确认目录含 images/ 与 labels/",
                    "整理失败", "warning",
                )
                return
            txt_paths = iter_txt_paths_from_splits(split_groups)
            class_names, name_source, meta_file = resolve_yolo_class_names(
                img_dir_n, label_dir, txt_paths,
            )
            if not class_names:
                self.trigger_message("标签中未找到有效 class_id", "整理失败", "warning")
                return
            scan = scan_yolo_dataset(txt_paths, class_names)
            scan["splits"] = {n: len(p) for n, p in split_groups}
            scan["class_source"] = name_source
            class_stats = scan["class_stats"]
            yaml_kpt = load_yaml_kpt_shape(meta_file)
            target_dirs.add(dataset_root)
            target_dirs.add(label_dir)
            for split_name, _ in split_groups:
                if split_name != "all":
                    target_dirs.add(os.path.join(label_dir, split_name))
                    target_dirs.add(os.path.join(img_dir_n, split_name))
        else:
            ann_dir_e = ann_dir or img_dir
            json_paths = collect_json_paths(img_dir, ann_dir_e)
            if not json_paths:
                self.trigger_message("未找到 JSON 文件", "整理失败", "warning")
                return
            scan = scan_json_pose_dataset(json_paths)
            class_stats = scan["class_stats"]
            class_names = scan.get("class_names", [])
            target_dirs.add(os.path.abspath(ann_dir_e))
            target_dirs.add(os.path.abspath(img_dir))
            for split in ("train", "val", "test"):
                sd = os.path.join(ann_dir_e, split)
                if os.path.isdir(sd):
                    target_dirs.add(sd)
                    img_sd = os.path.join(img_dir, split)
                    if os.path.isdir(img_sd):
                        target_dirs.add(img_sd)

        if out_dir:
            target_dirs.add(out_dir)
            for split in ("train", "val", "test"):
                sd = os.path.join(out_dir, split)
                if os.path.isdir(sd):
                    target_dirs.add(sd)

        result = organize_cats(
            pose_config, class_stats, class_names,
            list(target_dirs),
            yaml_kpt_shape=yaml_kpt,
            save_global_path=global_pose_path,
        )

        self.kpt_panel.apply_scan_result(scan)
        self.append_log("─" * 60)
        self.append_log("一键识别整理完成")
        if result["added"]:
            self.append_log(f"  新增位姿类别: {', '.join(result['added'])}")
        if result["removed"]:
            self.append_log(f"  删除多余类别: {', '.join(result['removed'])}")
        if result["updated"]:
            self.append_log(f"  更新关键点数: {', '.join(result['updated'])}")
        if not result["added"] and not result["removed"] and not result["updated"]:
            self.append_log("  类别已与数据集一致，无需变更")
        self.append_log(
            f"  保留类别 ({len(result['class_names'])}): {result['class_names']}"
        )
        self.append_log(f"  全局位姿配置 → {global_pose_path}")
        for d in sorted(target_dirs):
            self.append_log(f"  已写入 classes.txt / pose_classes.json → {d}")

        self.trigger_message(
            f"整理完成：新增 {len(result['added'])}，"
            f"删除 {len(result['removed'])}，"
            f"更新 {len(result['updated'])}",
            "整理完成", "info",
        )

    def _refresh_ratio_text(self):
        t, v, te = [int(r * 100) for r in self.ratios]
        if te > 0:
            txt = f"Train {t}%   ·   Val {v}%   ·   Test {te}%"
        else:
            txt = f"Train {t}%   ·   Val {v}%   ·   Test 0  (关闭)"
        self.ratio_text_lbl.setText(txt)
        self.ratio_bar.set_ratios(self.ratios)

    # ---------------------------------------------------------
    # 配置持久化
    # ---------------------------------------------------------
    def _load_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig().get_tool_config("dataset_tool")
        self._cfg_img = cfg.get("image_dir", "")
        self._cfg_ann = cfg.get("ann_dir", "")
        self._cfg_out = cfg.get("output_dir", "")
        idx = int(cfg.get("mode_index", 0))
        self._cfg_mode_idx = max(0, min(idx, len(MODES) - 1))
        t = cfg.get("train_ratio", 70)
        v = cfg.get("val_ratio", 20)
        te = cfg.get("test_ratio", 10)
        self.ratios = (t / 100.0, v / 100.0, te / 100.0)

    def _apply_config_to_ui(self):
        self.img_input.setText(self._cfg_img)
        self.ann_input.setText(self._cfg_ann)
        self.out_input.setText(self._cfg_out)
        self.mode_combo.setCurrentIndex(self._cfg_mode_idx)
        self._refresh_ratio_text()

    def _save_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig()
        cfg.set("dataset_tool", "image_dir", self.img_input.text())
        cfg.set("dataset_tool", "ann_dir", self.ann_input.text())
        cfg.set("dataset_tool", "output_dir", self.out_input.text())
        cfg.set("dataset_tool", "mode_index", self.mode_combo.currentIndex())
        t = int(self.ratios[0] * 100)
        v = int(self.ratios[1] * 100)
        te = int(self.ratios[2] * 100)
        cfg.set("dataset_tool", "train_ratio", t)
        cfg.set("dataset_tool", "val_ratio", v)
        cfg.set("dataset_tool", "test_ratio", te)
        cfg.save()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_cfg_applied", False):
            self._apply_config_to_ui()
            self._cfg_applied = True

    def closeEvent(self, event):
        self._save_config()
        event.accept()

    # ---------------------------------------------------------
    def _get_default_dir(self, category="origin_data"):
        base = _ROOT
        if category == "origin_data":
            d = os.path.join(base, "origin_data")
            return d if os.path.exists(d) else base
        elif category == "yolo":
            d = os.path.join(base, "runs")
            return d if os.path.exists(d) else base
        elif category == "mlp":
            d = os.path.join(base, "stored_weights", "model", "armor_tradition")
            return d if os.path.exists(d) else base
        return base

    def select_dir(self, line_edit, category="origin_data"):
        default_dir = self._get_default_dir(category)
        d = QFileDialog.getExistingDirectory(self, "选择目录", default_dir)
        if d:
            line_edit.setText(d)

    def open_ratio_dialog(self):
        dlg = RatioDialog(self.ratios, self)
        if dlg.exec():
            self.ratios = dlg.ratios
            self._refresh_ratio_text()

    # ---------------------------------------------------------
    def append_log(self, text):
        self.console.append(text)
        self.console.moveCursor(QTextCursor.End)

    def trigger_message(self, text, title, status):
        try:
            DialogOver(self, text, title, status)
        except Exception:
            QMessageBox.information(self, title, text)

    def start_processing(self):
        img_dir = self.img_input.text().strip()
        out_dir = self.out_input.text().strip()
        ann_dir = self.ann_input.text().strip()

        if not img_dir or not out_dir:
            self.trigger_message("请先选择源图片目录和输出目录！", "表单未完成", "warning")
            return

        idx = self.mode_combo.currentIndex()
        mode_key = MODES[idx][0]

        if mode_key == "FROM_YOLO_JSON":
            from core.yolo_to_json import normalize_yolo_dataset_dirs
            img_dir, label_dir, _ = normalize_yolo_dataset_dirs(img_dir, ann_dir)
            if not os.path.isdir(label_dir):
                self.trigger_message(
                    "未找到 labels 目录，请选择数据集根目录或手动指定 labels/",
                    "表单未完成", "warning",
                )
                return
            ann_dir = label_dir
        elif mode_key == "KPTS_TO_DET_JSON":
            from core.yolo_to_json import normalize_yolo_dataset_dirs
            img_dir, label_dir, _ = normalize_yolo_dataset_dirs(img_dir, ann_dir)
            if os.path.isdir(label_dir):
                ann_dir = label_dir

        keep_kpt_indices = None
        if mode_key in ("FROM_YOLO_JSON", "FILTER_JSON_KPTS", "KPTS_TO_DET_JSON"):
            if self.kpt_panel._checkboxes:
                keep_kpt_indices = self.kpt_panel.get_keep_indices()
                if mode_key == "KPTS_TO_DET_JSON":
                    if len(keep_kpt_indices) != 4:
                        self.trigger_message(
                            "请勾选恰好 4 个关键点序号", "提示", "warning",
                        )
                        return
                elif not keep_kpt_indices:
                    self.trigger_message("请至少选择一个关键点序号", "提示", "warning")
                    return
            elif mode_key in ("FILTER_JSON_KPTS", "KPTS_TO_DET_JSON"):
                self.trigger_message("请先点击「分析数据集」", "提示", "warning")
                return

        self.console.clear()
        self.start_btn.setEnabled(False)
        self.start_btn.setText("处理中…")
        self.status_dot.set_state("working")
        self.status_lbl.setText("处理中")
        self.status_lbl.setProperty("role", "status_warn")

        self.worker = DatasetWorker(
            mode_key, img_dir, ann_dir, out_dir, self.ratios,
            keep_kpt_indices=keep_kpt_indices,
            force_rm_14_classes=(
                mode_key == "TO_YOLO_POSE" and self.force_rm14_chk.isChecked()
            ),
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.finish_signal.connect(self.on_process_finish)
        self.worker.start()

    def on_process_finish(self, success, msg):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("开始执行转换")
        if success:
            self.status_dot.set_state("done")
            self.status_lbl.setText("完成")
            self.status_lbl.setProperty("role", "status_success")
            self.trigger_message(msg, "处理完成", "success")
        else:
            self.status_dot.set_state("error")
            self.status_lbl.setText("失败")
            self.status_lbl.setProperty("role", "status_error")
            self.trigger_message(msg, "处理异常", "danger")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetToolWindow()
    window.show()
    sys.exit(app.exec())
