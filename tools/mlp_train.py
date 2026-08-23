# -*- coding: utf-8 -*-
"""
MLP装甲板分类器训练工具
支持数据增强、训练进度监控、ONNX导出
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import time
import random
import numpy as np
import cv2
from collections import OrderedDict

from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QLabel, QPushButton, QLineEdit, QSpinBox,
                               QDoubleSpinBox, QCheckBox, QComboBox, QTextEdit,
                               QProgressBar, QFileDialog, QMessageBox, QTabWidget,
                               QFormLayout, QScrollArea, QSizePolicy, QSplitter, QFrame)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================================================
# 加载编辑风样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()


# ============================================================================
# 数据集文件夹 → 标签映射
# ============================================================================
def discover_dataset_classes(root_dir):
    """
    从数据集目录动态发现类别映射。
    优先读取 label.txt（格式: "名字:idx_X"），未找到则扫描子文件夹。
    返回 (folder_to_label, label_to_index, index_to_label, num_classes)
    """
    folder_to_label = {}
    label_to_folder = {}

    label_txt_path = os.path.join(root_dir, 'label.txt')
    if os.path.exists(label_txt_path):
        with open(label_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                name, folder = line.split(':', 1)
                name = name.strip()
                folder = folder.strip()
                if name and folder:
                    folder_to_label[folder] = name
                    label_to_folder[name] = folder

    if folder_to_label:
        for entry in os.scandir(root_dir):
            if entry.is_dir() and entry.name not in folder_to_label:
                folder_to_label[entry.name] = entry.name
                label_to_folder[entry.name] = entry.name
    else:
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                folder_to_label[entry.name] = entry.name
                label_to_folder[entry.name] = entry.name

    sorted_labels = sorted(label_to_folder.keys())
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    num_classes = len(label_to_index)

    return folder_to_label, label_to_index, index_to_label, num_classes

# ============================================================================
# 数据增强
# ============================================================================
class RandomAugmentation:
    def __init__(self, h_flip=0.5, rotation=5, brightness=0.2, contrast=0.2,
                 affine_shift=3, affine_scale=(0.9, 1.1)):
        self.h_flip = h_flip
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.affine_shift = affine_shift
        self.affine_scale = affine_scale

    def __call__(self, img):
        # img: numpy array HxWx3 BGR
        if random.random() < self.h_flip:
            img = cv2.flip(img, 1)

        if self.rotation > 0:
            angle = random.uniform(-self.rotation, self.rotation)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if self.brightness > 0 or self.contrast > 0:
            brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = np.clip(img * contrast_factor + brightness_factor * 20 - 10, 0, 255).astype(np.uint8)

        if self.affine_shift > 0:
            h, w = img.shape[:2]
            dx = random.uniform(-self.affine_shift, self.affine_shift)
            dy = random.uniform(-self.affine_shift, self.affine_shift)
            scale = random.uniform(*self.affine_scale)
            M = np.float32([[scale, 0, dx], [0, scale, dy]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return img


# ============================================================================
# 数据集
# ============================================================================
class ArmorDataset(Dataset):
    def __init__(self, root_dir, input_size=(20, 28), input_channels=1, augment=False,
                 folder_to_label=None, label_to_index=None):
        self.root_dir = root_dir
        self.input_size = input_size
        self.input_channels = input_channels
        self.augment = augment
        self.samples = []
        self.augmenter = RandomAugmentation() if augment else None
        self.folder_to_label = folder_to_label or {}
        self.label_to_index = label_to_index or {}

        self._load_samples()

    def _load_samples(self):
        for folder, label in self.folder_to_label.items():
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label_idx = self.label_to_index.get(label, -1)
            if label_idx < 0:
                continue
            for fname in os.listdir(folder_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                fpath = os.path.join(folder_path, fname)
                self.samples.append((fpath, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]

        if self.input_channels == 1:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.input_size[1], self.input_size[0]), dtype=np.uint8)
            else:
                if self.augmenter is not None:
                    img = self.augmenter(img)
                img = cv2.resize(img, (self.input_size[0], self.input_size[1]))
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img = img.astype(np.float32) / 255.0
            img = (img - 0.0) / 1.0
            img = torch.from_numpy(img).float().unsqueeze(0)
        else:
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            else:
                if self.augmenter is not None:
                    img = self.augmenter(img)
                img = cv2.resize(img, (self.input_size[0], self.input_size[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, torch.tensor(label, dtype=torch.long)


# ============================================================================
# CNN 模型
# ============================================================================
class NumberClassifier(nn.Module):
    def __init__(self, num_classes=8, input_H=28, input_W=20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_H, input_W)
            x = self._forward_features(dummy)
            flat_dim = x.view(-1).shape[0]

        self.fc1 = nn.Linear(flat_dim, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_mobilenet_model(num_classes):
    from torchvision.models import mobilenet_v3_small
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ============================================================================
# 训练线程
# ============================================================================
class MLPTrainWorker(QThread):
    log_signal = Signal(str)
    batch_signal = Signal(int, int, float)        # batch_idx, total_batches, loss
    progress_signal = Signal(int, int, float, float, float, float)  # epoch, total, train_acc, val_acc, train_loss, val_loss
    finished_signal = Signal(bool, str)

    def __init__(self, model_save_path, dataset_dir, epochs, batch_size, lr,
                 weight_decay, train_ratio, augment, device, num_workers=4,
                 architecture='number_classifier', input_w=20, input_h=28,
                 folder_to_label=None, label_to_index=None, index_to_label=None):
        super().__init__()
        self.model_save_path = model_save_path
        self.dataset_dir = dataset_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_ratio = train_ratio
        self.augment = augment
        self.device = device
        self.num_workers = num_workers
        self.architecture = architecture
        self.input_w = input_w
        self.input_h = input_h
        self.folder_to_label = folder_to_label or {}
        self.label_to_index = label_to_index or {}
        self.index_to_label = index_to_label or {}
        self.num_classes = len(self.index_to_label)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.log_signal.emit("=" * 50)
            self.log_signal.emit("MLP 分类器训练开始")
            self.log_signal.emit(f"数据集: {self.dataset_dir}")
            self.log_signal.emit(f"架构: {self.architecture}, 输入尺寸: {self.input_w}x{self.input_h}")
            self.log_signal.emit(f"Epochs: {self.epochs}, Batch: {self.batch_size}, LR: {self.lr}")
            self.log_signal.emit(f"设备: {self.device}")
            self.log_signal.emit(f"数据增强: {'是' if self.augment else '否'}")
            self.log_signal.emit("=" * 50)

            input_channels = 1 if self.architecture == 'number_classifier' else 3
            full_dataset = ArmorDataset(
                self.dataset_dir,
                input_size=(self.input_w, self.input_h),
                input_channels=input_channels,
                augment=self.augment,
                folder_to_label=self.folder_to_label,
                label_to_index=self.label_to_index,
            )

            if len(full_dataset) == 0:
                self.finished_signal.emit(False, "数据集为空，请检查路径")
                return

            self.log_signal.emit(f"数据集样本数: {len(full_dataset)}")

            total = len(full_dataset)
            train_size = int(total * self.train_ratio)
            val_size = total - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers, pin_memory=True
            )

            self.log_signal.emit(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

            if self.architecture == 'number_classifier':
                model = NumberClassifier(num_classes=self.num_classes, input_H=self.input_h, input_W=self.input_w).to(self.device)
            else:
                model = create_mobilenet_model(num_classes=self.num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            best_val_acc = 0.0
            best_model_state = None

            for epoch in range(self.epochs):
                if self._stop:
                    self.log_signal.emit("训练已停止")
                    break

                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                    self.batch_signal.emit(batch_idx + 1, len(train_loader), loss.item())

                    if batch_idx % 10 == 0:
                        self.log_signal.emit(
                            f"Epoch [{epoch+1}/{self.epochs}] "
                            f"Batch [{batch_idx}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f}"
                        )

                train_acc = 100.0 * train_correct / train_total
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                val_acc = 100.0 * val_correct / val_total
                val_loss /= len(val_loader)
                scheduler.step()

                self.log_signal.emit(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
                )
                self.progress_signal.emit(epoch + 1, self.epochs, train_acc, val_acc, train_loss, val_loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    self.log_signal.emit(f"*** 新的最佳验证准确率: {best_val_acc:.2f}% ***")

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            self.log_signal.emit(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")

            os.makedirs(os.path.dirname(self.model_save_path) or '.', exist_ok=True)

            pt_path = self.model_save_path
            torch.save(model.state_dict(), pt_path)
            self.log_signal.emit(f"PyTorch 模型已保存: {pt_path}")

            self._export_onnx(model, pt_path.replace('.pt', '.onnx'))
            self._save_labels(pt_path.replace('.pt', '_labels.txt'))
            label_txt_path = os.path.join(os.path.dirname(pt_path), 'label.txt')
            self._save_labels(label_txt_path)
            preprocess_path = os.path.join(os.path.dirname(pt_path), 'preprocess.json')
            self._save_preprocess(preprocess_path)

            self.finished_signal.emit(True, f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")

        except Exception as e:
            import traceback
            self.log_signal.emit(f"训练出错: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, f"训练出错: {str(e)}")

    def _export_onnx(self, model, onnx_path):
        try:
            input_channels = 1 if self.architecture == 'number_classifier' else 3
            dummy = torch.zeros(1, input_channels, self.input_h, self.input_w).to(self.device)
            model.eval()
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'],
                output_names=['output'],
                opset_version=18
            )
            self.log_signal.emit(f"ONNX 模型已导出: {onnx_path}")
        except Exception as e:
            self.log_signal.emit(f"ONNX 导出失败: {e}")

    def _save_labels(self, path):
        labels = [self.index_to_label[i] for i in range(self.num_classes)]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels))
        self.log_signal.emit(f"标签已保存: {path}")

    def _save_preprocess(self, path):
        if self.architecture == 'number_classifier':
            cfg = {
                'mean': [0.0],
                'std': [1.0],
                'input_size': [self.input_w, self.input_h],
                'channels': 1,
                'architecture': 'number_classifier'
            }
        else:
            cfg = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'input_size': [self.input_w, self.input_h],
                'channels': 3,
                'architecture': 'mobilenet_v3_small'
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4)
        self.log_signal.emit(f"预处理配置已保存: {path}")


# ============================================================================
# 训练窗口
# ============================================================================
class MLPTrainWindow(QMainWindow):
    def __init__(self, architecture='number_classifier'):
        super().__init__()
        self.architecture = architecture
        if architecture == 'number_classifier':
            self.setWindowTitle("MLP (Conv) 分类器训练")
            self._default_w, self._default_h = 20, 28
            self._channels_info = "灰度二值化 (1通道)"
            self._channels_role = "status_warn"
        else:
            self.setWindowTitle("MobileNetV3 分类器训练")
            self._default_w, self._default_h = 56, 56
            self._channels_info = "彩色 (3通道, ImageNet归一化)"
            self._channels_role = "status_success"
        self.setMinimumSize(700, 600)
        self.setStyleSheet(EDITORIAL_QSS)
        self._train_worker = None
        self._init_ui()

    def _init_ui(self):
        font_bold = QFont("Microsoft YaHei", 9, QFont.Bold)
        font_normal = QFont("Microsoft YaHei", 9)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title_text = "MLP (Conv) 装甲板分类器训练" if self.architecture == 'number_classifier' else "MobileNetV3 装甲板分类器训练"
        title = QLabel(title_text)
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ==================== 顶部状态卡片（YOLO 风格） ====================
        status_widget = QFrame()
        status_widget.setProperty("role", "card")
        status_layout = QGridLayout(status_widget)
        status_layout.setSpacing(12)

        self.status_label = QLabel("等待开始训练")
        self.status_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.status_label.setProperty("role", "muted")

        self.epoch_label = QLabel("Epoch: -/-")
        self.epoch_label.setFont(font_bold)
        self.epoch_label.setProperty("role", "accent")

        self.loss_label = QLabel("Loss: -")
        self.loss_label.setFont(font_normal)
        self.loss_label.setProperty("role", "muted")

        self.acc_label = QLabel("Train: -  Val: -")
        self.acc_label.setFont(font_normal)
        self.acc_label.setProperty("role", "accent")

        self.epoch_progress_bar = QProgressBar()
        self.epoch_progress_bar.setRange(0, 100)
        self.epoch_progress_bar.setValue(0)
        self.epoch_progress_bar.setTextVisible(True)

        status_layout.addWidget(self.status_label, 0, 0, 1, 2)
        status_layout.addWidget(self.epoch_label, 1, 0)
        status_layout.addWidget(self.loss_label, 1, 1)
        status_layout.addWidget(self.acc_label, 1, 2)
        status_layout.addWidget(self.epoch_progress_bar, 2, 0, 1, 3)

        layout.addWidget(status_widget)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._create_dataset_tab(font_bold, font_normal), "数据集")
        tabs.addTab(self._create_train_tab(font_bold, font_normal), "训练参数")
        tabs.addTab(self._create_aug_tab(font_bold, font_normal), "数据增强")

        self.log_edit = QTextEdit()
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        layout.addWidget(QLabel("训练日志:"))
        layout.addWidget(self.log_edit)

        batch_progress_layout = QHBoxLayout()
        batch_progress_layout.addWidget(QLabel("Batch:"))
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setTextVisible(True)
        self.lbl_batch = QLabel("-/-")
        batch_progress_layout.addWidget(self.batch_progress_bar, 1)
        batch_progress_layout.addWidget(self.lbl_batch)
        layout.addLayout(batch_progress_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setFont(font_bold)
        self.btn_start.setProperty("role", "primary")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start_train)
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setFont(font_bold)
        self.btn_stop.setProperty("role", "danger")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_train)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _create_dataset_tab(self, font_bold, font_normal):
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("数据集设置")
        f = QFormLayout(g)

        self.edt_dataset = QLineEdit()
        self.edt_dataset.setFont(font_normal)
        self.edt_dataset.setPlaceholderText("选择 Armor Transform Dataset 目录...")
        btn_browse = QPushButton("浏览")
        btn_browse.setFont(font_normal)
        btn_browse.clicked.connect(self._on_select_dataset)
        h = QHBoxLayout()
        h.addWidget(self.edt_dataset)
        h.addWidget(btn_browse)
        f.addRow("数据集路径:", h)

        self.lbl_dataset_info = QLabel("未选择数据集")
        self.lbl_dataset_info.setFont(font_normal)
        f.addRow("数据集信息:", self.lbl_dataset_info)

        v.addWidget(g)

        g2 = QGroupBox("模型输出设置")
        f2 = QFormLayout(g2)

        self.edt_output = QLineEdit()
        self.edt_output.setFont(font_normal)
        self.edt_output.setText("./output_mlp")
        btn_out = QPushButton("浏览")
        btn_out.setFont(font_normal)
        btn_out.clicked.connect(self._on_select_output)
        h2 = QHBoxLayout()
        h2.addWidget(self.edt_output)
        h2.addWidget(btn_out)
        f2.addRow("输出目录:", h2)

        self.edt_model_name = QLineEdit()
        self.edt_model_name.setFont(font_normal)
        self.edt_model_name.setText("number_classifier")
        f2.addRow("模型名称:", self.edt_model_name)

        v.addWidget(g2)

        g3 = QGroupBox("模型架构设置")
        f3 = QFormLayout(g3)

        arch_name = "NumberClassifier (Conv MLP)" if self.architecture == 'number_classifier' else "MobileNetV3-Small"
        lbl_arch = QLabel(arch_name)
        lbl_arch.setFont(font_bold)
        f3.addRow("架构:", lbl_arch)

        size_layout = QHBoxLayout()
        lbl_w = QLabel("宽:")
        lbl_w.setFont(font_normal)
        self.spin_input_w = QSpinBox()
        self.spin_input_w.setFont(font_normal)
        self.spin_input_w.setRange(8, 512)
        self.spin_input_w.setValue(self._default_w)
        lbl_h = QLabel("高:")
        lbl_h.setFont(font_normal)
        self.spin_input_h = QSpinBox()
        self.spin_input_h.setFont(font_normal)
        self.spin_input_h.setRange(8, 512)
        self.spin_input_h.setValue(self._default_h)
        size_layout.addWidget(lbl_w)
        size_layout.addWidget(self.spin_input_w)
        size_layout.addWidget(lbl_h)
        size_layout.addWidget(self.spin_input_h)
        size_layout.addStretch()
        f3.addRow("输入尺寸 (W×H):", size_layout)

        self.lbl_channels_info = QLabel(self._channels_info)
        self.lbl_channels_info.setFont(font_normal)
        self.lbl_channels_info.setProperty("role", self._channels_role)
        f3.addRow("输入类型:", self.lbl_channels_info)

        v.addWidget(g3)
        v.addStretch()
        return w

    def _create_train_tab(self, font_bold, font_normal):
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("训练参数")
        f = QFormLayout(g)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setFont(font_normal)
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(50)
        f.addRow("Epochs:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setFont(font_normal)
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(32)
        f.addRow("Batch Size:", self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setFont(font_normal)
        self.spin_lr.setRange(1e-6, 1e-1)
        self.spin_lr.setDecimals(6)
        self.spin_lr.setSingleStep(1e-4)
        self.spin_lr.setValue(1e-3)
        f.addRow("学习率 (LR):", self.spin_lr)

        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setFont(font_normal)
        self.spin_wd.setRange(0.0, 0.1)
        self.spin_wd.setDecimals(5)
        self.spin_wd.setSingleStep(1e-4)
        self.spin_wd.setValue(1e-4)
        f.addRow("Weight Decay:", self.spin_wd)

        self.spin_train_ratio = QDoubleSpinBox()
        self.spin_train_ratio.setFont(font_normal)
        self.spin_train_ratio.setRange(0.5, 0.95)
        self.spin_train_ratio.setDecimals(2)
        self.spin_train_ratio.setSingleStep(0.05)
        self.spin_train_ratio.setValue(0.8)
        f.addRow("训练集比例:", self.spin_train_ratio)

        self.cmb_device = QComboBox()
        self.cmb_device.setFont(font_normal)
        self.cmb_device.addItems(['cuda', 'cpu'])
        if torch.cuda.is_available():
            self.cmb_device.setCurrentText('cuda')
        else:
            self.cmb_device.setCurrentText('cpu')
        f.addRow("设备:", self.cmb_device)

        v.addWidget(g)
        v.addStretch()
        return w

    def _create_aug_tab(self, font_bold, font_normal):
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("数据增强（推理时启用可提升泛化能力）")
        f = QFormLayout(g)

        self.chk_aug = QCheckBox("启用数据增强")
        self.chk_aug.setFont(font_bold)
        self.chk_aug.setChecked(True)
        f.addRow("", self.chk_aug)

        self.spin_hflip = QDoubleSpinBox()
        self.spin_hflip.setFont(font_normal)
        self.spin_hflip.setRange(0.0, 1.0)
        self.spin_hflip.setDecimals(2)
        self.spin_hflip.setValue(0.5)
        f.addRow("水平翻转概率:", self.spin_hflip)

        self.spin_rotation = QSpinBox()
        self.spin_rotation.setFont(font_normal)
        self.spin_rotation.setRange(0, 30)
        self.spin_rotation.setValue(5)
        f.addRow("旋转角度 (±°):", self.spin_rotation)

        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.setFont(font_normal)
        self.spin_brightness.setRange(0.0, 0.5)
        self.spin_brightness.setDecimals(2)
        self.spin_brightness.setValue(0.15)
        f.addRow("亮度扰动:", self.spin_brightness)

        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setFont(font_normal)
        self.spin_contrast.setRange(0.0, 0.5)
        self.spin_contrast.setDecimals(2)
        self.spin_contrast.setValue(0.15)
        f.addRow("对比度扰动:", self.spin_contrast)

        self.spin_affine = QSpinBox()
        self.spin_affine.setFont(font_normal)
        self.spin_affine.setRange(0, 10)
        self.spin_affine.setValue(3)
        f.addRow("仿射平移 (px):", self.spin_affine)

        v.addWidget(g)
        v.addStretch()
        return w

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

    def _on_select_dataset(self):
        default_dir = self._get_default_dir("origin_data")
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录", default_dir)
        if path:
            self.edt_dataset.setText(path)
            self._update_dataset_info(path)

    def _update_dataset_info(self, path):
        folder_to_label, label_to_index, index_to_label, num_classes = discover_dataset_classes(path)
        counts = {}
        total = 0
        for folder, label in folder_to_label.items():
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                cnt = len([f for f in os.listdir(folder_path)
                           if os.path.splitext(f)[1].lower() in ('.jpg', '.png', '.bmp')])
                counts[label] = cnt
                total += cnt
        if counts:
            info = f"共 {num_classes} 类 | {total} 张图片 | " + " | ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
            self.lbl_dataset_info.setText(info)
        else:
            self.lbl_dataset_info.setText("未找到有效图片")

    def _on_select_output(self):
        default_dir = self._get_default_dir("mlp")
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", default_dir)
        if path:
            self.edt_output.setText(path)

    def _on_start_train(self):
        dataset_path = self.edt_dataset.text().strip()
        if not dataset_path or not os.path.isdir(dataset_path):
            QMessageBox.warning(self, "错误", "请选择有效的数据集目录")
            return

        output_dir = self.edt_output.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "错误", "请选择输出目录")
            return
        os.makedirs(output_dir, exist_ok=True)

        model_name = self.edt_model_name.text().strip() or "number_classifier"
        model_path = os.path.join(output_dir, f"{model_name}.pt")

        self.log_edit.clear()
        self._append_log("开始训练...")

        folder_to_label, label_to_index, index_to_label, num_classes = discover_dataset_classes(dataset_path)
        self._append_log(f"发现 {num_classes} 个类别: {', '.join(sorted(label_to_index.keys()))}")

        self._train_worker = MLPTrainWorker(
            model_save_path=model_path,
            dataset_dir=dataset_path,
            epochs=self.spin_epochs.value(),
            batch_size=self.spin_batch.value(),
            lr=self.spin_lr.value(),
            weight_decay=self.spin_wd.value(),
            train_ratio=self.spin_train_ratio.value(),
            augment=self.chk_aug.isChecked(),
            device=self.cmb_device.currentText(),
            num_workers=4,
            architecture=self.architecture,
            input_w=self.spin_input_w.value(),
            input_h=self.spin_input_h.value(),
            folder_to_label=folder_to_label,
            label_to_index=label_to_index,
            index_to_label=index_to_label,
        )
        self._train_worker.log_signal.connect(self._append_log)
        self._train_worker.batch_signal.connect(self._on_batch_progress)
        self._train_worker.progress_signal.connect(self._on_epoch_progress)
        self._train_worker.finished_signal.connect(self._on_finished)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("训练进行中...", "accent")
        self.epoch_progress_bar.setValue(0)
        self.batch_progress_bar.setValue(0)
        self._train_worker.start()

    def _on_stop_train(self):
        if self._train_worker:
            self._train_worker.stop()
            self._append_log("正在停止训练...")

    def _set_status(self, text, role="muted"):
        self.status_label.setText(text)
        self.status_label.setProperty("role", role)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _on_batch_progress(self, batch_idx, total_batches, loss):
        pct = int(batch_idx / total_batches * 100) if total_batches > 0 else 0
        self.batch_progress_bar.setValue(pct)
        self.lbl_batch.setText(f"{batch_idx}/{total_batches}")
        self.loss_label.setText(f"Loss: {loss:.4f}")

    def _on_epoch_progress(self, epoch, total, train_acc, val_acc, train_loss, val_loss):
        self.epoch_label.setText(f"Epoch: {epoch}/{total}")
        pct = int(epoch / total * 100) if total > 0 else 0
        self.epoch_progress_bar.setValue(pct)
        self.acc_label.setText(f"Train: {train_acc:.1f}%  Val: {val_acc:.1f}%")

    def _on_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(msg)
        if success:
            self._set_status("训练完成", "accent")
            self.epoch_progress_bar.setValue(100)
        else:
            self._set_status("训练失败/已停止", "muted")
        if success:
            QMessageBox.information(self, "训练完成", msg)
        else:
            QMessageBox.warning(self, "训练失败", msg)

    def _append_log(self, text):
        self.log_edit.append(text)
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )


# ============================================================================
# 入口
# ============================================================================
if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = MLPTrainWindow()
    w.show()
    sys.exit(app.exec())
