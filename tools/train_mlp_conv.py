#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 mlp_conv 分类器（mix Classify 对齐，Armor Transform Dataset）.

直接运行 python tools/train_mlp_conv.py；训练参数在 main() 顶部写死。
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.mlp_conv import (  # noqa: E402
    MAX_ONNX_BYTES,
    ROI_INPUT_HEIGHT,
    ROI_INPUT_WIDTH,
    SampleRecord,
    assert_onnx_size_under_limit,
    build_class_names,
    discover_training_samples,
    estimate_fp32_bytes,
    export_onnx_model,
    preprocess_roi_image,
    save_label_file,
    save_preprocess_config,
    MlpConv,
    NUM_NETWORK_CLASSES,
)

DEFAULT_DATASET_DIR = _REPO_ROOT / "datasets" / "Armor Transform Dataset"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "stored_weights" / "model" / "armor_tradition" / "mlp_conv"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
# 训练/验证/测试 = 8:1:1（分层按类抽样）
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_NUM_WORKERS = 4


class ArmorRoiDataset(Dataset):
    """Armor Transform Dataset -> mix 同款 OTSU 二值 ROI。"""

    def __init__(self, samples: list[SampleRecord], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        record = self.samples[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((ROI_INPUT_HEIGHT, ROI_INPUT_WIDTH), dtype=np.uint8)
        if self.augment:
            image = self._augment(image)
        normalized = preprocess_roi_image(image)
        tensor = torch.from_numpy(normalized).float().unsqueeze(0)
        label = torch.tensor(record.network_class_id, dtype=torch.long)
        return tensor, label

    def _augment(self, gray_image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            gray_image = cv2.flip(gray_image, 1)
        angle = random.uniform(-5.0, 5.0)
        height, width = gray_image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(gray_image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def split_samples(
    samples: list[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    """分层按类抽样，划分 train / val / test（默认 8:1:1）。

    每类内部打乱后按比例切三段；保证 test 至少 1 张（若该类 >=3 张）。
    """
    grouped: dict[int, list[SampleRecord]] = {}
    for sample in samples:
        grouped.setdefault(sample.network_class_id, []).append(sample)
    train_samples: list[SampleRecord] = []
    val_samples: list[SampleRecord] = []
    test_samples: list[SampleRecord] = []
    rng = random.Random(seed)
    for class_samples in grouped.values():
        shuffled = list(class_samples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 1:
            train_samples.extend(shuffled)
            continue
        if n == 2:
            train_samples.append(shuffled[0])
            test_samples.append(shuffled[1])
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        # 防止 train+val 越界，剩余全给 test
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_val = max(0, n - n_train - n_test)
        train_samples.extend(shuffled[:n_train])
        val_samples.extend(shuffled[n_train:n_train + n_val])
        test_samples.extend(shuffled[n_train + n_val:])
    return train_samples, val_samples, test_samples


def build_dataloaders(
    train_samples: list[SampleRecord],
    val_samples: list[SampleRecord],
    test_samples: list[SampleRecord],
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        ArmorRoiDataset(train_samples, augment=augment),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ArmorRoiDataset(val_samples, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ArmorRoiDataset(test_samples, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    for batch_index, (inputs, labels) in enumerate(data_loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct_count += int((predictions == labels).sum().item())
        sample_count += labels.size(0)
    average_loss = total_loss / max(sample_count, 1)
    accuracy = 100.0 * correct_count / max(sample_count, 1)
    return average_loss, accuracy


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """单次评估，返回 (平均 loss, 准确率%)。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_val_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, None, device)
        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"val loss={val_loss:.4f} acc={val_acc:.2f}%"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_accuracy


def save_training_artifacts(model: nn.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    pt_path = output_dir / "mlp_conv.pt"
    onnx_path = output_dir / "mlp_conv.onnx"
    torch.save(model.state_dict(), pt_path)
    export_onnx_model(model, onnx_path)
    onnx_bytes = assert_onnx_size_under_limit(onnx_path, MAX_ONNX_BYTES)
    save_label_file(output_dir, NUM_NETWORK_CLASSES)
    save_preprocess_config(output_dir)
    print(f"PyTorch: {pt_path}")
    print(f"ONNX: {onnx_path} ({onnx_bytes / 1024:.1f} KB, limit {MAX_ONNX_BYTES // 1024} KB)")
    print(f"Labels: {output_dir / 'label.txt'}")
    print(f"Preprocess: {output_dir / 'preprocess.json'}")


def print_dataset_summary(samples: list[SampleRecord]) -> None:
    counts: dict[int, int] = {}
    for sample in samples:
        counts[sample.network_class_id] = counts.get(sample.network_class_id, 0) + 1
    print("样本分布 (network class -> count):")
    for class_id in sorted(counts):
        class_name = build_class_names()[class_id]
        print(f"  {class_id:>2} {class_name:<10}: {counts[class_id]}")


def main() -> None:
    # ===== 写死训练参数（改这里即可） =====
    path_dataset = _REPO_ROOT / "datasets/Armor Transform Dataset"
    path_saved = _REPO_ROOT / "stored_weights/model/armor_tradition/mlp_conv"
    epochs = 200
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-4
    train_ratio = 0.8   # 8:1:1
    val_ratio = 0.1
    num_workers = 4
    seed = 42
    use_augment = True
    width_mult = 1.24  # ~800KB ONNX (202K params)
    device_name = "auto"
    # =====================================

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not path_dataset.is_dir():
        print(f"[ERROR] 数据集不存在: {path_dataset}", file=sys.stderr)
        sys.exit(1)

    samples = discover_training_samples(path_dataset)
    if not samples:
        print(f"[ERROR] 未找到训练样本: {path_dataset}", file=sys.stderr)
        sys.exit(1)

    print_dataset_summary(samples)
    train_samples, val_samples, test_samples = split_samples(
        samples, train_ratio, val_ratio, seed
    )
    print(f"Train/Val/Test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)}")
    print(f"输出: {path_saved} | epochs={epochs}")

    device = resolve_device(device_name)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_samples, val_samples, test_samples, batch_size, num_workers, use_augment
    )

    model = MlpConv(
        num_classes=NUM_NETWORK_CLASSES,
        width_mult=width_mult,
    ).to(device)
    param_bytes = estimate_fp32_bytes(model)
    print(
        f"Model={model.__class__.__name__} | "
        f"params={sum(p.numel() for p in model.parameters())} | "
        f"fp32≈{param_bytes / 1024:.1f} KB"
    )

    started = time.time()
    model, best_val_accuracy = train_model(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    # ===== 最终测试集评估（held-out test set） =====
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(
        f"Test  loss={test_loss:.4f} acc={test_acc:.2f}% "
        f"({len(test_samples)} samples)"
    )

    elapsed = time.time() - started
    print(f"Best val acc: {best_val_accuracy:.2f}% | elapsed {elapsed:.1f}s")

    save_training_artifacts(model, path_saved)
    print("mix 权重已导出；请按目标工程自行拷贝 best.pt / onnx")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
