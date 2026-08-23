# -*- coding: utf-8 -*-
"""层级标注自定义训练循环 (DecoupledEdgeYOLO + 自定义 Dataset).

数据格式 (YOLO pose, 5 kpt):
  每行: class_id cx cy w h  kpt0_x kpt0_y kpt0_v ... kpt4_x kpt4_y kpt4_v
  KP0-3 = 四角点 (TL/TR/BL/BR).
  KP4 vis = L2 子类码 (0=none/1=up/2=mid/3=down), 坐标复用 bbox 中心.
  颜色已并入 class_id, 不再单独占 KP (L1 删除).

训练:
  1. HierarchicalDataset 解析 label txt, 返回 image tensor + gt (boxes/kpts/l2).
  2. DecoupledEdgeYOLO.forward_with_gt 输出 pose + L2 logits (L2 stop-gradient).
  3. HierarchicalLoss 组合 CIoU/L1 角点 + L2 CE.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 让脚本独立可跑
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.hierarchical_model import DecoupledEdgeYOLO, HierarchicalLoss
import scripts.hierarchical_paths as P


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class HierSample:
    image_path: str
    image: np.ndarray              # HWC uint8
    boxes: np.ndarray              # [N, 4] xywh (归一化)
    kpts: np.ndarray               # [N, 5, 3]
    l2_codes: np.ndarray           # [N]
    cls_ids: np.ndarray            # [N]


def _letterbox(img: np.ndarray, target: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Top-left 对齐 letterbox (与 ONNX 导出一致)."""
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target - new_w
    pad_h = target - new_h
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale, pad_w, pad_h


def _parse_label_line(line: str, kpt_count: int = 5) -> tuple[int, np.ndarray, np.ndarray]:
    """解析一行 YOLO pose label.
    返回 (cls_id, box_xywh[4], kpts[kpt_count, 3]).
    """
    parts = line.strip().split()
    cls_id = int(parts[0])
    box = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
    kpt_flat = np.array([float(x) for x in parts[5:]], dtype=np.float32).reshape(-1, 3)
    if kpt_flat.shape[0] < kpt_count:
        pad = np.zeros((kpt_count - kpt_flat.shape[0], 3), dtype=np.float32)
        kpt_flat = np.concatenate([kpt_flat, pad], axis=0)
    return cls_id, box, kpt_flat[:kpt_count]


class HierarchicalDataset(Dataset):
    def __init__(self, images_dir: str | Path, labels_dir: str | Path,
                 imgsz: int = 640, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.samples: list[Path] = []
        for ext in P.IMG_SUFFIXES:
            for p in self.images_dir.rglob(f"*{ext}"):
                self.samples.append(p)
        self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path = self.samples[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        img_lb, scale, _, _ = _letterbox(img, self.imgsz)

        boxes = []
        kpts = []
        l2_codes = []
        cls_ids = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cls_id, box, kpt = _parse_label_line(line, P.KPT_SHAPE[0])
                    # box xywh 归一化 → 像素 (letterbox 空间)
                    cx, cy, bw, bh = box
                    cls_ids.append(cls_id)
                    boxes.append([cx * W * scale, cy * H * scale, bw * W * scale, bh * H * scale])
                    # kpts 归一化 → 像素
                    kpt_px = kpt.copy()
                    kpt_px[:, 0] = kpt[:, 0] * W * scale
                    kpt_px[:, 1] = kpt[:, 1] * H * scale
                    kpts.append(kpt_px)
                    # KP4 = L2 子类 (vis 编码); 非 BO/RO 兵种 L2 恒 0
                    l2_codes.append(int(kpt[4, 2]) if kpt.shape[0] > 4 else 0)

        img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.tensor(boxes if boxes else np.zeros((0, 4), dtype=np.float32),
                                  dtype=torch.float32),
            "kpts": torch.tensor(kpts if kpts else np.zeros((0, P.KPT_SHAPE[0], 3), dtype=np.float32),
                                 dtype=torch.float32),
            "l2": torch.tensor(l2_codes if l2_codes else np.zeros((0,), dtype=np.int64),
                               dtype=torch.int64),
            "cls": torch.tensor(cls_ids if cls_ids else np.zeros((0,), dtype=np.int64),
                                dtype=torch.int64),
        }
        return img_tensor, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def train_one_epoch(model: DecoupledEdgeYOLO, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, loss_fn: HierarchicalLoss,
                    device: torch.device, epoch: int, scaler: Optional[torch.cuda.amp.GradScaler] = None):
    model.train()
    total_loss = 0.0
    n = 0
    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        # 合并 batch 的 gt
        all_boxes = []
        all_kpts = []
        all_l2 = []
        all_batch_idx = []
        for b_idx, t in enumerate(targets):
            if t["boxes"].shape[0] > 0:
                all_boxes.append(xywh_to_xyxy(t["boxes"].to(device)))
                all_kpts.append(t["kpts"].to(device))
                all_l2.append(t["l2"].to(device))
                all_batch_idx.append(torch.full((t["boxes"].shape[0],), b_idx,
                                                dtype=torch.long, device=device))
        if all_boxes:
            gt_boxes = torch.cat(all_boxes, dim=0)
            gt_kpts = torch.cat(all_kpts, dim=0)
            gt_l2 = torch.cat(all_l2, dim=0)
            gt_batch_idx = torch.cat(all_batch_idx, dim=0)
        else:
            gt_boxes = torch.zeros((0, 4), device=device)
            gt_kpts = torch.zeros((0, P.KPT_SHAPE[0], 3), device=device)
            gt_l2 = torch.zeros((0,), dtype=torch.int64, device=device)
            gt_batch_idx = None

        optimizer.zero_grad()
        if scaler is not None and P.AMP:
            with torch.cuda.amp.autocast():
                out = model.forward_with_gt(imgs, gt_boxes, gt_batch_idx)
                loss_dict = loss_fn(out["pose_preds"], gt_boxes, gt_kpts,
                                    out["l2_logits"], gt_l2)
            scaler.scale(loss_dict["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model.forward_with_gt(imgs, gt_boxes, gt_batch_idx)
            loss_dict = loss_fn(out["pose_preds"], gt_boxes, gt_kpts,
                                out["l2_logits"], gt_l2)
            loss_dict["total"].backward()
            optimizer.step()

        total_loss += loss_dict["total"].item()
        n += 1
        if i % 20 == 0:
            print(f"[epoch {epoch}] iter {i}/{len(loader)}  "
                  f"total={loss_dict['total'].item():.4f}  "
                  f"box={loss_dict['box'].item():.4f}  kpt={loss_dict['kpt'].item():.4f}  "
                  f"l2={loss_dict['l2'].item():.4f}")
    return total_loss / max(n, 1)


def main():
    P.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    P.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(P.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_ds = HierarchicalDataset(P.TRAIN_IMAGES, P.TRAIN_LABELS,
                                   imgsz=P.TRAIN_IMGSZ, augment=True)
    val_ds = HierarchicalDataset(P.VAL_IMAGES, P.VAL_LABELS,
                                 imgsz=P.TRAIN_IMGSZ, augment=False)
    print(f"train samples = {len(train_ds)}, val samples = {len(val_ds)}")
    if len(train_ds) == 0:
        print("ERROR: 训练集为空, 请先用 dataset_tool 生成层级 YOLO pose 数据.")
        return

    train_loader = DataLoader(train_ds, batch_size=P.BATCH_SIZE, shuffle=True,
                              num_workers=P.WORKERS, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=P.BATCH_SIZE, shuffle=False,
                            num_workers=P.WORKERS, collate_fn=collate_fn,
                            pin_memory=True) if len(val_ds) > 0 else None

    model = DecoupledEdgeYOLO(
        num_classes=P.NUM_CLASSES,
        num_l2=P.NUM_L2,
    ).to(device)
    loss_fn = HierarchicalLoss(
        num_l2=P.NUM_L2,
        pose_box_w=P.POSE_BOX_W, pose_kpt_w=P.POSE_KPT_W,
        l2_w=P.L2_W,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=P.LR, weight_decay=P.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=P.EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if (P.AMP and torch.cuda.is_available()) else None

    best_loss = float("inf")
    for epoch in range(P.EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                  device, epoch, scaler)
        scheduler.step()
        # 保存
        ckpt_path = P.WEIGHTS_DIR / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        if tr_loss < best_loss:
            best_loss = tr_loss
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       P.WEIGHTS_DIR / "best.pt")
        print(f"== epoch {epoch} done  train_loss={tr_loss:.4f}  best={best_loss:.4f} ==")

    print("训练完成. 权重:", P.WEIGHTS_DIR)


if __name__ == "__main__":
    main()
