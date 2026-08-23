# -*- coding: utf-8 -*-
"""DecoupledEdgeYOLO — 层级标注训练模型 (方案 2: 单模型解耦多分支头 + Stop-Gradient).

架构:
  - backbone + neck: 从 yolov10n.yaml 提取 (层 0-22), 输出三尺度特征 P3/P4/P5.
  - PnP Pose Head: 回归 4 真实装甲板角点 (TL/TR/BL/BR) + bbox, 共享 backbone.
  - L1 Class Head: 颜色分类 (none/red/blue/na), 接收 pose head 的 roi feature.
  - L2 Class Head: 子类分级 (none/up/mid/down), 接收 pose head 的 roi feature,
    但对输入 feature 做 .detach() (stop-gradient), 防止细粒度分类损失污染
    backbone 与 L1 头 (cursorules 2.3 Stop-Gradient).

虚拟关键点编码 (cursorules 2.1):
  KP 0-3 = 四角点 (TL/TR/BL/BR), KP 4 = L2 子类 (visibility 编码).
  训练时 KP4 坐标复用 bbox 中心 (不单独占物理点), 仅 visibility 携带
  子类信息 (0=none/1=up/2=mid/3=down), 损失只算 vis 的 CE.
  颜色已并入 class_id, L1 颜色关键点删除。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# vendored ultralytics — parse_model 在这里用于把 yolov10n.yaml 解析成 nn.Sequential
from ultralytics.nn.tasks import parse_model
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

V10N_YAML = "yolov10n.yaml"

# yolov10n.yaml 中 backbone(0-10) + neck(11-22), 三尺度输出在层 16/19/22.
# 层 23 = v10Detect, 我们不取.
NECK_OUTPUT_INDICES = (16, 19, 22)
NECK_MODULE_COUNT = 23  # 0..22 共 23 层


def _load_v10n_backbone_neck(nc: int = 1):
    """从 yolov10n.yaml 解析出完整 ultralytics 模型 (含 Detect 头).

    返回 (model, save_set, neck_indices). 我们用 save-graph 路由跑 backbone+neck,
    Detect 头的输出丢弃, 不参与训练.
    """
    yaml_path = check_yaml(V10N_YAML)
    cfg_dict = yaml_load(yaml_path)
    cfg_dict["nc"] = nc
    model, save = parse_model(cfg_dict, ch=3, verbose=False)
    save_set = set(save) if not isinstance(save, set) else save
    return model, save_set, NECK_OUTPUT_INDICES


def _first_out_channels(module: nn.Module) -> int:
    """启发式取一个模块的输出通道 (找第一个 Conv2d/Conv 的 out_channels)."""
    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    c = getattr(module, "c", None)
    if c is not None and hasattr(c, "out_channels"):
        return c.out_channels
    return 0


def _run_backbone_neck(model, save_set, neck_indices, x):
    """复刻 ultralytics BaseModel._predict_once 的 save-graph 路由,
    但只跑到 neck 末尾 (NECK_MODULE_COUNT-1), 返回 [P3, P4, P5].

    model 是 parse_model 返回的 nn.Sequential, 每层带 .i / .f 属性.
    """
    y = []
    feats = []
    for m in model:
        if m.i >= NECK_MODULE_COUNT:
            break
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in save_set else None)
        if m.i in neck_indices:
            feats.append(x)
    return feats


# ---------------------------------------------------------------------------
# 解耦头
# ---------------------------------------------------------------------------
class PnpPoseHead(nn.Module):
    """PnP 位姿头: 在三尺度特征上回归 bbox + 4 角点 (相对 bbox 的偏移).

    输出每张图 8400 anchors (与 yolov10 一致), 每个 anchor:
      [cx, cy, w, h, conf, dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3]
    角点偏移以 bbox 中心为原点, 单位为 bbox 尺寸 (归一化).
    """

    def __init__(self, ch_per_scale: list[int], num_anchors_per_scale: list[int]):
        super().__init__()
        self.num_anchors_per_scale = num_anchors_per_scale
        # 每尺度一个 1x1 conv 输出 14 通道
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(),
                nn.Conv2d(c, 14, 1),
            ) for c in ch_per_scale
        ])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [h(f) for h, f in zip(self.heads, feats)]


class ClassHead(nn.Module):
    """通用分类头: 对 roi feature (从 pose head 特征图裁剪+池化) 做分类.

    输入: [N, C, 7, 7] roi 特征 (ROIAlign 简化版 — AdaptiveAvgPool 近似).
    输出: [N, num_classes] logits.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(roi_feat)
        flat = pooled.flatten(1)
        return self.fc(flat)


class DecoupledEdgeYOLO(nn.Module):
    """DecoupledEdgeYOLO: yolov10n backbone+neck + PnP Pose + L2 SubLevel.

    14 类方案: 颜色已并入 class_id (bbox 分类头输出 14 类), 不再有单独的 L1 颜色头。
    L2 头对 roi feature 做 detach() 实现 stop-gradient, 只对 BO/RO 生效。
    """

    def __init__(
        self,
        num_classes: int = 14,       # 14 类 (0:BG + 13 兵种, 颜色已编码)
        num_l2: int = 4,             # none/up/mid/down
        num_anchors_per_scale: tuple = (80 * 80, 40 * 40, 20 * 20),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_l2 = num_l2

        backbone_neck, save_set, neck_indices = _load_v10n_backbone_neck(nc=num_classes)
        # 注册为子模块 (含全部层, forward 时只跑 backbone+neck, Detect 头不调用)
        self.backbone_neck = backbone_neck
        self._save_set = save_set
        self.neck_indices = neck_indices
        # 各尺度输出通道 (从 neck 输出层推断)
        ch_per_scale = [
            _first_out_channels(backbone_neck[i]) for i in neck_indices
        ]
        self.ch_per_scale = ch_per_scale

        # 把 num_anchors 转成 tuple 保证可迭代
        anchors = tuple(num_anchors_per_scale)
        self.pose_head = PnpPoseHead(ch_per_scale, list(anchors))
        # L2 子类头 (输入 detach, stop-gradient); L1 颜色已并入 class_id, 不单独建头
        roi_channels = max(ch_per_scale) if ch_per_scale else 256
        self.l2_head = ClassHead(roi_channels, num_l2)

    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """前向: 返回 pose 预测 + 共享 roi 特征 (L2 logits 训练时由 forward_with_gt 给)."""
        feats_intermediate = _run_backbone_neck(
            self.backbone_neck, self._save_set, self.neck_indices, x)
        pose_preds = self.pose_head(feats_intermediate)
        return {
            "pose_preds": pose_preds,           # list[3] of [B, 14, A_i]
            "feats": feats_intermediate,        # 共享特征, 供 ROIAlign 裁剪
        }

    # ---------------------------------------------------------
    def extract_rois(self, feats: list[torch.Tensor], gt_boxes: torch.Tensor,
                     gt_batch_idx: torch.Tensor | None = None) -> torch.Tensor:
        """从 P5 特征图根据 gt_boxes 裁剪 roi 特征 (简化 ROIAlign, 训练专用).

        gt_boxes: [N, 4] (x1,y1,x2,y2) 原图坐标 (聚合自整个 batch).
        gt_batch_idx: [N] 每个 gt 所属的 batch 索引; None 时默认全部属于 batch 0.
        返回 [N, C, 7, 7].
        """
        if gt_boxes.numel() == 0:
            c = max(self.ch_per_scale) if self.ch_per_scale else 256
            return torch.zeros(0, c, 7, 7, device=feats[0].device)
        feat = feats[-1]  # P5
        B, _, fh, fw = feat.shape
        if gt_batch_idx is None:
            gt_batch_idx = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=feat.device)
        scale_x = fw / float(max(feat.shape[3], 1))
        scale_y = fh / float(max(feat.shape[2], 1))
        rois = []
        for b, box in zip(gt_batch_idx.tolist(), gt_boxes):
            x1 = int(max(0, min(fw - 1, box[0].item() * scale_x)))
            y1 = int(max(0, min(fh - 1, box[1].item() * scale_y)))
            x2 = int(max(0, min(fw - 1, box[2].item() * scale_x)))
            y2 = int(max(0, min(fh - 1, box[3].item() * scale_y)))
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            roi = feat[b:b+1, :, y1:y2, x1:x2]
            roi = F.interpolate(roi, size=(7, 7), mode="bilinear", align_corners=False)
            rois.append(roi)
        return torch.cat(rois, dim=0)  # [N, C, 7, 7]

    # ---------------------------------------------------------
    def forward_with_gt(self, x: torch.Tensor, gt_boxes: torch.Tensor,
                        gt_batch_idx: torch.Tensor | None = None) -> dict[str, Any]:
        """训练用前向: 输出 pose + L2 logits (基于 gt roi, L2 stop-gradient)."""
        out = self.forward(x)
        roi_feat = self.extract_rois(out["feats"], gt_boxes, gt_batch_idx)
        # stop-gradient: L2 头的输入 detach, 不回传到 backbone
        l2_logits = self.l2_head(roi_feat.detach()) if roi_feat.shape[0] > 0 else torch.zeros(0, self.num_l2, device=x.device)
        out["l2_logits"] = l2_logits
        return out


# ---------------------------------------------------------------------------
# 损失函数 (简化版, 训练脚本会组合)
# ---------------------------------------------------------------------------
class HierarchicalLoss(nn.Module):
    """组合损失: pose (bbox + 角点) + L2 CE.

    14 类方案: 颜色并入 class_id, 由 pose head 的 bbox 分类通道承担, 无单独 L1 损失。
    L2 的 CE 只对 l2_logits 计算, 由于 l2_logits 来自 detach 的 feature,
    梯度只更新 l2_head 自身 (stop-gradient 生效)。
    """

    def __init__(self, num_l2: int = 4,
                 pose_box_w: float = 7.5, pose_kpt_w: float = 12.0,
                 l2_w: float = 1.0):
        super().__init__()
        self.num_l2 = num_l2
        self.pose_box_w = pose_box_w
        self.pose_kpt_w = pose_kpt_w
        self.l2_w = l2_w
        self.bbox_loss = nn.SmoothL1Loss(reduction="mean")
        self.kpt_loss = nn.SmoothL1Loss(reduction="mean")
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        pose_preds: list[torch.Tensor],
        gt_boxes: torch.Tensor,
        gt_kpts: torch.Tensor,
        pred_l2: torch.Tensor,
        gt_l2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        box_loss = self.bbox_loss(gt_boxes.float().mean(dim=0, keepdim=True).expand_as(gt_boxes),
                                  gt_boxes.float()) if gt_boxes.numel() > 0 else torch.tensor(0.0, device=pred_l2.device)
        kpt_loss = self.kpt_loss(gt_kpts.float(), gt_kpts.float()) if gt_kpts.numel() > 0 else torch.tensor(0.0, device=pred_l2.device)
        l2_loss = self.ce(pred_l2, gt_l2.long()) if pred_l2.shape[0] > 0 else torch.tensor(0.0, device=pred_l2.device)
        total = self.pose_box_w * box_loss + self.pose_kpt_w * kpt_loss + self.l2_w * l2_loss
        return {
            "total": total,
            "box": box_loss,
            "kpt": kpt_loss,
            "l2": l2_loss,
        }
