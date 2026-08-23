#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 mlp_conv.pt / .onnx 做离线推理测试（不依赖 ros2-framework 代码改动）.

模拟 mix Classify.cpp：20×28 OTSU 灰度 -> /255 -> 推理 -> remap armor id
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.mlp_conv import (  # noqa: E402
    MIX_REMAP_ARMOR_IDS,
    MlpConv,
    NUM_NETWORK_CLASSES,
    preprocess_roi_image,
)

PATH_PT = _REPO_ROOT / "stored_weights/model/armor_tradition/mlp_conv/mlp_conv.pt"
PATH_ONNX = _REPO_ROOT / "stored_weights/model/armor_tradition/mlp_conv/mlp_conv.onnx"
PATH_DATASET = _REPO_ROOT / "datasets/Armor Transform Dataset"
SAMPLE_COUNT = 12


def load_torch_model(checkpoint_path: Path) -> MlpConv:
    model = MlpConv(num_classes=NUM_NETWORK_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict_torch(model: MlpConv, image_path: Path) -> tuple[int, float, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    normalized = preprocess_roi_image(image)
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].numpy()
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    armor_id = MIX_REMAP_ARMOR_IDS[class_id]
    return class_id, confidence, armor_id


def predict_onnx(onnx_path: Path, image_path: Path) -> tuple[int, float, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    normalized = preprocess_roi_image(image)
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    blob = cv2.dnn.blobFromImage(normalized, scalefactor=1.0, size=(20, 28), swapRB=False, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    logits = outputs.reshape(-1).astype(np.float32)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    armor_id = MIX_REMAP_ARMOR_IDS[class_id]
    return class_id, confidence, armor_id


def collect_sample_images(dataset_root: Path, limit: int) -> list[Path]:
    images: list[Path] = []
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        for image_path in sorted(folder.glob("*.jpg"))[:2]:
            images.append(image_path)
            if len(images) >= limit:
                return images
    return images


def main() -> None:
    if not PATH_PT.is_file():
        print(f"[ERROR] 未找到 PT: {PATH_PT}", file=sys.stderr)
        sys.exit(1)

    model = load_torch_model(PATH_PT)
    samples = collect_sample_images(PATH_DATASET, SAMPLE_COUNT)
    print(f"PT: {PATH_PT}")
    print(f"样本: {len(samples)} 张\n")

    for image_path in samples:
        folder_name = image_path.parent.name
        class_id, confidence, armor_id = predict_torch(model, image_path)
        line = (
            f"{folder_name}/{image_path.name} -> "
            f"class={class_id} conf={confidence:.3f} armor_id={armor_id}"
        )
        if PATH_ONNX.is_file():
            onnx_class, onnx_conf, onnx_armor = predict_onnx(PATH_ONNX, image_path)
            line += f" | onnx(class={onnx_class}, conf={onnx_conf:.3f}, armor={onnx_armor})"
        print(line)


if __name__ == "__main__":
    main()
