#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP训练数据增强脚本
将图片处理为 20x28 大小，并添加各种干扰效果
批量从一张图输出多张增强图到指定文件夹
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path

# ============== 用户填入的路径 ==============
INPUT_IMAGE_PATH = "/home/zz/Desktop/LuoHuaLabel/origin_data/截图 2026-05-18 21-54-21.png"  # 输入图片路径
OUTPUT_DIR = "/home/zz/Desktop/LuoHuaLabel/aug_test_output"  # 输出目录
NUM_OUTPUT_PER_IMAGE = 100  # 每张输入图输出多少张

# ============== 增强参数 ==============
TARGET_SIZE = (20, 28)  # W x H
ROTATION_RANGE = (-10, 10)  # 旋转角度范围(度)
SCALE_RANGE = (0.9, 1.5)  # 缩放范围
BRIGHTNESS_RANGE = (0.01, 0.1)  # 亮度范围
CONTRAST_RANGE = (0.6, 1.0)  # 对比度范围
BLUR_KERNEL_RANGE = (0, 0)  # 模糊核大小(奇数)
NOISE_VAR_RANGE = (0, 0)  # 噪声方差
HALO_INTENSITY_RANGE = (0.3, 1)  # 光晕强度


def apply_brightness_contrast(img, brightness=1.0, contrast=1.0):
    """亮度/对比度调整"""
    img = img.astype(np.float32)
    img = contrast * (img - 128) + 128 + brightness * 50
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_light_halo(img, intensity=0.5):
    """左右向中间扩散的光晕效果 - 边缘亮、中心暗"""
    h, w = img.shape[:2]
    original = img.astype(np.float32)
    result = original.copy()
    max_dist = w / 2.0
    boost = intensity * 150

    for x in range(w):
        dist = min(x, w - 1 - x)
        t = dist / max_dist
        delta = (0.5 - t) * boost
        result[:, x] = original[:, x] + delta

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_blur(img, kernel_size=3):
    """高斯模糊"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_noise(img, var=10):
    """添加高斯噪声"""
    sigma = var ** 0.5
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_affine_transform(img, angle=0, scale=1.0, shear=0):
    """仿射变换 (旋转+缩放+剪切)"""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += shear * w
    M[1, 2] += shear * h

    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment_image(img):
    """应用随机增强"""
    aug = img.copy()

    brightness = random.uniform(*BRIGHTNESS_RANGE)
    contrast = random.uniform(*CONTRAST_RANGE)
    aug = apply_brightness_contrast(aug, brightness, contrast)

    if random.random() < 0.5:
        intensity = random.uniform(*HALO_INTENSITY_RANGE)
        aug = apply_light_halo(aug, intensity)

    angle = random.uniform(*ROTATION_RANGE)
    scale = random.uniform(*SCALE_RANGE)
    shear = random.uniform(-0.1, 0.1)
    aug = apply_affine_transform(aug, angle, scale, shear)

    if random.random() < 0.3:
        kernel = random.choice([1, 3, 5])
        aug = apply_blur(aug, kernel)

    if random.random() < 0.3:
        var = random.uniform(*NOISE_VAR_RANGE)
        aug = apply_noise(aug, var)

    return aug


def resize_and_crop(img, target_w, target_h):
    """缩放后中心裁剪到目标尺寸"""
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y = (new_h - target_h) // 2
    x = (new_w - target_w) // 2
    cropped = scaled[y:y+target_h, x:x+target_w]
    return cropped


def process_image(input_path, output_dir, num_output=20):
    """处理单张图片"""
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图片: {input_path}")
        return 0

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = resize_and_crop(img, TARGET_SIZE[0], TARGET_SIZE[1])

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    count = 0
    for i in range(num_output):
        aug = augment_image(img)

        filename = f"{i+1:04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, aug)
        count += 1

    return count


def main():
    if not INPUT_IMAGE_PATH:
        print("错误: 请先设置 INPUT_IMAGE_PATH 变量")
        return

    if not OUTPUT_DIR:
        print("错误: 请先设置 OUTPUT_DIR 变量")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = Path(INPUT_IMAGE_PATH)

    if input_path.is_file():
        count = process_image(str(input_path), OUTPUT_DIR, NUM_OUTPUT_PER_IMAGE)
        print(f"处理完成: {input_path.name} → {count} 张图片")
        print(f"输出目录: {OUTPUT_DIR}")
    elif input_path.is_dir():
        total = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in input_path.glob(ext):
                count = process_image(str(img_path), OUTPUT_DIR, NUM_OUTPUT_PER_IMAGE)
                if count > 0:
                    print(f"处理: {img_path.name} → {count} 张")
                    total += count
        print(f"总计处理: {total} 张图片")
        print(f"输出目录: {OUTPUT_DIR}")
    else:
        print(f"错误: 输入路径不存在: {INPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
