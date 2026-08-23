# -*- coding: utf-8 -*-
"""D1-1 回归：mlp_demo 结果预览必须有内存上限。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.mlp_demo import PREVIEW_MAX_LONG_EDGE, make_preview  # noqa: E402


def _img(h, w):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_big_image_downscaled():
    big = _img(3000, 4000)  # 4000*3000*3 = 36MB
    preview = make_preview(big)
    assert max(preview.shape[:2]) <= PREVIEW_MAX_LONG_EDGE
    assert preview.dtype == np.uint8


def test_small_image_kept_as_is():
    small = _img(480, 640)
    assert make_preview(small) is small, "小图不应额外复制（避免双倍峰值）"


def test_memory_budget_of_batch():
    """模拟批量：1000 张 1080p 结果的预览总量必须远小于原图直存。"""
    results = [make_preview(_img(1080, 1920)) for _ in range(50)]
    per_item = max(int(r.nbytes) for r in results)
    # 1080p 原图 6.2MB；预览长边 1024 → 宽 576 → ~1.7MB
    assert per_item < 2 * 1024 * 1024, "预览单帧应 < 2MB"
    assert len(results) == 50
