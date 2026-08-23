# -*- coding: utf-8 -*-
"""L3 资源回归：切图 RSS 边界（审计 T7）。

断言：连续加载 40 张 4K 图后，Canvas 场景只保留单张底图，
RSS 增量必须远小于 40×33MB（全保留时的理论下限 1.3GB）。
"""
import gc
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RSS_BUDGET_MB = 300  # 单张 4K ARGB32 ≈ 33MB；允许 Qt 缓存/噪声，但远低于 1.3GB


@pytest.mark.resource
def test_canvas_image_switch_rss_bounded(qtbot, tmp_path, rss_mb):
    from core.canvas import Canvas

    W, H = 3840, 2160
    import cv2
    import numpy as np

    paths = []
    for i in range(40):
        gray = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        p = tmp_path / f"4k_{i:02d}.jpg"
        # 渐变图压缩快；写入失败直接让测试失败
        assert cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        paths.append(str(p))

    scene = Canvas()
    # 预热 3 张并强制稳定，取基线
    for p in paths[:3]:
        scene.load_image(p)
    gc.collect()
    baseline = rss_mb()

    for p in paths:
        scene.load_image(p)

    gc.collect()
    delta = rss_mb() - baseline
    print(f"\nRSS: baseline={baseline:.0f}MB after={baseline + delta:.0f}MB delta={delta:.0f}MB")
    assert delta < RSS_BUDGET_MB, (
        f"连切 40 张 4K 后 RSS 增长 {delta:.0f}MB 超预算 {RSS_BUDGET_MB}MB —— 疑似像素缓存泄漏"
    )
