# -*- coding: utf-8 -*-
"""性能基线（pytest-benchmark）：为核心热路径建立时间基准，防止性能退化悄悄合入。

基线选择：
- make_preview：mlp_demo 每张图回调的必经路径
- resolve_open_layout：打开任何目录的入口路径
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture
def preview_source():
    import numpy as np

    return np.random.default_rng(7).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)


def test_benchmark_make_preview(benchmark, preview_source):
    from tools.mlp_demo import make_preview

    result = benchmark(make_preview, preview_source)
    assert max(result.shape[:2]) <= 1024


def test_benchmark_resolve_open_layout(benchmark, tmp_path, synthetic_image):
    from core.dataset_layout import resolve_open_layout

    for i in range(200):
        img = synthetic_image(64, 48, seed=i)
        dst = tmp_path / img.name
        shutil.move(str(img), str(dst))

    def _run():
        return resolve_open_layout(str(tmp_path))

    result = benchmark(_run)
    assert len(result.image_paths) == 200
