# -*- coding: utf-8 -*-
"""ArmorLabel 测试全局夹具。

- 强制 Qt 离屏渲染（无窗口环境下可跑 GUI 逻辑）
- faulthandler 捕获段错误栈（按键连按/线程析构崩溃类回归的守夜人）
- 提供：repo_root / synthetic_image / rss 等夹具
"""
import os
import sys
from pathlib import Path

# 必须在 import PySide6 / pytest-qt 之前设置（Linux CI 无显示器）
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("ARMORLABEL_TEST", "1")

import faulthandler

faulthandler.enable()

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

# 把仓库根插到 site-packages 之后：本地 core/ui/tools 可导入，
# 同时避免残缺的 vendored ultralytics/ 抢在 pip 包前面。
_root = str(ROOT)
if _root in sys.path:
    sys.path.remove(_root)
_site_idxs = [i for i, p in enumerate(sys.path) if "site-packages" in p.replace("\\", "/")]
_insert_at = (max(_site_idxs) + 1) if _site_idxs else len(sys.path)
sys.path.insert(_insert_at, _root)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


def _gradient_image(width: int, height: int, seed: int) -> np.ndarray:
    """渐变图：编码快、文件小，足以触发真实解码路径。"""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, width, dtype=np.uint8)
    img = np.tile(base, (height, 1))
    noise = rng.integers(0, 6, size=(height, width), dtype=np.uint8)
    gray = ((img.astype(np.uint16) + noise) % 256).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def synthetic_image(tmp_path: Path):
    """返回工厂：写一张合成 JPEG，返回路径。"""
    counter = {"n": 0}

    def _make(width=640, height=480, seed=None):
        seed = seed if seed is not None else counter["n"]
        counter["n"] += 1
        img = _gradient_image(width, height, seed)
        path = tmp_path / f"img_{counter['n']:04d}.jpg"
        assert cv2.imwrite(str(path), img), f"无法写入 {path}"
        return path

    return _make


@pytest.fixture
def rss_mb():
    """进程 RSS(MB) 采样器。"""
    import psutil

    proc = psutil.Process(os.getpid())

    def _sample() -> float:
        return proc.memory_info().rss / (1024 * 1024)

    return _sample
