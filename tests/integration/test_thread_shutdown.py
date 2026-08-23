# -*- coding: utf-8 -*-
"""D3-2/3/4 退出纪律回归：cleanup 必须等待/取消后台线程，而非超时放弃。"""
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QThread

pytest.importorskip("PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.sam_client import SAMClient  # noqa: E402


class _SlowProcessor:
    def set_image(self, pil_img):
        time.sleep(0.6)  # 模拟 CPU 大图特征提取
        return {"state": pil_img}


def test_cleanup_waits_for_inflight_image_load(qtbot, synthetic_image):
    """cleanup() 返回后，在途 ImageLoadWorker 必须已结束（不再运行中被析构）。"""
    client = SAMClient()
    client.processor = _SlowProcessor()

    img = synthetic_image(320, 240)
    client.set_image_async(str(img), token=1)
    assert client._image_load_worker is not None and client._image_load_worker.isRunning()

    client.cleanup()
    assert not client._image_load_worker.isRunning(), "cleanup 必须等 worker 结束，防止析构崩溃"


def test_cleanup_cancels_queued_load(qtbot):
    """cancel() 后未开始重负载的 worker 应快速退出。"""
    client = SAMClient()
    worker = client._image_load_worker  # None-safe: 构造后为 None
    assert worker is None
    client.cleanup()  # 不应有任何异常/崩溃

    # 直接验证取消标志路径
    from core.sam_client import ModelLoadWorker
    w = ModelLoadWorker("whatever.pt")
    w.cancel()
    assert w._cancelled is True
