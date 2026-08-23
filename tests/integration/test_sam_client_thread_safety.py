# -*- coding: utf-8 -*-
"""D3-1 崩溃回归：模型加载线程重入必须被守卫拦截。

原始缺陷：模型加载期间（3.4GB，5-30s 窗口）按住 Q 键 auto-repeat 反复触发
load_model_async → 旧 ModelLoadWorker 引用被覆盖 → QThread 运行中析构 → qFatal SIGABRT。
"""
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QThread, Signal

pytest.importorskip("PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import sam_client as sam_client_mod  # noqa: E402


class _SlowLoadWorker(QThread):
    """模拟 3.4GB checkpoint 的慢加载（原 ModelLoadWorker 的可控替身）。"""

    loaded = Signal(object, object, bool, str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for _ in range(100):  # 最长 2s
            if self._cancelled:
                break
            time.sleep(0.02)
        self.loaded.emit(None, None, False, "测试替身完成")


@pytest.fixture
def qtclient(qtbot):
    from core.sam_client import SAMClient

    client = SAMClient()
    yield client
    client.cleanup()


def test_load_model_async_reentry_guarded(qtbot, monkeypatch, qtclient):
    """第二次重入不得新建/替换运行中的 worker（D3-1 修复的精确断言）。"""
    monkeypatch.setattr(sam_client_mod, "ModelLoadWorker", _SlowLoadWorker)

    statuses = []
    qtclient.model_status_changed.connect(lambda ok, msg: statuses.append(msg))

    qtclient.load_model_async("fake.pt")
    first = qtclient.load_worker
    assert first is not None and first.isRunning()

    # 模拟按住 Q 键的 auto-repeat 连击
    for _ in range(10):
        qtclient.load_model_async("fake.pt")

    assert qtclient.load_worker is first, "守卫必须保持同一 worker 实例"
    assert any("请勿重复触发" in m for m in statuses), "应向用户提示正在加载"

    qtbot.waitUntil(lambda: not first.isRunning(), timeout=10000)
    qtclient.load_model_async("fake.pt")  # 结束后允许再次加载
    assert qtclient.load_worker is not first


def test_lora_load_model_async_guard_exists():
    """LoRA 客户端是同构副本，守卫必须同样存在（静态源码断言）。"""
    src = Path(sam_client_mod.__file__).with_name("lora_sam_client.py").read_text(encoding="utf-8")
    body = src.split("def load_model_async")[1].split("\n    def ")[0]
    assert "isRunning()" in body, "lora_sam_client.load_model_async 必须含 isRunning 守卫"
