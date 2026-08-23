# -*- coding: utf-8 -*-
"""D3-2 回归：closeEvent 必须等待运行中的 BUFF 标注线程。

原始缺陷：BUFF 全自动推理中关窗 → 子 QThread 运行中被析构 → abort。
本测试构造真实 MainWindow（拦截 3.4GB 模型加载），挂一个假 buff worker，
close() 后断言线程已被等待结束。
"""
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QThread

pytest.importorskip("PySide6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class _SleeperWorker(QThread):
    def __init__(self, seconds=1.0, parent=None):
        super().__init__(parent)
        self._seconds = seconds

    def run(self):
        time.sleep(self._seconds)


@pytest.fixture(scope="module")
def main_window(qapp):
    """构造真实 MainWindow，但拦截两个重量级模型加载。"""
    from core.sam_client import SAMClient
    from core.lora_sam_client import LoraSamClient
    import main as main_mod

    saved = (SAMClient.load_model_async, LoraSamClient.load_model_async)
    SAMClient.load_model_async = lambda self, p: None
    LoraSamClient.load_model_async = lambda self, p, a: None
    try:
        window = main_mod.MainWindow()
    finally:
        SAMClient.load_model_async, LoraSamClient.load_model_async = saved
    yield window
    window.close()  # 触发 closeEvent → 两个 client.cleanup()
    window.deleteLater()


@pytest.mark.gui_main
def test_close_event_waits_for_buff_worker(qtbot, main_window):
    w = main_window
    worker = _SleeperWorker(1.0)
    w._buff_sam3_worker = worker
    worker.start()
    assert worker.isRunning()

    w.close()  # 触发 closeEvent → wait(10000)
    assert not worker.isRunning(), "closeEvent 必须等待 buff 线程结束"
    w._buff_sam3_worker = None


@pytest.mark.gui_main
def test_filter_match_unit(qtbot, main_window):
    """批量状态机的筛选匹配纯逻辑：状态机正确性抽样。"""
    w = main_window

    w._filter_class = None
    w._filter_sort = "default"
    w._filter_aspect_anomaly = False

    # mode=any（仅看有标注）：0 标注不通过，>=1 通过
    w._filter_mode = "any"
    w._image_label_counts = {"a.jpg": 0, "b.jpg": 3}
    assert w._image_matches_filter("a.jpg") is False
    assert w._image_matches_filter("b.jpg") is True

    # mode=ge（>=2 个标注）
    w._filter_mode = "ge"
    w._filter_value = 2
    assert w._image_matches_filter("a.jpg") is False
    assert w._image_matches_filter("b.jpg") is True

    # 类别筛选：标签集合不包含目标类 → 拒绝
    w._filter_mode = "all"
    w._filter_class = "armor_red"
    w._image_label_sets = {"b.jpg": {"armor_blue"}}
    assert w._image_matches_filter("b.jpg") is False
    w._image_label_sets = {"b.jpg": {"armor_red", "car"}}
    assert w._image_matches_filter("b.jpg") is True
