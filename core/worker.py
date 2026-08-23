"""
core/worker.py — 后台 Worker 基类.

历史包袱: 项目里有 8+ 个 QThread 子类, 全部自己定义
  log_signal / progress_signal / finished_signal / run()
且都没包 try/except, 异常直接吞.

本基类统一: 子类只覆写 _do_work(), 框架负责:
  - 三种标准信号
  - 异常捕获并转化为 failed result
  - 日志转发 (可连接到 UI console)
"""
from __future__ import annotations
import traceback
from PySide6.QtCore import QThread, Signal


class BaseWorker(QThread):
    """所有后台线程的基类.

    子类示例:
        class MyWorker(BaseWorker):
            def _do_work(self):
                for i in range(100):
                    self.emit_log(f"step {i}")
                    self.emit_progress(i + 1, 100)
                self.emit_done(True, "ok")

    主线程用法:
        w = MyWorker()
        w.log_signal.connect(console.append)
        w.progress_signal.connect(bar.set_progress)
        w.finished_signal.connect(on_finish)
        w.start()
    """

    log_signal = Signal(str)
    progress_signal = Signal(int, int)   # current, total
    finished_signal = Signal(bool, str)  # success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    # ---- 子类调用 ----
    def emit_log(self, msg: str) -> None:
        self.log_signal.emit(msg)

    def emit_progress(self, current: int, total: int) -> None:
        self.progress_signal.emit(current, total)

    def emit_done(self, success: bool = True, message: str = "") -> None:
        self.finished_signal.emit(success, message)

    def cancel(self) -> None:
        """请求取消. 子类在 _do_work 中定期检查 self._cancelled."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    # ---- 框架 ----
    def run(self) -> None:
        """框架入口. 异常会被捕获并发送 failed 信号."""
        try:
            self._do_work()
            if not self._cancelled:
                self.finished_signal.emit(True, "")
        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f"[ERR] {e}\n{tb}")
            self.finished_signal.emit(False, str(e))

    def _do_work(self) -> None:
        """子类覆写. 不要在这里 catch 后吞掉异常, 让框架处理."""
        raise NotImplementedError
