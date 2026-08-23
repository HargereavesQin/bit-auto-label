"""
core/debug.py — 调试模式基础设施.

模式控制:
  - 默认关闭. 开启方式:
    1) 命令行: python main.py --debug
    2) 设置对话框里勾选"调试模式" (见 ui/settings_dialog.py)
    3) 代码: from core.debug import DebugMode; DebugMode.enable()

功能:
  - print() 自动转为带时间戳 + 模块名的 logging
  - 异常全栈自动写入 logs/debug_{date}.log
  - 可选: event_bus 信号 emit 记录
  - 可选: SAM/YOLO/MLP Worker 关键阶段记录

用法:
  from core.debug import debug_log
  debug_log("SAM model loaded")  # 仅调试模式开启时有效
"""
from __future__ import annotations
import os
import sys
import atexit
import traceback
from datetime import datetime, timezone
from typing import TextIO

# ---------------------------------------------------------------------------
# 单例状态
# ---------------------------------------------------------------------------

_log_file: TextIO | None = None
_enabled: bool = False


class DebugMode:
    """全局调试模式控制器. 只在第一次 enable() 时初始化日志文件."""

    @staticmethod
    def enable() -> bool:
        global _enabled, _log_file
        if _enabled:
            return True
        _enabled = True
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(logs_dir, f"debug_{date_str}.log")
            _log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            _log_file.write(f"[DEBUG] session started {datetime.now(timezone.utc).isoformat()}\n")
            atexit.register(DebugMode._close)
            return True
        except Exception:
            _enabled = False
            return False

    @staticmethod
    def disable() -> None:
        global _enabled
        _enabled = False
        DebugMode._close()

    @staticmethod
    def is_enabled() -> bool:
        return _enabled

    @staticmethod
    def _close() -> None:
        global _log_file
        if _log_file is not None:
            try:
                _log_file.write(f"[DEBUG] session ended {datetime.now(timezone.utc).isoformat()}\n")
                _log_file.close()
            except Exception:
                pass
            _log_file = None


def init_debug_mode() -> None:
    """应用启动时调用一次. 检测 --debug 命令行参数并启用."""
    if "--debug" in sys.argv:
        DebugMode.enable()


def debug_log(msg: str) -> None:
    """仅调试模式开启时写日志."""
    if not _enabled or _log_file is None:
        return
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    _log_file.write(f"{ts}  {msg}\n")


def _debug_excepthook(exc_type, exc_value, exc_tb):
    """调试模式下把未捕获异常写入日志, 然后恢复默认处理."""
    if _enabled and _log_file is not None:
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _log_file.write(f"\n[UNCAUGHT] {datetime.now(timezone.utc).isoformat()}\n{tb_str}\n")
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def install_debug_excepthook() -> None:
    """替换 sys.excepthook, 调试模式下捕获未处理异常."""
    if _enabled:
        sys.excepthook = _debug_excepthook
