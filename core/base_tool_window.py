"""
core/base_tool_window.py — Tool 窗口基类.

项目里有 8 个 standalone tool 窗口 (yolo_train / mlp_train / ...) 都各自:
  - 加载 EDITORIAL_QSS
  - 装 header bar (icon + title)
  - 装 status dot
  - 装 log console
  - 持久化 config 到 ~/.config/LuoHuaLabel/tool_configs.yaml
  - 处理 closeEvent
重复 ~200 行 / 每个 tool.

本基类把这些都收敛进来, 子类只写"我自己的内容"即可.
"""
from __future__ import annotations
import os
from typing import Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTextEdit, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QTextCursor

from .event_bus import event_bus
from ui.icons import Icon, Icons, C


# 编辑风样式 — 由 main.py 集中加载, 这里也兜底加载一次
def _load_editorial_qss() -> str:
    qss_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()


class StatusDot(QWidget):
    """标题栏右侧 idle/working/done/error 小圆点."""

    def __init__(self, state: str = "idle", parent=None):
        super().__init__(parent)
        self.setObjectName("statusDot")
        self.setProperty("state", state)
        self.setFixedSize(8, 8)

    def set_state(self, state: str) -> None:
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class BaseToolWindow(QMainWindow):
    """所有 standalone tool 窗口的基类.

    用法:
        class YoloTrainWindow(BaseToolWindow):
            TOOL_NAME = "yolo_train"   # 唯一标识, 用于 config 持久化

            def build_ui(self):
                self.add_section("数据集  /  DATA", self._build_data_form())
                self.add_execute_button("开始训练", self._on_start)
                self.add_log_console(min_height=160)

            def _build_data_form(self):
                ...
                return widget
    """

    TOOL_NAME: str = ""   # 子类必须设, 用于 config 持久化
    WINDOW_TITLE: str = "Tool"

    def __init__(self, parent=None, icon: Optional[QIcon] = None, title: Optional[str] = None, subtitle: str = ""):
        super().__init__(parent)
        self.setStyleSheet(EDITORIAL_QSS)
        self.setWindowTitle(title or self.WINDOW_TITLE)
        if icon:
            self.setWindowIcon(icon)
        self._subtitle = subtitle

        # ---- 中心 widget 布局 ----
        self._central = QWidget()
        self.setCentralWidget(self._central)
        self._outer = QVBoxLayout(self._central)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        # ---- Header ----
        self._build_header()

        # ---- Body 容器 ----
        self._body = QVBoxLayout()
        self._body.setContentsMargins(28, 24, 28, 18)
        self._body.setSpacing(20)
        self._outer.addLayout(self._body, 1)

        # ---- Status dot / label (默认 idle) ----
        self._status_dot = StatusDot("idle", self)
        self._status_lbl = QLabel("就绪")
        self._status_lbl.setProperty("role", "status_info")

        # ---- 子类自己加东西 ----
        try:
            self.build_ui()
        except Exception as e:
            import traceback
            print(f"[BaseToolWindow] build_ui failed: {e}")
            traceback.print_exc()

        # ---- 自动加载 config ----
        self._load_tool_config()

        # 广播
        if self.TOOL_NAME:
            event_bus().tool_window_opened.emit(self.TOOL_NAME)

    # ---------------------------------------------------------
    # Header (icon + 衬线标题 + 副标题 + 状态点)
    # ---------------------------------------------------------
    def _build_header(self) -> None:
        header = QWidget()
        header.setObjectName("headerBar")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(28, 18, 28, 16)
        h_layout.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title = QLabel(self.windowTitle().split("  /  ")[0].split(" · ")[0])
        title.setProperty("role", "serif_title")
        text_col.addWidget(title)

        if self._subtitle:
            sub = QLabel(self._subtitle)
            sub.setProperty("role", "serif_subtitle")
            text_col.addWidget(sub)
        h_layout.addLayout(text_col, 1)

        # 状态点 + label
        status_col = QHBoxLayout()
        status_col.setSpacing(8)
        status_col.addWidget(self._status_dot, 0, Qt.AlignVCenter)
        status_col.addWidget(self._status_lbl, 0, Qt.AlignVCenter)
        h_layout.addLayout(status_col, 0)

        self._outer.addWidget(header)

    # ---------------------------------------------------------
    # 公共 API — 给子类用
    # ---------------------------------------------------------
    def add_section(self, label: str, body_widget: QWidget) -> None:
        """加一组 [小标题 + 内容 widget]."""
        sec = QLabel(label)
        sec.setProperty("role", "section")
        self._body.addWidget(sec)
        self._body.addWidget(body_widget)

    def add_widget(self, widget: QWidget) -> None:
        """直接加一个 widget 到 body (不分组)."""
        self._body.addWidget(widget)

    def add_layout(self, layout) -> None:
        """直接加一个 layout 到 body."""
        self._body.addLayout(layout)

    def add_log_console(self, min_height: int = 160) -> QTextEdit:
        """加底部日志控制台, 同时返回引用, 子类 log() 用."""
        sec = QLabel("处理日志  /  CONSOLE")
        sec.setProperty("role", "section")
        self._body.addWidget(sec)
        console = QTextEdit()
        console.setReadOnly(True)
        console.setProperty("role", "log")
        console.setMinimumHeight(min_height)
        self._body.addWidget(console, 1)
        self._console = console
        return console

    def add_stretch(self) -> None:
        self._body.addStretch()

    def log(self, text: str) -> None:
        """往 console 追加一行."""
        if hasattr(self, "_console") and self._console is not None:
            self._console.append(text)
            self._console.moveCursor(QTextCursor.End)

    def set_status(self, text: str, state: str = "idle") -> None:
        """更新右上角状态点 + 文字. state ∈ idle/working/done/error."""
        self._status_lbl.setText(text)
        role_map = {
            "idle":    "status_info",
            "working": "status_warn",
            "done":    "status_success",
            "error":   "status_error",
        }
        self._status_lbl.setProperty("role", role_map.get(state, "status_info"))
        # force style refresh after role change
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        self._status_dot.set_state(state)

    # ---------------------------------------------------------
    # 持久化 (子类不感知)
    # ---------------------------------------------------------
    def _load_tool_config(self) -> None:
        if not self.TOOL_NAME:
            return
        try:
            from core.tool_config import ToolConfig
            cfg = ToolConfig().get_tool_config(self.TOOL_NAME)
            self.load_tool_config(cfg)
        except Exception as e:
            print(f"[BaseToolWindow] load_tool_config failed: {e}")

    def _save_tool_config(self) -> None:
        if not self.TOOL_NAME:
            return
        try:
            from core.tool_config import ToolConfig
            tc = ToolConfig()
            data = self.dump_tool_config()
            for k, v in (data or {}).items():
                tc.set(self.TOOL_NAME, k, v)
            tc.save()
        except Exception as e:
            print(f"[BaseToolWindow] save_tool_config failed: {e}")

    def load_tool_config(self, cfg: dict) -> None:
        """子类覆写. cfg 是 dict, 来自 tool_config.yaml."""
        pass

    def dump_tool_config(self) -> dict:
        """子类覆写. 返回要保存的 dict."""
        return {}

    def closeEvent(self, event):
        self._save_tool_config()
        if self.TOOL_NAME:
            event_bus().tool_window_closed.emit(self.TOOL_NAME)
        super().closeEvent(event)
