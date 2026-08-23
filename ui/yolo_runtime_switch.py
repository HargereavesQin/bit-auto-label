# -*- coding: utf-8 -*-
"""YOLO10 / YOLO11 / YOLO12 / YOLO26 运行时切换控件。"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QButtonGroup, QMessageBox,
)
from PySide6.QtCore import Signal, Qt

from core.yolo_runtime import (
    ALL_RUNTIMES, YOLO26, get_runtime, set_runtime, apply_runtime,
    runtime_label, uses_vendored,
)


class YoloRuntimeSwitch(QWidget):
    """多按钮切换 YOLO 运行时，写入全局 QSettings。"""

    runtime_changed = Signal(str, str)  # new_runtime, previous_runtime

    def __init__(self, parent=None, *, compact: bool = False):
        super().__init__(parent)
        self._compact = compact
        self._runtime = get_runtime()
        self._buttons: dict[str, QPushButton] = {}
        self._setup_ui()
        self._sync_buttons()
        self._refresh_info()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if not self._compact:
            lbl = QLabel("Ultralytics")
            lbl.setProperty("role", "form_label")
            layout.addWidget(lbl)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for runtime in ALL_RUNTIMES:
            label = runtime.upper()
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("role", "tag")
            btn.clicked.connect(lambda checked=False, r=runtime: self._on_pick(r))
            self._buttons[runtime] = btn
            self._group.addButton(btn)
            layout.addWidget(btn)

        self.info_lbl = QLabel("")
        self.info_lbl.setProperty("role", "caption")
        self.info_lbl.setWordWrap(True)
        layout.addWidget(self.info_lbl, 1)

    def current_runtime(self) -> str:
        return self._runtime

    def sync_from_settings(self):
        self._runtime = get_runtime()
        self._sync_buttons()
        self._refresh_info()

    def _sync_buttons(self):
        for runtime, btn in self._buttons.items():
            btn.setChecked(runtime == self._runtime)

    def _refresh_info(self):
        try:
            info = apply_runtime(self._runtime)
            self.info_lbl.setText(
                f"{runtime_label(self._runtime)}  →  v{info.ultralytics_version}"
            )
        except Exception as exc:
            self.info_lbl.setText(f"{runtime_label(self._runtime)}  →  不可用: {exc}")

    def _confirm_runtime_switch(self, runtime: str) -> bool:
        was_vendored = uses_vendored(self._runtime)
        will_vendored = uses_vendored(runtime)
        if was_vendored == will_vendored:
            return True
        if runtime == YOLO26:
            msg = (
                "YOLO26 使用 pip 版 Ultralytics 8.4+，与内嵌 YOLO10/11/12 互不干扰。\n"
                "切换后需重新加载/选择模型权重。\n\n是否继续？"
            )
            title = "切换 YOLO26"
        else:
            msg = (
                f"{runtime_label(runtime)} 使用仓库内嵌 Ultralytics 8.3。\n"
                "切换后需重新加载/选择模型权重。\n\n是否继续？"
            )
            title = f"切换 {runtime.upper()}"
        reply = QMessageBox.question(
            self, title, msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def _on_pick(self, runtime: str):
        if runtime == self._runtime:
            self._sync_buttons()
            return

        previous = self._runtime
        if not self._confirm_runtime_switch(runtime):
            self._sync_buttons()
            return

        self._runtime = set_runtime(runtime)
        self._sync_buttons()
        self._refresh_info()
        self.runtime_changed.emit(self._runtime, previous)
