"""
ui/dialog_utils.py — Dialog 通用工具: 标题条 + 内容布局.

所有 6 个 dialog 共用:
- 带图标 + 双语标题的标题条
- 标准 16px 边距的 vertical layout
- 底部 OK / Cancel 按钮行
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .icons import Icon, Icons, C


def build_dialog_header(parent: QDialog, icon_name: str, title: str, subtitle: str = "") -> QWidget:
    """生成 dialog 顶部的标题条 (图标 + 标题 + 副标题).

    返回 QWidget, 调用方 addWidget 到自己的 layout.
    """
    wrapper = QWidget(parent)
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(16, 12, 16, 8)
    layout.setSpacing(10)

    icon_lbl = QLabel()
    icon_lbl.setPixmap(Icon.pixmap(icon_name, size=22, color=C.AMBER))
    icon_lbl.setFixedWidth(28)
    layout.addWidget(icon_lbl, 0, Qt.AlignTop)

    text_col = QVBoxLayout()
    text_col.setContentsMargins(0, 0, 0, 0)
    text_col.setSpacing(0)

    title_lbl = QLabel(title)
    f = QFont()
    f.setBold(True)
    f.setPointSize(12)
    title_lbl.setFont(f)
    title_lbl.setProperty("role", "dialog_title")
    text_col.addWidget(title_lbl)

    if subtitle:
        sub_lbl = QLabel(subtitle)
        sub_lbl.setProperty("role", "dialog_subtitle")
        text_col.addWidget(sub_lbl)

    layout.addLayout(text_col, 1)

    # 右侧装饰: 一个小 accent 圆点 (RM 战队色)
    accent_dot = QLabel()
    accent_dot.setFixedSize(8, 8)
    accent_dot.setProperty("role", "accent_dot")
    layout.addWidget(accent_dot, 0, Qt.AlignTop | Qt.AlignRight)

    # 底部 1px 分隔线
    return wrapper


def build_dialog_footer(parent: QDialog, on_ok=None, on_cancel=None,
                        ok_text="确定", ok_role="primary",
                        cancel_text="取消") -> QWidget:
    """生成 dialog 底部的按钮行."""
    wrapper = QWidget(parent)
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(16, 8, 16, 14)
    layout.setSpacing(8)

    layout.addStretch()

    cancel_btn = QPushButton(cancel_text)
    cancel_btn.setCursor(Qt.PointingHandCursor)
    cancel_btn.setProperty("role", "ghost")
    cancel_btn.clicked.connect(parent.reject if on_cancel is None else on_cancel)
    layout.addWidget(cancel_btn)

    ok_btn = QPushButton(ok_text)
    ok_btn.setCursor(Qt.PointingHandCursor)
    ok_btn.setProperty("role", ok_role)
    ok_btn.setDefault(True)
    ok_btn.setAutoDefault(True)
    ok_btn.clicked.connect(parent.accept if on_ok is None else on_ok)
    layout.addWidget(ok_btn)

    return wrapper


def setup_dialog_base(dialog: QDialog, icon_name: str, title: str, subtitle: str = ""):
    """在 dialog 上设置窗口属性 + 标准 minimum size."""
    dialog.setWindowTitle(title)
    dialog.setWindowIcon(Icon.icon(icon_name, size=24, color=C.AMBER))
    dialog.setMinimumSize(520, 480)
    return dialog
