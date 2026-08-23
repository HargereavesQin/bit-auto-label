"""
core/event_bus.py — 全局信号总线.

设计动机:
  现状是 MainWindow 是 god object, 所有跨模块信号都 connect 到它.
  加新功能基本都得改 main.py 的某处.

  引入 EventBus 后, 任何模块都可以:
    from core.event_bus import event_bus
    event_bus.image_loaded.connect(my_handler)
    event_bus.image_loaded.emit(path)
  这样新功能不再需要碰 main.py.

单例 (lazy) — 必须由 QApplication 已创建之后再调用, 因为 QObject 需要一个 thread.
"""
from __future__ import annotations
from PySide6.QtCore import QObject, Signal


class EventBus(QObject):
    """全项目唯一信号总线. 实例由 event_bus() 工厂返回."""

    # ---- 图片相关 ----
    image_opened = Signal(str)           # path
    image_loaded = Signal(str)           # path (image is decoded and shown)
    image_saved = Signal(str)            # path
    image_dir_changed = Signal(str)      # dir path
    image_about_to_change = Signal()     # pre-switch (暂停 SAM / 清理临时态)

    # ---- 形状 (Shape) 相关 ----
    shape_added = Signal(object)
    shape_removed = Signal(object)
    shape_modified = Signal(object)
    shapes_cleared = Signal()
    current_shape_changed = Signal(object)  # selection

    # ---- 标注模式 ----
    canvas_mode_changed = Signal(str)    # "rect" / "poly" / ...
    auto_mode_changed = Signal(str)      # "armor" / "buff-merge"

    # ---- SAM / 模型状态 ----
    model_loading = Signal(str)          # state: "loading" / "ready" / "error"
    model_loaded = Signal(str)           # path
    sam_status = Signal(str, str)        # state, message
    sam_point_prompt = Signal(float, float, str)  # x, y, label ("fg"/"bg")
    sam_text_prompt = Signal(str)        # text

    # ---- 工具窗口 ----
    tool_window_opened = Signal(str)
    tool_window_closed = Signal(str)

    # ---- 状态栏 / 日志 ----
    status_message = Signal(str, str)    # text, level ("info"/"warning"/"error"/"success"/"active")
    log_message = Signal(str)            # 自由日志行

    # ---- 批量标注 ----
    batch_started = Signal(int)          # total
    batch_progress = Signal(int, int)    # current, total
    batch_finished = Signal(int)         # processed

    # ---- 类别 / Pose 配置 ----
    class_list_changed = Signal(list)    # [name, ...]
    pose_category_changed = Signal(str)  # name
    pose_category_added = Signal(str)
    pose_category_removed = Signal(str)

    # ---- 配置变化 ----
    settings_changed = Signal(str)       # key


_bus: EventBus | None = None


def event_bus() -> EventBus:
    """返回全局 EventBus 单例. 第一次调用时构造."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
