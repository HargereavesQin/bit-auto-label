"""
ui/icons.py — SVG 图标库 (RM 竞技风, 1.5px 描边线框)

所有图标都是 24x24 viewBox 的单色线性图标, 通过 QSvgRenderer
渲染到 QPixmap 上, 再包成 QIcon. 颜色由 setColor() 控制.

用法:
    icon = Icon.rect()                     # 默认色
    icon2 = Icon.rect(color="#C9A36B")     # 香槟金
    btn.setIcon(icon)
    btn.setIconSize(QSize(20, 20))
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

_DEFAULT_COLOR = "#E8E8EA"


# -------------------------------------------------------------
# SVG 源 (Feather/Lucide 风格, 1.5 描边, 24x24)
# 每个常量是一个完整的 SVG 字符串, "{color}" 会被替换成具体颜色.
# -------------------------------------------------------------

_SVG: dict[str, str] = {
    "rect": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="5" width="18" height="14" rx="1.5"/>
    </svg>''',

    "polygon": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3 L21 9 L18 20 L6 20 L3 9 Z"/>
        <circle cx="12" cy="3" r="0.8" fill="{color}"/>
        <circle cx="21" cy="9" r="0.8" fill="{color}"/>
        <circle cx="18" cy="20" r="0.8" fill="{color}"/>
        <circle cx="6" cy="20" r="0.8" fill="{color}"/>
        <circle cx="3" cy="9" r="0.8" fill="{color}"/>
    </svg>''',

    "point": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3" fill="{color}"/>
        <line x1="12" y1="2" x2="12" y2="5"/>
        <line x1="12" y1="19" x2="12" y2="22"/>
        <line x1="2" y1="12" x2="5" y2="12"/>
        <line x1="19" y1="12" x2="22" y2="12"/>
    </svg>''',

    "obb": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 8 L16 4 L20 16 L9 20 Z"/>
        <path d="M3 12 L12 9" stroke-dasharray="2 2"/>
        <circle cx="5" cy="8" r="1.2" fill="{color}"/>
    </svg>''',

    "pose": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="5" width="18" height="14" rx="1"/>
        <circle cx="7" cy="9" r="1.3" fill="{color}"/>
        <circle cx="17" cy="9" r="1.3" fill="{color}"/>
        <circle cx="7" cy="15" r="1.3" fill="{color}"/>
        <circle cx="17" cy="15" r="1.3" fill="{color}"/>
        <line x1="7" y1="9" x2="17" y2="9" stroke-dasharray="1.5 1.5"/>
        <line x1="7" y1="15" x2="17" y2="15" stroke-dasharray="1.5 1.5"/>
    </svg>''',

    "folder": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 7 a2 2 0 0 1 2 -2 h4 l2 2 h8 a2 2 0 0 1 2 2 v8 a2 2 0 0 1 -2 2 H5 a2 2 0 0 1 -2 -2 Z"/>
    </svg>''',

    "gear": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/>
        <path d="M12 1 L12 4 M12 20 L12 23 M4.2 4.2 L6.3 6.3 M17.7 17.7 L19.8 19.8
                 M1 12 L4 12 M20 12 L23 12 M4.2 19.8 L6.3 17.7 M17.7 6.3 L19.8 4.2"/>
    </svg>''',

    "help": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="9"/>
        <path d="M9.1 9 a3 3 0 0 1 5.8 1 c0 2 -3 3 -3 3"/>
        <line x1="12" y1="17" x2="12" y2="17.01"/>
    </svg>''',

    "plus": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="5" x2="12" y2="19"/>
        <line x1="5" y1="12" x2="19" y2="12"/>
    </svg>''',

    "trash": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 6 L21 6 M8 6 L8 4 a2 2 0 0 1 2 -2 h4 a2 2 0 0 1 2 2 L16 6
                 M5 6 L6 20 a2 2 0 0 0 2 2 h8 a2 2 0 0 0 2 -2 L19 6"/>
        <line x1="10" y1="11" x2="10" y2="17"/>
        <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>''',

    "bolt": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M13 2 L4 14 L12 14 L11 22 L20 10 L12 10 Z" fill="{color}" fill-opacity="0.12"/>
    </svg>''',

    "bulb": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 18 L15 18 M10 22 L14 22 M12 2 a7 7 0 0 1 4 12.7 L16 18 L8 18
                 L8 14.7 A7 7 0 0 1 12 2 Z"/>
    </svg>''',

    "target": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="9"/>
        <circle cx="12" cy="12" r="5"/>
        <circle cx="12" cy="12" r="1.5" fill="{color}"/>
    </svg>''',

    "chart": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <line x1="3" y1="20" x2="21" y2="20"/>
        <rect x="6" y="11" width="3" height="9"/>
        <rect x="11" y="6" width="3" height="14"/>
        <rect x="16" y="14" width="3" height="6"/>
    </svg>''',

    "broom": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M19 4 L14 9 M9 14 L4 19 M14 4 L20 10 L18 12 L12 6 Z"/>
        <path d="M5 21 L9 17 M7 19 L11 15 M9 21 L13 17"/>
    </svg>''',

    "merge": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M8 3 L8 8 a4 4 0 0 0 4 4 a4 4 0 0 1 4 4 L16 21"/>
        <path d="M3 8 L8 3 L13 8"/>
        <path d="M11 16 L16 21 L21 16"/>
    </svg>''',

    "rename": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 20 L20 20 M16 4 L20 4 L20 8 M4 20 L11 13 L13 15 L6 22 Z"/>
        <line x1="13" y1="6" x2="18" y2="11"/>
    </svg>''',

    "video": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="6" width="13" height="12" rx="1.5"/>
        <path d="M16 10 L21 7 L21 17 L16 14 Z"/>
    </svg>''',

    "dataset": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <ellipse cx="12" cy="5" rx="8" ry="3"/>
        <path d="M4 5 L4 19 a8 3 0 0 0 16 0 L20 5"/>
        <path d="M4 12 a8 3 0 0 0 16 0"/>
    </svg>''',

    "fusion": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <circle cx="8" cy="8" r="4"/>
        <circle cx="16" cy="16" r="4"/>
        <path d="M11 11 L13 13 M8 12 L12 12 M12 8 L12 12"
            stroke-dasharray="2 2"/>
    </svg>''',

    "auto": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2 L15 9 L22 9 L16 13 L18 21 L12 17 L6 21 L8 13 L2 9 L9 9 Z"/>
    </svg>''',

    "train": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="5" y="3" width="14" height="14" rx="3"/>
        <line x1="5" y1="10" x2="19" y2="10"/>
        <circle cx="9" cy="14" r="1" fill="{color}"/>
        <circle cx="15" cy="14" r="1" fill="{color}"/>
        <path d="M8 17 L7 21 M16 17 L17 21"/>
    </svg>''',

    "play": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <polygon points="6,4 20,12 6,20" fill="{color}" fill-opacity="0.15"/>
    </svg>''',

    "neural": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <circle cx="6" cy="6" r="2"/>
        <circle cx="18" cy="6" r="2"/>
        <circle cx="6" cy="18" r="2"/>
        <circle cx="18" cy="18" r="2"/>
        <circle cx="12" cy="12" r="2.5" fill="{color}" fill-opacity="0.2"/>
        <line x1="7.5" y1="7.5" x2="10.5" y2="10.5"/>
        <line x1="16.5" y1="7.5" x2="13.5" y2="10.5"/>
        <line x1="7.5" y1="16.5" x2="10.5" y2="13.5"/>
        <line x1="16.5" y1="16.5" x2="13.5" y2="13.5"/>
    </svg>''',

    "armor": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3 L20 6 L20 13 a9 9 0 0 1 -8 8 a9 9 0 0 1 -8 -8 L4 6 Z"/>
        <path d="M9 12 L11 12 L11 16 L13 16 L13 12 L15 12"
            stroke="{color}" stroke-width="1.5"/>
    </svg>''',

    "scan": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 7 L3 4 L6 4 M18 4 L21 4 L21 7 M21 17 L21 20 L18 20 M6 20 L3 20 L3 17"/>
        <line x1="3" y1="12" x2="21" y2="12" stroke-dasharray="3 2"/>
    </svg>''',

    "settings_sliders": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <line x1="4" y1="6" x2="20" y2="6"/>
        <line x1="4" y1="12" x2="20" y2="12"/>
        <line x1="4" y1="18" x2="20" y2="18"/>
        <circle cx="9" cy="6" r="2" fill="{color}"/>
        <circle cx="15" cy="12" r="2" fill="{color}"/>
        <circle cx="11" cy="18" r="2" fill="{color}"/>
    </svg>''',

    "format": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 5 a2 2 0 0 1 2 -2 h8 l6 6 v10 a2 2 0 0 1 -2 2 H6 a2 2 0 0 1 -2 -2 Z"/>
        <path d="M14 3 L14 9 L20 9"/>
        <text x="7" y="17" font-family="monospace" font-size="6" fill="{color}">.json</text>
    </svg>''',

    "crosshair": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="9"/>
        <line x1="12" y1="2" x2="12" y2="6"/>
        <line x1="12" y1="18" x2="12" y2="22"/>
        <line x1="2" y1="12" x2="6" y2="12"/>
        <line x1="18" y1="12" x2="22" y2="12"/>
        <circle cx="12" cy="12" r="1.5" fill="{color}"/>
    </svg>''',

    "spark": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3 L13.5 9.5 L20 11 L13.5 12.5 L12 19 L10.5 12.5 L4 11 L10.5 9.5 Z"
            fill="{color}" fill-opacity="0.15"/>
        <path d="M19 4 L19.7 6 L21.5 6.5 L19.7 7 L19 9 L18.3 7 L16.5 6.5 L18.3 6 Z"
            fill="{color}"/>
    </svg>''',

    "list": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <line x1="8" y1="6" x2="20" y2="6"/>
        <line x1="8" y1="12" x2="20" y2="12"/>
        <line x1="8" y1="18" x2="20" y2="18"/>
        <circle cx="4" cy="6" r="1" fill="{color}"/>
        <circle cx="4" cy="12" r="1" fill="{color}"/>
        <circle cx="4" cy="18" r="1" fill="{color}"/>
    </svg>''',

    "instances": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="8" height="8" rx="1"/>
        <rect x="13" y="3" width="8" height="8" rx="1"/>
        <rect x="3" y="13" width="8" height="8" rx="1"/>
        <rect x="13" y="13" width="8" height="8" rx="1"/>
    </svg>''',

    "check": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round">
        <polyline points="4,12 10,18 20,6"/>
    </svg>''',

    "x": '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
        viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5"
        stroke-linecap="round" stroke-linejoin="round">
        <line x1="6" y1="6" x2="18" y2="18"/>
        <line x1="18" y1="6" x2="6" y2="18"/>
    </svg>''',
}


# -------------------------------------------------------------
# Icon 类: 提供 QIcon / QPixmap 工厂
# -------------------------------------------------------------

class Icon:
    """SVG 图标工厂. 静态方法返回 QIcon / QPixmap."""

    _cache: dict[tuple[str, int, str], QPixmap] = {}

    @staticmethod
    def pixmap(name: str, size: int = 20, color: str = _DEFAULT_COLOR) -> QPixmap:
        """获取指定大小和颜色的 QPixmap (带缓存)."""
        key = (name, size, color)
        cached = Icon._cache.get(key)
        if cached is not None:
            return cached

        if name not in _SVG:
            raise KeyError(f"unknown icon: {name}")

        svg = _SVG[name].format(color=color)
        renderer = QSvgRenderer(svg.encode("utf-8"))

        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter)
        painter.end()

        Icon._cache[key] = pm
        return pm

    @staticmethod
    def icon(name: str, size: int = 20, color: str = _DEFAULT_COLOR) -> QIcon:
        """获取 QIcon (用于 QPushButton.setIcon / QAction.setIcon)."""
        pm = Icon.pixmap(name, size, color)
        return QIcon(pm)


# -------------------------------------------------------------
# 预设语义角色
# -------------------------------------------------------------

class Icons:
    """常用图标速查表, 避免到处写字符串."""

    RECT     = "rect"
    POLY     = "polygon"
    POINT    = "point"
    OBB      = "obb"
    POSE     = "pose"

    FOLDER   = "folder"
    GEAR     = "gear"
    HELP     = "help"
    PLUS     = "plus"
    TRASH    = "trash"
    CHECK    = "check"
    X        = "x"

    BOLT     = "bolt"        # SAM 智能辅助
    BULB     = "bulb"        # 光条外接
    TARGET   = "target"
    SPARK    = "spark"       # 自动标注提示

    CHART    = "chart"
    BROOM    = "broom"
    MERGE    = "merge"
    RENAME   = "rename"
    VIDEO    = "video"
    DATASET  = "dataset"
    FUSION   = "fusion"
    AUTO     = "auto"
    TRAIN    = "train"
    PLAY     = "play"
    NEURAL   = "neural"      # MLP
    ARMOR    = "armor"       # 装甲板
    SCAN     = "scan"        # 推理
    SLIDERS  = "settings_sliders"
    FORMAT   = "format"
    CROSSHAIR = "crosshair"
    LIST     = "list"
    INSTANCES = "instances"


# RM 主题色 (与 style.qss 同步)
class C:
    """主题色常量.

    Editorial Warm Dark — 暖灰底 + 单一香槟金点缀.
    语义:   香槟金 = 主要操作/强调
            雾绿   = 成功/完成
            暮霞   = 危险/错误
            暗灰   = 静默/分隔
    """
    # 中性灰 (背景/文本) — 编辑风暖灰
    BG_APP        = "#0F0F12"
    BG_PANEL      = "#15151A"
    BG_ELEVATED   = "#1A1A20"
    BG_INPUT      = "#0A0A0D"

    BORDER        = "#26262D"
    BORDER_STRONG = "#3A3A42"
    BORDER_SUBTLE = "#1A1A20"

    TEXT          = "#ECECF0"
    TEXT_MUTED    = "#8C8C95"
    TEXT_FAINT    = "#5A5A62"
    TEXT_DISABLED = "#3A3A42"

    # 主色 — 香槟金 (替换原琥珀,更克制典雅)
    AMBER         = "#C9A36B"
    AMBER_BRIGHT  = "#DCBA80"
    AMBER_DIM     = "#9C7E50"

    # 状态色 — 单一香槟金体系下的语义
    MUTED         = "#5A5A62"   # 暗灰, 用作斜线分割的右端点
    SUCCESS       = "#8FA68F"   # 雾绿
    DANGER        = "#B5756A"   # 暮霞


# 便利函数
def set_btn_icon(btn, icon_name: str, size: int = 16, color: str = None):
    """给按钮设置图标 + 调整图标尺寸."""
    color = color or C.TEXT
    btn.setIcon(Icon.icon(icon_name, size=size, color=color))
    btn.setIconSize(QSize(size, size))


def set_action_icon(action, icon_name: str, size: int = 16, color: str = None):
    """给 QAction 设置图标."""
    color = color or C.TEXT
    action.setIcon(Icon.icon(icon_name, size=size, color=color))
