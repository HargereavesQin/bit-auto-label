# -*- coding: utf-8 -*-
"""
YOLO 运行时版本切换 — YOLO10/11/12 (仓库内嵌 8.3) / YOLO26 (pip 8.4+).

仓库根目录存在 ultralytics/ 时，默认会遮蔽 pip 包；本模块按设置调整 sys.path 并清理 import 缓存。
"""
from __future__ import annotations

import logging
import os
import shutil
import site
import sys
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QSettings

YOLO10 = "yolo10"
YOLO11 = "yolo11"
YOLO12 = "yolo12"
YOLO26 = "yolo26"

VENDORED_RUNTIMES = frozenset({YOLO10, YOLO11, YOLO12})
PIP_RUNTIMES = frozenset({YOLO26})
ALL_RUNTIMES = (YOLO10, YOLO11, YOLO12, YOLO26)

SETTINGS_KEY = "YOLO_RUNTIME"
DEFAULT_RUNTIME = YOLO12

_RUNTIME_LABELS = {
    YOLO10: "YOLO10 · 内嵌 Ultralytics 8.3",
    YOLO11: "YOLO11 · 内嵌 Ultralytics 8.3",
    YOLO12: "YOLO12 · 内嵌 Ultralytics 8.3",
    YOLO26: "YOLO26 · pip Ultralytics 8.4+",
}


class RuntimeInfo(NamedTuple):
    runtime: str
    ultralytics_version: str
    ultralytics_path: str


_APPLIED_BACKEND: str | None = None  # "vendored" | "pip"

_REQUIRED_ASSETS = ("bus.jpg", "zidane.jpg")


def get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def uses_vendored(runtime: str | None) -> bool:
    return normalize_runtime(runtime) in VENDORED_RUNTIMES


def _runtime_backend(runtime: str) -> str:
    return "vendored" if uses_vendored(runtime) else "pip"


def _clear_ultralytics_log_handlers() -> None:
    """ultralytics 重载时 set_logging 会重复 addHandler，训练日志会成倍刷屏。"""
    for name, logger_obj in list(logging.root.manager.loggerDict.items()):
        if name != "ultralytics" and not str(name).startswith("ultralytics."):
            continue
        logger = logger_obj if isinstance(logger_obj, logging.Logger) else logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def normalize_runtime(value: str | None) -> str:
    if value in ALL_RUNTIMES:
        return value
    return DEFAULT_RUNTIME


def runtime_label(runtime: str | None = None) -> str:
    return _RUNTIME_LABELS.get(normalize_runtime(runtime), _RUNTIME_LABELS[YOLO12])


def get_runtime() -> str:
    settings = QSettings("ArmorLabel", "settings")
    settings.beginGroup("Parameters")
    value = settings.value(SETTINGS_KEY, DEFAULT_RUNTIME)
    settings.endGroup()
    return normalize_runtime(str(value) if value is not None else DEFAULT_RUNTIME)


def set_runtime(runtime: str) -> str:
    runtime = normalize_runtime(runtime)
    settings = QSettings("ArmorLabel", "settings")
    settings.beginGroup("Parameters")
    settings.setValue(SETTINGS_KEY, runtime)
    settings.endGroup()
    return runtime


def _purge_ultralytics_modules() -> None:
    _clear_ultralytics_log_handlers()
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            del sys.modules[name]


def ensure_ultralytics_assets() -> None:
    """补全 vendored ultralytics/assets（AMP 自检需要 bus.jpg 等）。"""
    from ultralytics.utils import ASSETS, ASSETS_URL

    assets_dir = Path(ASSETS)
    assets_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in _REQUIRED_ASSETS if not (assets_dir / n).is_file()]
    if not missing:
        return

    for sp in site.getsitepackages():
        src_dir = Path(sp) / "ultralytics" / "assets"
        if not src_dir.is_dir():
            continue
        for name in missing:
            src = src_dir / name
            if src.is_file():
                shutil.copy2(src, assets_dir / name)
        missing = [n for n in missing if not (assets_dir / n).is_file()]
        if not missing:
            return

    from ultralytics.utils.downloads import safe_download

    for name in missing:
        safe_download(
            f"{ASSETS_URL}/{name}",
            dir=str(assets_dir),
            file=name,
            unzip=False,
            exist_ok=True,
        )

    still_missing = [n for n in _REQUIRED_ASSETS if not (assets_dir / n).is_file()]
    if still_missing:
        raise RuntimeError(
            f"Ultralytics 资源文件缺失: {still_missing}。"
            "请检查网络或 conda 环境中的 ultralytics 包，"
            "或在训练设置中关闭 AMP（amp=False）。"
        )


def _paths_to_block_for_pip(repo_root: str) -> set[str]:
    blocked = {os.path.normpath(repo_root)}
    cwd = os.path.normpath(os.getcwd())
    if cwd == blocked:
        blocked.add(cwd)
    return blocked


def apply_runtime(runtime: str | None = None) -> RuntimeInfo:
    """配置 import 路径并返回当前 ultralytics 信息。"""
    global _APPLIED_BACKEND
    runtime = normalize_runtime(runtime or get_runtime())
    backend = _runtime_backend(runtime)

    if _APPLIED_BACKEND == backend and "ultralytics" in sys.modules:
        import ultralytics  # noqa: WPS433
        ensure_ultralytics_assets()
        return RuntimeInfo(runtime, ultralytics.__version__, ultralytics.__file__)

    repo_root = os.path.normpath(get_repo_root())
    _purge_ultralytics_modules()

    if uses_vendored(runtime):
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        else:
            sys.path.remove(repo_root)
            sys.path.insert(0, repo_root)
    else:
        blocked = _paths_to_block_for_pip(repo_root)
        cleaned: list[str] = []
        for entry in sys.path:
            if not entry:
                if os.path.normpath(os.getcwd()) in blocked:
                    continue
                cleaned.append(entry)
                continue
            try:
                norm = os.path.normpath(os.path.abspath(entry))
            except OSError:
                cleaned.append(entry)
                continue
            if norm in blocked:
                continue
            cleaned.append(entry)
        sys.path = cleaned
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.append(sp)

    import ultralytics  # noqa: WPS433 — intentional lazy import after path fix

    ensure_ultralytics_assets()
    _APPLIED_BACKEND = backend
    return RuntimeInfo(runtime, ultralytics.__version__, ultralytics.__file__)


def import_yolo_class():
    """返回 ultralytics.YOLO 类（须先调用 apply_runtime）。"""
    from ultralytics import YOLO

    return YOLO


def load_yolo_model(model_path: str, task: str | None = None):
    """加载 YOLO 模型（自动应用运行时，支持从路径识别 ONNX/Engine 任务类型）。"""
    apply_runtime()
    from ultralytics import YOLO

    if not task and isinstance(model_path, str):
        path_lower = model_path.lower()
        if "pose" in path_lower:
            task = "pose"
        elif "seg" in path_lower:
            task = "segment"
        elif "obb" in path_lower:
            task = "obb"

    if task:
        try:
            return YOLO(model_path, task=task)
        except Exception:
            pass
    return YOLO(model_path)


def preset_weight_name(
    task: str = "pose", size: str = "n", runtime: str | None = None,
) -> str:
    """当前运行时下常用预训练权重名（仅名称，不保证已下载）。"""
    runtime = normalize_runtime(runtime or get_runtime())
    task = (task or "detect").lower()
    size = size or "n"

    if runtime == YOLO10:
        # v10 仅官方提供检测权重
        return f"yolov10{size}.pt"

    if runtime == YOLO11:
        if task == "pose":
            return f"yolo11{size}-pose.pt"
        if task == "segment":
            return f"yolo11{size}-seg.pt"
        if task == "obb":
            return f"yolo11{size}-obb.pt"
        return f"yolo11{size}.pt"

    if runtime == YOLO26:
        if task == "pose":
            return f"yolo26{size}-pose.pt"
        if task == "segment":
            return f"yolo26{size}-seg.pt"
        if task == "obb":
            return f"yolo26{size}-obb.pt"
        return f"yolo26{size}.pt"

    # YOLO12
    if task == "pose":
        return f"yolov12{size}-pose.pt"
    if task == "segment":
        return f"yolov12{size}-seg.pt"
    if task == "obb":
        return f"yolov12{size}-obb.pt"
    return f"yolov12{size}.pt"
