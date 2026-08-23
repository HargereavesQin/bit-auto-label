# -*- coding: utf-8 -*-
"""torch.load 安全封装。

默认 weights_only=True（只允许张量/基本类型，阻断 pickle RCE）。
仅当用户显式设置环境变量 ARMORLABEL_UNSAFE_LOAD=1 时，
才回退到 weights_only=False 以加载旧式整模块 pickle 文件。
"""
import os


def torch_load_safe(path, map_location="cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as safe_err:
        if os.environ.get("ARMORLABEL_UNSAFE_LOAD") == "1":
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"拒绝加载 {path}: 非受信 checkpoint（weights_only 模式解析失败: {safe_err}）。"
            "如确信文件来源可靠且为旧式整模块 pickle，可设置环境变量 ARMORLABEL_UNSAFE_LOAD=1 后重试。"
        ) from safe_err
