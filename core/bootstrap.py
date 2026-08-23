# -*- coding: utf-8 -*-
"""应用启动引导：在 import 第三方 ML 库之前抑制已知无害警告。"""
import warnings


def suppress_third_party_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Importing from timm.models.layers is deprecated",
        category=FutureWarning,
    )
