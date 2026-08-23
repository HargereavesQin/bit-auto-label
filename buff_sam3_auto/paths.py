"""包内默认资源路径。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def package_root() -> Path:
    return Path(__file__).resolve().parent


def weights_dir() -> Path:
    return package_root() / "weights"


def default_onnx_path() -> Path:
    """包内自带的 YOLO bbox ONNX（yolo11n_buff）。"""
    return weights_dir() / "yolo11n_buff.onnx"


def default_bpe_candidates(extra: Optional[str] = None) -> List[str]:
    """按优先级列出可能的 SAM3 BPE 词表路径。"""
    root = package_root().parent
    candidates: List[str] = []
    if extra and str(extra).strip():
        candidates.append(str(Path(extra).expanduser()))
    candidates.extend(
        [
            str(weights_dir() / "bpe_simple_vocab_16e6.txt.gz"),
            str(
                root
                / "sam3"
                / "sam3"
                / "assets"
                / "bpe_simple_vocab_16e6.txt.gz"
            ),
            str(
                root
                / "armor-label"
                / "sam3"
                / "sam3"
                / "assets"
                / "bpe_simple_vocab_16e6.txt.gz"
            ),
        ]
    )
    try:
        import sam3

        sam3_root = os.path.dirname(os.path.abspath(sam3.__file__))
        for dirpath, _, files in os.walk(sam3_root):
            if "bpe_simple_vocab_16e6.txt.gz" in files:
                candidates.insert(
                    0, os.path.join(dirpath, "bpe_simple_vocab_16e6.txt.gz")
                )
                break
    except Exception:
        pass
    return candidates
