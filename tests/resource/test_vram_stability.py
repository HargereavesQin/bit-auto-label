# -*- coding: utf-8 -*-
"""L3 显存回归（审计 T1）：SAM3 跨样本 state 不得累积。

复刻 sam_client 的单槽持有策略：set_image → 换引用 → del 旧 state，
连切 N 张后 torch.cuda.memory_allocated 增量必须平稳。
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")  # conftest 已设 offscreen；本测试不建窗口但保持一致
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CKPT = Path(__file__).resolve().parent.parent.parent / "models" / "sam3.pt"

pytestmark = [
    pytest.mark.vram,
    pytest.mark.resource,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 设备"),
    pytest.mark.skipif(not CKPT.exists(), reason=f"缺少 checkpoint: {CKPT}"),
]


def _find_bpe() -> Path:
    sam3_mod = __import__("sam3")
    roots = [Path(p) for p in getattr(sam3_mod, "__path__", [])]
    roots.append(Path(__file__).resolve().parent.parent.parent / "sam3")  # 仓库 vendored 目录兜底
    candidates = []
    for root in roots:
        candidates.extend(root.rglob("bpe_simple_vocab_16e6.txt.gz"))
    assert candidates, f"sam3 包内未找到 BPE 词表: {roots}"
    return candidates[0]


def _synthetic_pil(w, h, seed):
    import cv2
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def test_sam3_set_image_vram_stable():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda"
    model = build_sam3_image_model(
        bpe_path=str(_find_bpe()),
        checkpoint_path=str(CKPT),
        enable_inst_interactivity=True,
    )
    model.to(device)
    processor = Sam3Processor(model)

    sizes = [(1280, 720), (1920, 1080), (1600, 900), (1280, 960)] * 3  # 12 张混合分辨率

    state = None
    torch.cuda.synchronize()
    allocated = []

    for i, (w, h) in enumerate(sizes):
        pil = _synthetic_pil(w, h, seed=i)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            new_state = processor.set_image(pil)
        old = state
        state = new_state
        if old is not None:
            del old  # 复刻 sam_client._on_image_load_finished 的单槽替换
        torch.cuda.synchronize()
        if i >= 2:  # 跳过前两张（词表/首帧 JIT 抖动）
            allocated.append(torch.cuda.memory_allocated() / (1024 * 1024))

    print(f"\ncuda memory_allocated 序列(MB): {[f'{a:.0f}' for a in allocated]}")
    growth = allocated[-1] - allocated[0]
    assert growth < 512, f"连切 {len(sizes)} 张后显存净增长 {growth:.0f}MB —— 疑似 state 跨样本累积"

    del state, model, processor
    torch.cuda.empty_cache()
