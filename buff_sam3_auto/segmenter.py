"""SAM3 文本提示分割封装。

对齐 armor-label 调用链：
build_sam3_image_model -> Sam3Processor.set_image -> set_text_prompt。
"""

from __future__ import annotations

import os
import sys
import zipfile
from contextlib import nullcontext
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .paths import default_bpe_candidates, package_root

# 项目根下的 sam3 源码目录可能被识别成 namespace package；
# 显式把仓库根加入最前面，确保导入其内部真正的 sam3 包。
_ROOT = str(package_root().parent)
_SAM3_REPO = os.path.join(_ROOT, "sam3")
if os.path.isfile(os.path.join(_SAM3_REPO, "sam3", "__init__.py")):
    if _SAM3_REPO in sys.path:
        sys.path.remove(_SAM3_REPO)
    sys.path.insert(0, _SAM3_REPO)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False


def _resolve_bpe_path(bpe_path: Optional[str] = None) -> Optional[str]:
    for candidate in default_bpe_candidates(bpe_path):
        if os.path.isfile(candidate):
            return candidate
    return None


class RSegmenter:
    """SAM3 文本提示分割器（一次图像编码，多组提示词）。"""

    _PROMPTS = ("red_R", "blue_R")
    _AREA_RATIO = (1e-4, 0.05)

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cuda",
        bpe_path: Optional[str] = None,
        processor: Optional[object] = None,
    ) -> None:
        if processor is not None:
            self.processor = processor
            self.device = device
            return

        if not HAS_SAM3:
            raise RuntimeError(
                "未安装 sam3，请将 facebook/sam3 置于项目根或 pip install"
            )
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        resolved_bpe = _resolve_bpe_path(bpe_path)
        if not resolved_bpe:
            raise RuntimeError(
                "找不到 bpe_simple_vocab_16e6.txt.gz，请传入 bpe_path"
            )
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"找不到 SAM3 权重: {checkpoint_path}")
        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint_header = checkpoint_file.read(4)
        if checkpoint_header.startswith(b"PK") and not zipfile.is_zipfile(
            checkpoint_path
        ):
            raise RuntimeError(
                "SAM3 权重损坏或下载不完整（找不到 ZIP central directory）: "
                f"{checkpoint_path}"
            )
        model = build_sam3_image_model(
            bpe_path=resolved_bpe,
            checkpoint_path=checkpoint_path,
            enable_inst_interactivity=True,
        )
        model.to(device)
        self.processor = Sam3Processor(model)

    def _autocast(self):
        if self.device == "cpu" or not torch.cuda.is_available():
            return nullcontext()
        dtype = torch.bfloat16 if (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
        return torch.autocast(device_type=self.device, dtype=dtype)

    @staticmethod
    def _clone_state(state):
        """复制轻量状态容器，共享图像特征，避免每个文本提示重复编码图像。"""
        return state.copy() if isinstance(state, dict) else state

    def _best_from_output(
        self,
        out,
        image_shape: Tuple[int, int],
        area_ratio: Tuple[float, float],
    ) -> Tuple[Optional[np.ndarray], float]:
        masks, scores = out.get("masks", []), out.get("scores", [])
        if len(masks) == 0:
            return None, 0.0
        ih, iw = image_shape
        best, best_score = None, 0.0
        for i in range(len(masks)):
            m = (
                masks[i].detach().cpu().numpy()
                if torch.is_tensor(masks[i])
                else masks[i]
            )
            m = np.squeeze(m)
            raw_score = scores[i]
            s = float(
                raw_score.detach().cpu().item()
                if torch.is_tensor(raw_score)
                else raw_score
            )
            m = (m > 0.5).astype(np.uint8) * 255
            if m.shape != (ih, iw):
                m = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_NEAREST)
            ratio = float(np.count_nonzero(m)) / max(1, ih * iw)
            if area_ratio[0] <= ratio <= area_ratio[1] and s > best_score:
                best, best_score = m, s
        return best, best_score

    def segment_prompts(
        self,
        image_bgr: np.ndarray,
        prompts,
        area_ratio: Tuple[float, float] = _AREA_RATIO,
    ) -> Tuple[Optional[np.ndarray], float, str]:
        """同一图像编码只执行一次，依次尝试提示词并返回最高分 mask。"""
        result = self.segment_prompt_groups(
            image_bgr, {"result": (prompts, area_ratio)}
        )
        return result["result"]

    def segment_prompt_groups(self, image_bgr: np.ndarray, groups: dict) -> dict:
        """一次图像编码处理多组提示词。

        groups 格式: {name: (prompts, (min_area_ratio, max_area_ratio))}。
        返回: {name: (mask, score, prompt)}。
        """
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        results = {}
        with torch.inference_mode(), self._autocast():
            base_state = self.processor.set_image(pil)
            for name, (prompts, area_ratio) in groups.items():
                best_mask, best_score, best_prompt = None, 0.0, ""
                for prompt in prompts:
                    state = self._clone_state(base_state)
                    out = self.processor.set_text_prompt(
                        prompt=str(prompt), state=state
                    )
                    mask, score = self._best_from_output(
                        out, image_bgr.shape[:2], area_ratio
                    )
                    if mask is not None and score > best_score:
                        best_mask, best_score, best_prompt = (
                            mask,
                            score,
                            str(prompt),
                        )
                results[name] = (best_mask, best_score, best_prompt)
        return results

    def segment(
        self, image_bgr: np.ndarray, text_prompt: str
    ) -> Tuple[Optional[np.ndarray], float]:
        """单提示词分割，返回 (mask 0/255, score)。失败返回 (None, 0.0)。"""
        mask, score, _ = self.segment_prompts(image_bgr, (text_prompt,))
        return mask, score

    def segment_best(
        self, image_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, str]:
        """跑 red_R/blue_R，选 score 最高者。返回 (mask, score, prompt)。"""
        return self.segment_prompts(image_bgr, self._PROMPTS)

    @staticmethod
    def centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """mask 质心像素坐标。空 mask 返回 None。"""
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] <= 0:
            return None
        return M["m10"] / M["m00"], M["m01"] / M["m00"]
