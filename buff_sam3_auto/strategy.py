"""能量机关自动标注策略：ONNX bbox + SAM3 双轮廓 + 极轴五点。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import OrtBBoxDetector
from .geometry import (
    contour_centroid,
    select_circle_contour,
    select_r_component,
    solve_five_keypoints,
)
from .paths import default_onnx_path
from .segmenter import RSegmenter


def solve_keypoints_in_roi(
    segmenter: RSegmenter,
    roi: np.ndarray,
    is_red: bool,
) -> Tuple[Optional[List[Tuple[float, float, int]]], dict]:
    """在 ROI 内跑 SAM3 双提示分割（圆盘 + R 标）并做极轴五点解算。

    与 YOLO 无关，供半自动（手动拉框）与全自动共用。

    返回
    ----
    local_kps:
        ROI 局部坐标五点 [(x, y, visibility), ...]，失败为 None
    debug:
        调试字段（mask、轮廓、prompt 等）
    """
    circle_prompts = (
        BuffAutoStrategy.RED_CIRCLE_PROMPTS
        if is_red
        else BuffAutoStrategy.BLUE_CIRCLE_PROMPTS
    )
    r_prompts = (
        BuffAutoStrategy.RED_R_PROMPTS
        if is_red
        else BuffAutoStrategy.BLUE_R_PROMPTS
    )
    groups = segmenter.segment_prompt_groups(
        roi,
        {
            "circle": (circle_prompts, (0.003, 0.80)),
            "r": (r_prompts, (1e-4, 0.20)),
        },
    )
    circle_mask, circle_score, circle_prompt = groups["circle"]
    r_mask, r_score, r_prompt = groups["r"]
    debug = {
        "circle_mask": circle_mask,
        "r_mask": r_mask,
        "circle_prompt": circle_prompt,
        "circle_score": circle_score,
        "r_prompt": r_prompt,
        "r_score": r_score,
    }
    if circle_mask is None:
        debug["fail_reason"] = "circle_prompt_no_mask"
        return None, debug
    if r_mask is None:
        debug["fail_reason"] = "r_prompt_no_mask"
        return None, debug

    r_contour = select_r_component(r_mask)
    r_center = contour_centroid(r_contour) if r_contour is not None else None
    if r_center is None:
        debug["fail_reason"] = "r_component_not_found"
        return None, debug
    circle_contour = select_circle_contour(circle_mask, r_mask, r_center)
    if circle_contour is None:
        debug["fail_reason"] = "circle_contour_not_found"
        return None, debug
    local_kps = solve_five_keypoints(r_center, circle_contour)
    if local_kps is None:
        debug["fail_reason"] = "five_keypoints_solve_failed"
        return None, debug
    debug["circle_contour"] = circle_contour
    return local_kps, debug


class BuffAutoStrategy:
    """将 CUDA ONNX bbox、SAM3 文本分割与极轴几何求解串联。"""

    RED_CIRCLE_PROMPTS = (
        "red dart board",
        "red dart circle",
        "red circle",
        "dart_board",
        "dart_circle",
        "circle",
        "target_circle",
        "red_target",
    )
    BLUE_CIRCLE_PROMPTS = (
        "blue dart board",
        "blue dart circle",
        "blue circle",
        "dart_board",
        "dart_circle",
        "circle",
        "target_circle",
        "blue_target",
    )
    RED_R_PROMPTS = ("red R", "red_R", "R logo", "R", "R_logo")
    BLUE_R_PROMPTS = ("blue R", "blue_R", "R logo", "R", "R_logo")

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        sam_checkpoint: str = "",
        bpe_path: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.6,
        max_det: int = 100,
        device: str = "cuda",
        processor: Optional[object] = None,
    ) -> None:
        resolved_onnx = str(onnx_path or default_onnx_path())
        if not sam_checkpoint and processor is None:
            raise ValueError("必须提供 sam_checkpoint 或 processor（SAM3 处理器）")
        self.detector = OrtBBoxDetector(
            resolved_onnx,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
        )
        self.segmenter = RSegmenter(
            checkpoint_path=sam_checkpoint if processor is None else "",
            device=device,
            bpe_path=bpe_path,
            processor=processor,
        )

    @staticmethod
    def _crop_box(image: np.ndarray, xyxy):
        ih, iw = image.shape[:2]
        x1, y1, x2, y2 = xyxy
        x1i = max(0, int(np.floor(x1)))
        y1i = max(0, int(np.floor(y1)))
        x2i = min(iw, int(np.ceil(x2)))
        y2i = min(ih, int(np.ceil(y2)))
        return image[y1i:y2i, x1i:x2i], x1i, y1i, x2i, y2i

    def annotate(self, image: np.ndarray) -> Tuple[List[dict], Dict[int, dict]]:
        """对 BGR 图像做自动标注。

        返回
        ----
        annotations:
            每项含 cls / confidence / xywh(归一化) / kps(归一化五点)
        debug:
            det_index -> 调试字段（mask、轮廓、prompt 等）
        """
        ih, iw = image.shape[:2]
        annotations: List[dict] = []
        debug: Dict[int, dict] = {}
        for det_index, det in enumerate(self.detector.predict(image)):
            roi, ox, oy, x2, y2 = self._crop_box(image, det.xyxy)
            if roi.size == 0:
                continue
            local_kps, roi_debug = solve_keypoints_in_roi(
                self.segmenter, roi, is_red=(det.cls_id == 0)
            )
            if local_kps is None:
                continue

            x1f, y1f, x2f, y2f = det.xyxy
            annotations.append(
                {
                    "cls": det.cls_id,
                    "label": "RT" if det.cls_id == 0 else "BT",
                    "confidence": det.confidence,
                    "xyxy": (x1f, y1f, x2f, y2f),
                    "roi_rect": (ox, oy, x2 - ox, y2 - oy),
                    "xywh": (
                        ((x1f + x2f) * 0.5) / iw,
                        ((y1f + y2f) * 0.5) / ih,
                        (x2f - x1f) / iw,
                        (y2f - y1f) / ih,
                    ),
                    "kps": [
                        ((x + ox) / iw, (y + oy) / ih, visibility)
                        for x, y, visibility in local_kps
                    ],
                    "pixel_kps": [
                        (x + ox, y + oy, visibility)
                        for x, y, visibility in local_kps
                    ],
                }
            )
            debug[det_index] = {
                "bbox": det.xyxy,
                "circle_mask": roi_debug.get("circle_mask"),
                "r_mask": roi_debug.get("r_mask"),
                "circle_contour": roi_debug["circle_contour"] + np.array([[[ox, oy]]]),
                "circle_prompt": roi_debug.get("circle_prompt"),
                "circle_score": roi_debug.get("circle_score"),
                "r_prompt": roi_debug.get("r_prompt"),
                "r_score": roi_debug.get("r_score"),
                "roi_origin": (ox, oy),
                "roi_size": (x2 - ox, y2 - oy),
            }
        return annotations, debug

    @staticmethod
    def to_yolo_lines(annotations: List[dict]) -> List[str]:
        """将 annotate() 结果转为 YOLO pose txt 行。"""
        lines = []
        for ann in annotations:
            cx, cy, w, h = ann["xywh"]
            line = f"{ann['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            for kx, ky, visibility in ann["kps"]:
                line += f" {kx:.6f} {ky:.6f} {visibility}"
            lines.append(line)
        return lines
