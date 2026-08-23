"""YOLO pose ONNX：只解析 bbox / 类别，不使用网络关键点。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class BBoxDetection:
    cls_id: int
    confidence: float
    xyxy: Tuple[float, float, float, float]


class OrtBBoxDetector:
    """解析 pose ONNX 的 bbox/类别通道。"""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        num_classes: int = 2,
        conf: float = 0.5,
        iou: float = 0.6,
        max_det: int = 100,
        device: str = "cuda",
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        available = ort.get_available_providers()
        providers = []
        if device != "cpu" and "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                )
            )
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = int(imgsz)
        self.num_classes = int(num_classes)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.provider = self.session.get_providers()[0]

    def _preprocess(self, image: np.ndarray):
        ih, iw = image.shape[:2]
        ratio = min(self.imgsz / iw, self.imgsz / ih)
        nw, nh = int(round(iw * ratio)), int(round(ih * ratio))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.imgsz - nw) // 2
        pad_y = (self.imgsz - nh) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        blob = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        )
        blob *= 1.0 / 255.0
        return blob, ratio, pad_x, pad_y

    def predict(self, image: np.ndarray) -> List[BBoxDetection]:
        blob, ratio, pad_x, pad_y = self._preprocess(image)
        raw = self.session.run(None, {self.input_name: blob})[0]
        pred = raw[0].T if raw.ndim == 3 else raw
        if pred.ndim != 2 or pred.shape[1] < 4 + self.num_classes:
            raise RuntimeError(f"无法解析 ONNX 输出形状: {raw.shape}")

        scores_all = pred[:, 4 : 4 + self.num_classes]
        cls_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(len(pred)), cls_ids]
        keep = scores >= self.conf
        if not np.any(keep):
            return []

        xywh = pred[keep, :4].astype(np.float32)
        scores = scores[keep].astype(np.float32)
        cls_ids = cls_ids[keep].astype(np.int32)
        x = xywh[:, 0] - xywh[:, 2] * 0.5
        y = xywh[:, 1] - xywh[:, 3] * 0.5
        nms_boxes = np.column_stack((x, y, xywh[:, 2], xywh[:, 3]))

        # 分类别 NMS，避免红/蓝类别互相抑制。
        selected: List[int] = []
        for cls_id in np.unique(cls_ids):
            ids = np.flatnonzero(cls_ids == cls_id)
            picked = cv2.dnn.NMSBoxes(
                nms_boxes[ids].tolist(),
                scores[ids].tolist(),
                self.conf,
                self.iou,
            )
            if len(picked):
                selected.extend(ids[np.asarray(picked).reshape(-1)].tolist())
        selected.sort(key=lambda i: float(scores[i]), reverse=True)

        ih, iw = image.shape[:2]
        out = []
        for i in selected[: self.max_det]:
            x1 = (x[i] - pad_x) / ratio
            y1 = (y[i] - pad_y) / ratio
            x2 = (x[i] + xywh[i, 2] - pad_x) / ratio
            y2 = (y[i] + xywh[i, 3] - pad_y) / ratio
            x1 = float(np.clip(x1, 0, iw - 1))
            y1 = float(np.clip(y1, 0, ih - 1))
            x2 = float(np.clip(x2, 0, iw - 1))
            y2 = float(np.clip(y2, 0, ih - 1))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                out.append(
                    BBoxDetection(
                        int(cls_ids[i]), float(scores[i]), (x1, y1, x2, y2)
                    )
                )
        return out
