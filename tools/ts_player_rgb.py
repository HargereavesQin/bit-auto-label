#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS 视频播放器 + RGB 增益调节 + YOLO 检测框 (OpenCV trackbar).

用法:
    python tools/ts_player_rgb.py
    python tools/ts_player_rgb.py --model /path/best.pt --conf 0.35
    python tools/ts_player_rgb.py --model /path/best.onnx --device cuda

交互:
    Space   暂停/继续
    d       开关 YOLO 检测叠加
    s       保存当前 R/G/B 增益 (JSON + 控制台打印)
    左/右    快退/快进 1s
    q/Esc   退出
    进度条   直接拖动跳转 (ts 为流式, 跳转会有少量延迟)
"""
import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WINDOW = "TS Player (Space=pause, d=detect, s=save, q=quit)"
DEFAULT_VIDEO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "camera_20260805_105958_part002.ts",
)
TRACKBAR_MAX = 200  # 0.0 ~ 2.0 gain, step 0.01
SEEK_TRACKBAR = "frame"


class PtDetector:
    """ultralytics .pt 权重包装, 输出与 OnnxDetector 一致的 dict 列表。"""

    backend = "ultralytics-pt"
    task = "detect"
    num_kpts = 0

    def __init__(self, pt_path: str, device: str = "cuda", imgsz: int = 640) -> None:
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"pt 不存在: {pt_path}")
        from ultralytics import YOLO
        self.model = YOLO(pt_path)
        self.device = device
        self.imgsz = imgsz
        names = self.model.names
        self.names = {int(k): v for k, v in names.items()} if isinstance(names, dict) \
            else {i: n for i, n in enumerate(names)}

    def infer(self, img_bgr: np.ndarray, conf_thres: float, iou_thres: float,
              max_det: int = 300) -> list[dict]:
        results = self.model.predict(
            img_bgr, conf=conf_thres, iou=iou_thres, max_det=max_det,
            imgsz=self.imgsz, device=self.device, verbose=False,
        )
        r = results[0]
        dets = []
        for xyxy, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
            dets.append({
                "cls": int(cls),
                "conf": float(conf),
                "box": [float(v) for v in xyxy],
                "kpts": [],
            })
        return dets


def build_detector(model_path: str, device: str, imgsz: int):
    from onnx_auto_annotate import OnnxDetector, TrtDetector
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pt":
        return PtDetector(model_path, device=device, imgsz=imgsz)
    if ext in (".engine", ".plan"):
        return TrtDetector(model_path, imgsz=imgsz)
    return OnnxDetector(model_path, device=device, imgsz=imgsz)


class Player:
    def __init__(self, video_path: str, save_path: str,
                 init_r: float, init_g: float, init_b: float,
                 detector=None, conf: float = 0.35, iou: float = 0.45,
                 max_det: int = 50) -> None:
        self.video_path = video_path
        self.save_path = save_path
        self.det = detector
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.detect_on = detector is not None

        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise SystemExit(f"无法打开视频: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.delay = max(1, int(round(1000.0 / self.fps)))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.seek_max = max(1, self.total - 1)
        print(f"视频: {video_path}  fps={self.fps:.1f}  frames={self.total}")

        self.paused = False
        self.frame = None
        self.idx = 0
        self.seek_requested = False

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
        cv2.createTrackbar("R", WINDOW, int(init_r * 100), TRACKBAR_MAX, _noop)
        cv2.createTrackbar("G", WINDOW, int(init_g * 100), TRACKBAR_MAX, _noop)
        cv2.createTrackbar("B", WINDOW, int(init_b * 100), TRACKBAR_MAX, _noop)
        cv2.createTrackbar(SEEK_TRACKBAR, WINDOW, 0, self.seek_max, self._on_seek)

    def _on_seek(self, pos: int) -> None:
        """拖动进度条 -> 标记跳转; 播放循环里真正执行 set()。"""
        if abs(pos - self.idx) <= 1:
            return
        self.seek_requested = True

    def _gain(self, name: str) -> float:
        return cv2.getTrackbarPos(name, WINDOW) / 100.0

    def _apply_gain(self, frame: np.ndarray, r: float, g: float, b: float) -> np.ndarray:
        f = frame.astype(np.float32)
        f[:, :, 0] *= b
        f[:, :, 1] *= g
        f[:, :, 2] *= r
        return np.clip(f, 0, 255).astype(np.uint8)

    def _save_gains(self, r: float, g: float, b: float) -> None:
        payload = {
            "video": os.path.basename(self.video_path),
            "r_gain": round(r, 3),
            "g_gain": round(g, 3),
            "b_gain": round(b, 3),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.save_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"[save] {self.save_path}  R={r:.3f} G={g:.3f} B={b:.3f}")

    def _seek_to(self, pos: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, f = self.cap.read()
        if ok:
            self.frame = f
            self.idx = pos

    def _step(self, offset_sec: int) -> None:
        target = int(np.clip(self.idx + int(offset_sec * self.fps), 0, self.seek_max))
        self._seek_to(target)

    def run(self) -> None:
        from onnx_auto_annotate import draw
        while True:
            if self.seek_requested:
                self.seek_requested = False
                self._seek_to(cv2.getTrackbarPos(SEEK_TRACKBAR, WINDOW))

            if not self.paused or self.frame is None:
                ok, f = self.cap.read()
                if not ok:
                    print("播放结束")
                    break
                self.frame = f
                self.idx += 1

            r, g, b = self._gain("R"), self._gain("G"), self._gain("B")
            vis = self._apply_gain(self.frame, r, g, b)

            n_det = -1
            if self.detect_on and self.det is not None:
                dets = self.det.infer(vis, self.conf, self.iou, self.max_det)
                vis = draw(vis, dets, self.det.names, self.det.num_kpts)
                n_det = len(dets)

            det_txt = f" dets={n_det}" if n_det >= 0 else ""
            info = f"[{self.idx}/{self.total or '?'}] R={r:.2f} G={g:.2f} B={b:.2f}{det_txt}"
            cv2.putText(vis, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, vis)

            # 播放时同步进度条位置
            try:
                cv2.setTrackbarPos(SEEK_TRACKBAR, WINDOW, self.idx)
            except cv2.error:
                pass

            key = cv2.waitKey(0 if self.paused else self.delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                self.paused = not self.paused
            elif key == ord("s"):
                self._save_gains(r, g, b)
            elif key == ord("d"):
                if self.det is not None:
                    self.detect_on = not self.detect_on
                    print(f"detect overlay: {'ON' if self.detect_on else 'OFF'}")
            elif key in (81, 2):      # Left
                self._step(-1)
            elif key in (83, 3):      # Right
                self._step(+1)

        self.cap.release()
        cv2.destroyAllWindows()


def _noop(_v):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description="TS player with RGB gain + YOLO detect overlay")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="video path (.ts)")
    ap.add_argument("--save", default=None, help="JSON output path")
    ap.add_argument("--init-r", type=float, default=1.0)
    ap.add_argument("--init-g", type=float, default=0.7)
    ap.add_argument("--init-b", type=float, default=1.0)
    ap.add_argument("--model", default=None,
                    help="YOLO 权重: .pt / .onnx / .engine (不传则只调 RGB)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"视频不存在: {args.video}")

    detector = None
    if args.model:
        detector = build_detector(args.model, args.device, args.imgsz)
        print(f"模型: {args.model}  backend={detector.backend}  names={detector.names}")

    save_path = args.save or f"rgb_gain_{os.path.splitext(os.path.basename(args.video))[0]}.json"
    Player(args.video, save_path, args.init_r, args.init_g, args.init_b,
           detector=detector, conf=args.conf, iou=args.iou, max_det=args.max_det).run()


if __name__ == "__main__":
    main()
