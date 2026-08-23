"""用 onnxruntime-gpu 对视频做 yolo11n-pose 推理。

用法: python tools/infer_pose_video.py [视频路径] [--onnx ONNX路径] [--out 输出路径]
默认: reddy.mp4 -> reddy_pose_out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = str(HERE / "runs" / "pose" / "yolo11n_buff2" / "weights" / "best.onnx")
DEFAULT_VIDEO = str(HERE / "reddy.mp4")

IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

# 5 个关键点连接: 0->1, 后四个点(1,2,3,4)首尾相连形成菱形
KPT_SKELETON = [
    (0, 1),            # 点0 连到 菱形顶点1
    (1, 2), (2, 3), (3, 4), (4, 1),  # 后四点闭合菱形
]
KPT_COLORS = [
    (0, 255, 255),   (255, 0, 255), (255, 255, 0),
    (0, 255, 0),     (255, 0, 0),
]
CLASS_NAMES = ["red_buff", "blue_buff"]
CLASS_COLORS = [(0, 0, 255), (255, 0, 0)]  # BGR


def letterbox(img: np.ndarray, new_size: int = 640):
    """等比缩放 + 居中填充到 new_size×new_size, 返回 (out, ratio, (pad_w, pad_h))。"""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    pad_h = (new_size - nh) // 2
    pad_w = (new_size - nw) // 2
    out[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return out, r, (pad_w, pad_h)


def preprocess(img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    out, r, (pw, ph) = letterbox(img, IMGSZ)
    # BGR -> RGB, HWC -> CHW, normalize
    blob = out[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis]  # add batch dim
    return blob, r, (pw, ph)


def postprocess(
    output: np.ndarray,
    orig_h: int,
    orig_w: int,
    ratio: float,
    pad: tuple[int, int],
    conf_thres: float,
    iou_thres: float,
) -> list[dict]:
    """解析 YOLO11-pose ONNX 输出。

    output: [1, 21, N]  ->  21 = 4(xywh) + 2(cls) + 15(5×3 kpt)
    """
    pred = output[0].T  # [N, 21]
    # 前 4 列: cx, cy, w, h (输入图像素坐标)
    boxes_xywh = pred[:, :4]
    # 第 5-6 列: 两类分数
    scores = pred[:, 4:6]
    # 第 7-21 列: 5 个关键点 (px, py, conf)
    kpts = pred[:, 6:]  # [N, 15]

    max_scores = scores.max(axis=1)
    mask = max_scores > conf_thres
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    cls_ids = scores[mask].argmax(axis=1)
    kpts = kpts[mask]

    # xywh -> xyxy (输入图坐标)
    xyxy = np.zeros((len(boxes_xywh), 4), dtype=np.float32)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # NMS
    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), max_scores.tolist(), conf_thres, iou_thres
    )
    if len(idxs) == 0:
        return []
    idxs = idxs.flatten()

    results = []
    pad_w, pad_h = pad
    for i in idxs:
        # 还原到原图坐标: 减去 padding, 除以 ratio
        x1 = (xyxy[i, 0] - pad_w) / ratio
        y1 = (xyxy[i, 1] - pad_h) / ratio
        x2 = (xyxy[i, 2] - pad_w) / ratio
        y2 = (xyxy[i, 3] - pad_h) / ratio
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        # 关键点还原
        kp_list = []
        for k in range(5):
            px = (kpts[i, k * 3] - pad_w) / ratio
            py = (kpts[i, k * 3 + 1] - pad_h) / ratio
            conf = kpts[i, k * 3 + 2]
            px = max(0, min(px, orig_w - 1))
            py = max(0, min(py, orig_h - 1))
            kp_list.append((px, py, conf))

        results.append({
            "cls": int(cls_ids[i]),
            "conf": float(max_scores[i]),
            "box": [x1, y1, x2, y2],
            "kpts": kp_list,
        })
    return results


def draw(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    h, w = img.shape[:2]
    # 根据图像分辨率动态自适应计算绘制比例与粗细
    scale_factor = max(1.0, min(w, h) / 600.0)
    box_thick = max(2, int(round(2 * scale_factor)))
    font_scale = max(0.6, 0.5 * scale_factor)
    font_thick = max(1, int(round(1.5 * scale_factor)))
    kpt_radius = max(5, int(round(5 * scale_factor)))
    line_thick = max(2, int(round(2.5 * scale_factor)))

    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cls = d["cls"]
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        label = f"{CLASS_NAMES[cls]} {d['conf']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        cv2.rectangle(img, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # 画骨架与关键点
        pts = []
        for k, (px, py, conf) in enumerate(d["kpts"]):
            if conf > 0.3:
                cv2.circle(img, (int(px), int(py)), kpt_radius,
                           KPT_COLORS[k % len(KPT_COLORS)], -1, cv2.LINE_AA)
                cv2.circle(img, (int(px), int(py)), kpt_radius + 1,
                           (0, 0, 0), 1, cv2.LINE_AA)
                pts.append((int(px), int(py)))
            else:
                pts.append(None)
        for a, b in KPT_SKELETON:
            if pts[a] is not None and pts[b] is not None:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), line_thick, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gui", action="store_true", default=True,
                    help="显示 OpenCV 实时窗口 (默认开启)")
    ap.add_argument("--no-gui", dest="gui", action="store_false",
                    help="不显示窗口, 只写视频文件")
    ap.add_argument("--no-save", action="store_true",
                    help="不保存输出视频")
    args = ap.parse_args()

    video_path = args.video
    onnx_path = args.onnx
    out_path = args.out or str(
        Path(video_path).with_stem(Path(video_path).stem + "_pose_out")
    )

    print(f"ONNX 模型: {onnx_path}")
    print(f"输入视频: {video_path}")
    if not args.no_save:
        print(f"输出视频: {out_path}")
    if args.gui:
        print("GUI 模式: 开启 (按 q 退出, 空格暂停)")
    print()

    sess = ort.InferenceSession(
        onnx_path,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    input_name = sess.get_inputs()[0].name
    print(f"输入: {input_name}")
    print(f"输出: {[o.name for o in sess.get_outputs()]}")
    print(f"providers: {sess.get_providers()}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"视频: {w}x{h} @ {fps:.1f}fps, 共 {total} 帧\n")

    window_title = "YOLO11n-Pose 推理 (q退出, 空格暂停)"
    if args.gui:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        # 计算适宜的默认窗口尺寸（如 1280x900）
        disp_w = 1280
        disp_h = int(round(1280 * (h / max(1, w))))
        cv2.resizeWindow(window_title, disp_w, disp_h)

    paused = False
    frame_idx = 0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            blob, ratio, pad = preprocess(frame)
            outputs = sess.run(None, {input_name: blob})
            dets = postprocess(
                outputs[0], h, w, ratio, pad, CONF_THRES, IOU_THRES
            )
            frame = draw(frame, dets)

            if writer is not None:
                writer.write(frame)
            if frame_idx % 50 == 0 or frame_idx == 1:
                n = len(dets)
                print(f"  帧 {frame_idx}/{total}: 检测到 {n} 个目标")

        if args.gui:
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord("q"):
                print("用户退出")
                break
            elif key == ord(" "):
                paused = not paused
                print(f"  {'暂停' if paused else '继续'}")

    cap.release()
    if writer is not None:
        writer.release()
    if args.gui:
        cv2.destroyAllWindows()
    print(f"\n完成! 共处理 {frame_idx} 帧"
          + (f", 输出: {out_path}" if writer is not None else ""))


if __name__ == "__main__":
    main()
