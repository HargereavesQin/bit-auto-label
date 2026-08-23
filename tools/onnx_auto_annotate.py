# -*- coding: utf-8 -*-
"""基于 infer_pose_video.py 的通用 ONNX / TensorRT 自动标注脚本。

- ONNX 走 onnxruntime；.engine 走 tensorrt + torch CUDA tensor 绑定（无需 pycuda）。
- 从模型 metadata 读取 task(detect/pose) / names / kpt_shape，自动适配输出布局。
- 图片目录: 逐张推理，向图片同源目录写 labelme JSON；classes.txt 不覆盖。
- 视频: 逐帧推理，抽帧+JSON 写入 --frames-dir，可选保存预览 mp4。

用法:
  # 图片目录自动标注 (ONNX)
  python tools/onnx_auto_annotate.py \
      --model /path/to/best_fp32.onnx --images /path/to/images \
      --annotate --policy skip --save-vis

  # TensorRT engine (re-radar-2026 radar 环境)
  /home/dreamchaser/re-radar-2026/radar/bin/python tools/onnx_auto_annotate.py \
      --model /home/dreamchaser/re-radar-2026/weights/car.engine \
      --images /path/to/images --annotate --policy skip

  # 视频抽帧 + 标注 + 预览
  python tools/onnx_auto_annotate.py \
      --model /path/to/car.engine --video /path/to/video.mp4 \
      --frames-dir /path/to/frames --annotate --save-video

  # 只预览不写 JSON
  python tools/onnx_auto_annotate.py --model ... --images ... --no-annotate --save-vis
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")

CLASS_COLORS = [
    (255, 107, 107), (78, 205, 196), (69, 183, 209), (150, 206, 180),
    (255, 234, 167), (255, 159, 243), (0, 210, 211), (255, 127, 80),
    (147, 112, 219), (60, 179, 113),
]

# 候选 TensorRT lib 目录：进程启动后改 LD_LIBRARY_PATH 无效，
# 先用 ctypes 全局加载 libnvinfer，再 import tensorrt。
_TRT_LIB_CANDIDATES = [
    "/home/dreamchaser/Downloads/TensorRT-10.12.0.36.Linux.x86_64-gnu.cuda-12.9/TensorRT-10.12.0.36/lib",
    os.path.expanduser("~/.local/lib/python3.10/site-packages/tensorrt_libs"),
]


def _ensure_trt_libs() -> None:
    import ctypes
    names = ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"]
    for d in _TRT_LIB_CANDIDATES:
        loaded_any = False
        for so_name in names:
            so = os.path.join(d, so_name)
            if os.path.isfile(so):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True
                except OSError:
                    pass
        if loaded_any:
            return


def parse_class_names(raw) -> dict[int, str]:
    """metadata 中的 names（str/dict）→ {int: str}。"""
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        return {int(k): str(v) for k, v in dict(parsed).items()}
    except Exception:
        return {0: str(raw)}


def parse_kpt_shape(raw):
    if not raw:
        return None
    try:
        return list(ast.literal_eval(raw) if isinstance(raw, str) else raw)
    except Exception:
        return None


def shape_dims(shape) -> tuple[int, ...] | None:
    dims = tuple(int(d) for d in shape)
    if any(d <= 0 for d in dims):
        return None
    return dims


# ---------------- 通用基类 ----------------

class BaseDetector:
    """统一接口: infer(img, conf, iou) → dets; 子类负责加载模型。"""

    task: str = "detect"
    names: dict[int, str] = {0: "class_0"}
    num_kpts: int = 0
    imgsz: int = 640
    backend: str = "base"

    def _resolve_kpts(self, num_channels: int | None) -> None:
        nc = len(self.names)
        if num_channels is not None:
            nk = (num_channels - 4 - nc) // 3
            if self.task == "pose" and nk > 0:
                self.num_kpts = nk
                return
        if self.task == "pose" and self.kpt_shape:
            self.num_kpts = self.kpt_shape[0]

    def info(self) -> str:
        return (f"backend={self.backend}, task={self.task}, nc={len(self.names)}, "
                f"names={self.names}, kpts={self.num_kpts}, imgsz={self.imgsz}")

    def infer(self, img_bgr: np.ndarray, conf_thres: float, iou_thres: float, max_det: int = 300) -> list[dict]:
        raise NotImplementedError

    def _postprocess(self, output, h, w, ratio, pad, conf_thres, iou_thres, max_det: int = 300):
        return postprocess(
            output, h, w, ratio, pad, conf_thres, iou_thres,
            nc=len(self.names), num_kpts=self.num_kpts, is_pose=(self.task == "pose"),
            max_det=max_det,
        )


# ---------------- ONNX Runtime 后端 ----------------

class OnnxDetector(BaseDetector):
    backend = "onnxruntime"

    def __init__(self, onnx_path: str, device: str = "cuda", imgsz: int = 640):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX 不存在: {onnx_path}")
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

        in_dims = shape_dims(self.session.get_inputs()[0].shape) or (1, 3, imgsz, imgsz)
        self.imgsz = in_dims[-1]

        md = self.session.get_modelmeta().custom_metadata_map
        self.task = md.get("task", "detect")
        self.names = parse_class_names(md.get("names")) or {0: "class_0"}
        self.kpt_shape = parse_kpt_shape(md.get("kpt_shape"))

        out_shape = self.session.get_outputs()[0].shape
        num_channels = out_shape[1] if len(out_shape) == 3 and isinstance(out_shape[1], int) else None
        self._resolve_kpts(num_channels)

    def info(self) -> str:
        return super().info() + f", providers={self.providers}"

    def infer(self, img_bgr: np.ndarray, conf_thres: float, iou_thres: float, max_det: int = 300) -> list[dict]:
        blob, ratio, pad = preprocess(img_bgr, self.imgsz)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs[0], img_bgr.shape[0], img_bgr.shape[1],
                                 ratio, pad, conf_thres, iou_thres, max_det=max_det)


# ---------------- TensorRT 后端 ----------------

class TrtDetector(BaseDetector):
    """直接加载 .engine。绑定用 torch CUDA tensor（无需 pycuda）。"""

    backend = "tensorrt"

    def __init__(self, engine_path: str, imgsz: int = 640):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"engine 不存在: {engine_path}")
        _ensure_trt_libs()
        import tensorrt as trt
        import torch

        meta = self._read_metadata(engine_path)
        self.task = meta.get("task", "detect") if meta else "detect"
        self.names = parse_class_names(meta.get("names")) if meta else {}
        self.kpt_shape = parse_kpt_shape(meta.get("kpt_shape")) if meta else None

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            data = f.read()
        if meta:
            offset = 4 + int.from_bytes(data[:4], "little")
            data = data[offset:]
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"engine 反序列化失败（平台/版本不匹配）: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.torch = torch
        self.device = torch.device("cuda:0")
        self._bindings: dict[str, torch.Tensor] = {}
        self._order: list[str] = []
        out_dims = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dims = shape_dims(self.engine.get_tensor_shape(name))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if dims is None:
                if is_input:
                    dims = (1, 3, imgsz, imgsz)
                    self.context.set_input_shape(name, dims)
                    dims = shape_dims(self.context.get_tensor_shape(name))
                else:
                    dims = shape_dims(self.context.get_tensor_shape(name))
            if dims is None:
                raise RuntimeError(f"无法确定 tensor 形状: {name}")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            buf = torch.empty(dims, dtype=torch_dtype, device=self.device)
            self._bindings[name] = buf
            self._order.append(name)
            if is_input:
                self.input_name = name
                self.imgsz = dims[-1]
            else:
                out_dims = dims

        if not self.names and out_dims is not None:
            nc = out_dims[1] - 4 if len(out_dims) == 3 else 1
            self.names = {k: f"class_{k}" for k in range(max(1, nc))}
        self._resolve_kpts(out_dims[1] if out_dims and len(out_dims) == 3 else None)
        self.input_dtype = self._bindings[self.input_name].dtype

    @staticmethod
    def _read_metadata(path: str):
        try:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(4), "little")
                return json.loads(f.read(n).decode("utf-8"))
        except Exception:
            return None

    def infer(self, img_bgr: np.ndarray, conf_thres: float, iou_thres: float, max_det: int = 300) -> list[dict]:
        blob, ratio, pad = preprocess(img_bgr, self.imgsz)
        inp = self._bindings[self.input_name]
        inp.copy_(self.torch.from_numpy(blob).to(self.device).to(inp.dtype))
        ptrs = [int(self._bindings[n].data_ptr()) for n in self._order]
        self.context.execute_v2(ptrs)
        out_name = [n for n in self._order if n != self.input_name][0]
        output = self._bindings[out_name].float().cpu().numpy()
        return self._postprocess(output, img_bgr.shape[0], img_bgr.shape[1],
                                 ratio, pad, conf_thres, iou_thres, max_det=max_det)


# ---------------- 预处理 / 后处理 (基于 infer_pose_video.py) ----------------

def letterbox(img: np.ndarray, new_size: int = 640):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    pad_h = (new_size - nh) // 2
    pad_w = (new_size - nw) // 2
    out[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return out, r, (pad_w, pad_h)


def preprocess(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    out, r, (pw, ph) = letterbox(img, imgsz)
    blob = out[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis]
    return blob, r, (pw, ph)


def postprocess(
    output: np.ndarray,
    orig_h: int,
    orig_w: int,
    ratio: float,
    pad: tuple[int, int],
    conf_thres: float,
    iou_thres: float,
    nc: int,
    num_kpts: int,
    is_pose: bool,
    max_det: int = 300,
) -> list[dict]:
    """通用解析: [1, 4+nc(+3*k), N]。detect 只取 box，pose 附加关键点。"""
    pred = output[0].T  # [N, C]
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:4 + nc]
    kpts = pred[:, 4 + nc:4 + nc + 3 * num_kpts] if is_pose else None

    max_scores = scores.max(axis=1)
    mask = max_scores > conf_thres
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    cls_ids = scores[mask].argmax(axis=1)
    if kpts is not None:
        kpts = kpts[mask]

    xyxy = np.zeros((len(boxes_xywh), 4), dtype=np.float32)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    idxs = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), max_scores.tolist(), conf_thres, iou_thres)
    if len(idxs) == 0:
        return []
    idxs = idxs.flatten()[:max_det]

    results = []
    pad_w, pad_h = pad

    def _clip(v, lo, hi):
        return max(lo, min(v, hi))

    for i in idxs:
        x1 = _clip((xyxy[i, 0] - pad_w) / ratio, 0, orig_w - 1)
        y1 = _clip((xyxy[i, 1] - pad_h) / ratio, 0, orig_h - 1)
        x2 = _clip((xyxy[i, 2] - pad_w) / ratio, 0, orig_w - 1)
        y2 = _clip((xyxy[i, 3] - pad_h) / ratio, 0, orig_h - 1)

        kp_list = []
        if is_pose and num_kpts > 0:
            for k in range(num_kpts):
                px = _clip((kpts[i, k * 3] - pad_w) / ratio, 0, orig_w - 1)
                py = _clip((kpts[i, k * 3 + 1] - pad_h) / ratio, 0, orig_h - 1)
                conf = float(kpts[i, k * 3 + 2])
                kp_list.append((px, py, conf))

        results.append({
            "cls": int(cls_ids[i]),
            "conf": float(max_scores[i]),
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "kpts": kp_list,
        })
    return results


# ---------------- 可视化 ----------------

def draw(img: np.ndarray, dets: list[dict], names: dict[int, str], num_kpts: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale_factor = max(1.0, min(w, h) / 600.0)
    box_thick = max(2, int(round(2 * scale_factor)))
    font_scale = max(0.6, 0.5 * scale_factor)
    font_thick = max(1, int(round(1.5 * scale_factor)))
    kpt_radius = max(4, int(round(4 * scale_factor)))

    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cls = d["cls"]
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        label = f"{names.get(cls, f'class_{cls}')} {d['conf']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        cv2.rectangle(img, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        for k, (px, py, conf) in enumerate(d["kpts"][:num_kpts]):
            if conf > 0.3:
                pt = (int(px), int(py))
                cv2.circle(img, pt, kpt_radius, CLASS_COLORS[k % len(CLASS_COLORS)], -1, cv2.LINE_AA)
                cv2.circle(img, pt, kpt_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# ---------------- JSON 写出 (与 core/yolo_client 同格式) ----------------

def kpt_visibility(conf: float) -> int:
    if conf >= 0.5:
        return 2
    if conf >= 0.1:
        return 1
    return 0


def dets_to_shapes(dets: list[dict], names: dict[int, str], num_kpts: int, is_pose: bool) -> list[dict]:
    shapes = []
    for d in dets:
        label = names.get(d["cls"], f"class_{d['cls']}")
        x1, y1, x2, y2 = d["box"]
        entry = {
            "label": label,
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "type": "pose" if is_pose else "rectangle",
        }
        if is_pose:
            kpts = [[float(px), float(py), kpt_visibility(conf)] for px, py, conf in d["kpts"][:num_kpts]]
            entry["keypoints"] = kpts
            entry["kpt_names"] = [f"kpt_{k}" for k in range(len(kpts))]
        shapes.append(entry)
    return shapes


def build_annotation(image_filename: str, img_w: int, img_h: int, shapes: list[dict]) -> dict:
    ann = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_filename,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w),
    }
    for s in shapes:
        entry = {
            "label": s["label"],
            "points": s["points"],
            "group_id": None,
            "shape_type": s["type"],
            "flags": {},
        }
        if s["type"] == "pose":
            entry["keypoints"] = s["keypoints"]
            entry["kpt_names"] = s["kpt_names"]
        ann["shapes"].append(entry)
    return ann


def save_annotation(json_path: str, image_filename: str, img_w: int, img_h: int,
                    shapes: list[dict], policy: str) -> str:
    """按 policy 写 JSON。返回动作: skipped/exists_empty/written/appended。"""
    has_json = os.path.exists(json_path)
    if policy == "skip" and has_json:
        return "skipped"
    if policy == "append" and has_json:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            shapes = existing.get("shapes", []) + [
                {**s, "shape_type": s["type"], "group_id": None, "flags": {}}
                for s in build_annotation(image_filename, img_w, img_h, shapes)["shapes"]
            ]
            ann = existing
            ann["shapes"] = shapes
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=2)
            return "appended"
        except Exception as e:
            print(f"  append 失败，改写为新 JSON: {json_path}: {e}")

    if not shapes and policy != "overwrite":
        return "exists_empty"  # 无检测且非覆盖：不写空 JSON

    ann = build_annotation(image_filename, img_w, img_h, shapes)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    return "written"


def write_classes_txt(directory: str, names: dict[int, str]) -> None:
    path = os.path.join(directory, "classes.txt")
    if os.path.exists(path):
        return
    ordered = [names[k] for k in sorted(names.keys())]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ordered) + "\n")
    print(f"已生成 classes.txt: {ordered}")


# ---------------- 图片目录模式 ----------------

def scan_images(root: str) -> list[str]:
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "vis"]  # 排除 --save-vis 输出目录
        for fn in files:
            if fn.endswith(IMAGE_EXTS):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _gui_show(window: str, vis: np.ndarray, paused: bool, delay: int, status: str) -> tuple[bool, bool]:
    """显示一帧并处理按键。返回 (paused, quit)。

    空格暂停/继续，Enter 立即下一张，q/ESC 退出；未暂停时按 delay(ms) 自动前进。
    """
    disp = vis.copy()
    cv2.putText(disp, status, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    while True:
        cv2.imshow(window, disp)
        key = cv2.waitKey(0 if paused else max(1, delay)) & 0xFF
        if key in (ord("q"), 27):
            return paused, True
        if key == ord(" "):
            paused = not paused
            if not paused:
                return paused, False
            continue
        if key == 13 and paused:  # Enter: 暂停中单步
            return paused, False
        if not paused:
            return paused, False


def run_images(det: BaseDetector, args) -> None:
    images = scan_images(args.images)
    if not images:
        raise SystemExit(f"目录下没有图片: {args.images}")
    print(f"共 {len(images)} 张图片, policy={args.policy}, annotate={args.annotate}")

    vis_dir = None
    if args.save_vis:
        vis_dir = os.path.join(args.images, "vis")
        os.makedirs(vis_dir, exist_ok=True)

    window_title = "ONNX 自动标注 (空格暂停, Enter单步, q退出)"
    paused = False
    if args.gui:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1280, 900)

    stats = {"written": 0, "appended": 0, "skipped": 0, "exists_empty": 0, "failed": 0}
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{idx}/{len(images)}] 读取失败: {img_path}")
            stats["failed"] += 1
            continue
        h, w = img.shape[:2]
        dets = det.infer(img, args.conf, args.iou, max_det=args.max_det)

        action = "no-annotate"
        if args.annotate:
            shapes = dets_to_shapes(dets, det.names, det.num_kpts, det.task == "pose")
            json_path = os.path.splitext(img_path)[0] + ".json"
            action = save_annotation(json_path, os.path.basename(img_path), w, h, shapes, args.policy)
            stats[action] = stats.get(action, 0) + 1

        vis = draw(img.copy(), dets, det.names, det.num_kpts)
        if vis_dir:
            cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_path)), vis)

        if idx % 50 == 0 or idx == 1 or idx == len(images):
            print(f"[{idx}/{len(images)}] {os.path.basename(img_path)}: {len(dets)} 个目标 ({action})")

        if args.gui:
            status = f"[{idx}/{len(images)}] {os.path.basename(img_path)}  dets={len(dets)}  {action}"
            paused, quit_now = _gui_show(window_title, vis, paused, args.delay, status)
            if quit_now:
                print("用户退出")
                break

    if args.gui:
        cv2.destroyAllWindows()
    if args.annotate:
        write_classes_txt(args.images, det.names)
    print(f"\n完成: {stats}")


# ---------------- 视频模式 ----------------

def run_video(det: BaseDetector, args) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = Path(args.video).stem
    print(f"视频: {w}x{h} @ {fps:.1f}fps, 共 {total} 帧")

    writer = None
    if args.save_video:
        out_path = args.out or str(Path(args.video).with_stem(stem + "_det_out"))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"预览视频: {out_path}")

    frames_dir = args.frames_dir
    if args.annotate:
        if not frames_dir:
            frames_dir = str(Path(args.video).with_name(stem + "_frames"))
        os.makedirs(frames_dir, exist_ok=True)
        print(f"抽帧目录: {frames_dir} (step={args.frame_step})")

    window_title = "ONNX 自动标注 (q退出, 空格暂停)"
    if args.gui:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1280, int(round(1280 * h / max(1, w))))

    paused = False
    frame_idx = 0
    saved = 0
    stats = {"written": 0, "appended": 0, "skipped": 0, "exists_empty": 0}
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            dets = det.infer(frame, args.conf, args.iou, max_det=args.max_det)
            vis = draw(frame.copy(), dets, det.names, det.num_kpts)

            if writer is not None:
                writer.write(vis)

            if args.annotate and frame_idx % args.frame_step == 0:
                img_name = f"{stem}_{frame_idx:06d}.jpg"
                img_path = os.path.join(frames_dir, img_name)
                cv2.imwrite(img_path, frame)
                shapes = dets_to_shapes(dets, det.names, det.num_kpts, det.task == "pose")
                json_path = os.path.splitext(img_path)[0] + ".json"
                action = save_annotation(json_path, img_name, w, h, shapes, args.policy)
                stats[action] = stats.get(action, 0) + 1
                saved += 1

            if frame_idx % 100 == 0 or frame_idx == 1:
                print(f"  帧 {frame_idx}/{total}: {len(dets)} 个目标, 已存 {saved} 帧")

            frame = vis
        if args.gui:
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord("q"):
                print("用户退出")
                break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
    if args.gui:
        cv2.destroyAllWindows()

    if args.annotate:
        write_classes_txt(frames_dir, det.names)
    print(f"\n完成: 处理 {frame_idx} 帧, 保存 {saved} 帧, {stats}")


# ---------------- main ----------------

def build_detector(args) -> BaseDetector:
    ext = os.path.splitext(args.model)[1].lower()
    backend = args.backend
    if backend == "auto":
        backend = "trt" if ext in (".engine", ".plan") else "ort"
    if backend == "trt":
        return TrtDetector(args.model, imgsz=args.imgsz)
    return OnnxDetector(args.model, device=args.device, imgsz=args.imgsz)


def main() -> None:
    ap = argparse.ArgumentParser(description="通用 ONNX/TensorRT 自动标注 (detect/pose → labelme JSON)")
    ap.add_argument("--model", "--onnx", dest="model", required=True,
                    help="模型路径: .onnx / .engine")
    ap.add_argument("--backend", choices=["auto", "ort", "trt"], default="auto",
                    help="推理后端 (默认按扩展名自动)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="图片目录 (递归)")
    src.add_argument("--video", help="视频路径")
    ap.add_argument("--annotate", dest="annotate", action="store_true", default=True,
                    help="写 labelme JSON (默认开)")
    ap.add_argument("--no-annotate", dest="annotate", action="store_false",
                    help="只推理/预览, 不写 JSON")
    ap.add_argument("--policy", choices=["overwrite", "skip", "append"], default="skip",
                    help="已有 JSON 处理策略 (默认 skip)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300, help="每帧最多检出数量 (默认 300)")
    ap.add_argument("--imgsz", type=int, default=640, help="动态输入模型时的推理尺寸")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--save-vis", action="store_true", help="图片模式: 存带框预览到 images/vis/")
    ap.add_argument("--save-video", action="store_true", help="视频模式: 存带框预览 mp4")
    ap.add_argument("--out", default=None, help="视频模式: 预览 mp4 输出路径")
    ap.add_argument("--frames-dir", default=None, help="视频模式: 抽帧+JSON 输出目录")
    ap.add_argument("--frame-step", type=int, default=1, help="视频模式: 每 N 帧存一帧 (默认 1)")
    ap.add_argument("--gui", action="store_true",
                    help="显示 OpenCV 实时窗口 (空格暂停, Enter单步, q退出)")
    ap.add_argument("--delay", type=int, default=800,
                    help="图片模式 GUI: 每张自动前进毫秒数 (默认 800)")
    args = ap.parse_args()

    det = build_detector(args)
    print(f"模型: {args.model}")
    print(det.info())

    if args.images:
        run_images(det, args)
    else:
        run_video(det, args)


if __name__ == "__main__":
    main()
