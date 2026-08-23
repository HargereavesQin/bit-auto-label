# -*- coding: utf-8 -*-
"""
YOLO 推理客户端 - 支持 detect/segment/pose/obb 任务类型
"""
import os
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject, QPointF, QRectF, QSettings
from PySide6.QtGui import QPolygonF

# mix 同款透视 ROI 常量 (Function.cpp getPerspectiveROI)
_ROI_W, _ROI_H, _LIGHT_LEN = 20, 28, 12
_BIG_W, _SMALL_W = 54, 32
# armor_id -> YOLO-pose 类名后缀 (保留 YOLO 颜色前缀 B/R/G)
_ARMOR_ID_TO_SUFFIX = {0: "G", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "O", 7: "B"}


def load_mlp_conv_pt(pt_path, device="cpu"):
    """加载 mlp_conv.pt (MobileNetV3SmallArmor)，自动尝试 width_mult=1.24/1.0。"""
    import torch
    from .mlp_conv import MlpConv, NUM_NETWORK_CLASSES
    from .safe_torch import torch_load_safe

    sd = torch_load_safe(pt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    last_err = None
    for width_mult in (1.24, 1.0):
        try:
            model = MlpConv(num_classes=NUM_NETWORK_CLASSES, width_mult=width_mult)
            model.load_state_dict(sd)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法加载 MobileNet 权重 {pt_path}: {last_err}")


def _sort_keypoints_xy(pts: np.ndarray) -> np.ndarray:
    """[lt, lb, rb, rt]，与 ros2-framework sortKeypoints 一致。"""
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2], pts[2:]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return np.array([left[0], left[1], right[1], right[0]], dtype=np.float32)


def _is_big_armor(pts: np.ndarray) -> bool:
    light_length = float(np.linalg.norm((pts[1] - pts[0]) + (pts[2] - pts[3])))
    armor_width = float(np.linalg.norm((pts[3] - pts[0]) + (pts[2] - pts[1])))
    return armor_width / max(light_length, 1e-6) > 3.2


def get_perspective_roi_gray(image_bgr: np.ndarray, kpts_xy: np.ndarray) -> np.ndarray | None:
    """4 关键点透视矫正 -> 20x28 灰度 (未 OTSU；preprocess_roi_image 内做)。"""
    import cv2

    if image_bgr is None or kpts_xy is None or len(kpts_xy) < 4:
        return None
    pts = _sort_keypoints_xy(kpts_xy[:4])
    w = _BIG_W if _is_big_armor(pts) else _SMALL_W
    top_y = (_ROI_H - _LIGHT_LEN) // 2 - 1
    bottom_y = top_y + _LIGHT_LEN
    dst = np.array(
        [[0, top_y], [0, bottom_y], [w - 1, bottom_y], [w - 1, top_y]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image_bgr, m, (w, _ROI_H))
    x0 = (w - _ROI_W) // 2
    roi = warped[:, x0 : x0 + _ROI_W]
    if roi.size == 0:
        return None
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi


def merge_yolo_label_with_armor_id(yolo_label: str, armor_id: int) -> str:
    """保留 YOLO 颜色前缀，用 MobileNet armor_id 覆盖数字后缀。armor_id=8 不覆盖。"""
    if armor_id not in _ARMOR_ID_TO_SUFFIX:
        return yolo_label
    if not yolo_label:
        return yolo_label
    color = yolo_label[0].upper() if yolo_label[0].upper() in ("B", "R", "G") else None
    if color is None:
        return yolo_label
    return color + _ARMOR_ID_TO_SUFFIX[armor_id]


def classify_roi_with_mlp(mlp_model, roi_gray: np.ndarray, conf_thresh: float = 0.5):
    """返回 (armor_id, confidence)。低置信度或失败返回 (8, 0.0)。"""
    import torch
    import torch.nn.functional as F
    from .mlp_conv import MIX_REMAP_ARMOR_IDS, preprocess_roi_image

    if mlp_model is None or roi_gray is None:
        return 8, 0.0
    try:
        normalized = preprocess_roi_image(roi_gray)
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
        device = next(mlp_model.parameters()).device
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = mlp_model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        net_cls = int(probs.argmax().item())
        conf = float(probs[net_cls].item())
        if conf < conf_thresh:
            return 8, conf
        armor_id = int(MIX_REMAP_ARMOR_IDS[net_cls])
        return armor_id, conf
    except Exception:
        return 8, 0.0


class YoloLoadWorker(QThread):
    loaded = Signal(object, bool, str)

    def __init__(self, weight_path, task=None):
        super().__init__()
        self.weight_path = weight_path
        self.task = task

    def run(self):
        try:
            from core.yolo_runtime import load_yolo_model
            model = load_yolo_model(self.weight_path, task=self.task)
            task = model.task
            names = model.names
            self.loaded.emit(
                model, True,
                f"YOLO 模型加载成功，任务类型: {task}，类别数: {len(names)}"
            )
        except Exception as e:
            self.loaded.emit(None, False, f"模型加载失败: {e}")


class YoloClient(QObject):
    model_status_changed = Signal(bool, str)
    detection_result = Signal(list, str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.task = None
        self.names = None
        self._load_worker = None

    def load_model_async(self, weight_path):
        if self._load_worker and self._load_worker.isRunning():
            return
        self._load_worker = YoloLoadWorker(weight_path)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.start()

    def _on_model_loaded(self, model, success, msg):
        if success:
            self.model = model
            self.task = model.task
            self.names = model.names
        self.model_status_changed.emit(success, msg)

    @staticmethod
    def get_device():
        settings = QSettings("ArmorLabel", "settings")
        settings.beginGroup("Parameters")
        device = str(settings.value("YOLO_DEVICE", "cuda"))
        settings.endGroup()
        return device

    def predict_image(self, image_path, conf=0.25, iou=0.7, imgsz=640):
        if not self.model:
            return []
        results = self.model.predict(
            image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            device=YoloClient.get_device()
        )
        if not results:
            return []
        return self._convert_results(results[0], image_path)

    def _convert_results(self, result, image_path):
        shapes = []
        h, w = result.orig_shape

        if self.task == "detect":
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    cls_id = int(clss[j])
                    label = self.names.get(cls_id, f"class_{cls_id}")
                    shapes.append({
                        "type": "rectangle",
                        "label": label,
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "conf": float(confs[j])
                    })

        elif self.task == "segment":
            boxes = result.boxes
            masks = result.masks
            if masks is not None and len(masks) > 0:
                xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else None
                confs = boxes.conf.cpu().numpy() if boxes is not None else np.ones(len(masks))
                clss = boxes.cls.cpu().numpy() if boxes is not None else np.zeros(len(masks))
                poly_list = masks.xy if hasattr(masks, 'xy') else []
                for j, poly in enumerate(poly_list):
                    if len(poly) < 3:
                        continue
                    points = [[float(pt[0]), float(pt[1])] for pt in poly]
                    cls_id = int(clss[j]) if xyxy is not None else 0
                    label = self.names.get(cls_id, f"class_{cls_id}")
                    shapes.append({
                        "type": "polygon",
                        "label": label,
                        "points": points,
                        "conf": float(confs[j]) if j < len(confs) else 1.0
                    })
                if boxes is not None and len(boxes) > len(poly_list):
                    for j in range(len(boxes)):
                        if j >= len(poly_list):
                            x1, y1, x2, y2 = xyxy[j]
                            cls_id = int(clss[j])
                            label = self.names.get(cls_id, f"class_{cls_id}")
                            shapes.append({
                                "type": "rectangle",
                                "label": label,
                                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                                "conf": float(confs[j])
                            })

        elif self.task == "pose":
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                kp_data = keypoints.xy.cpu().numpy() if keypoints is not None else None
                # 修正: has_visible 不是 ultralytics Keypoints 的标准属性,
                # 用 conf is not None 判定, 否则所有关键点会被强制 visibility=2.
                kp_conf = keypoints.conf.cpu().numpy() if keypoints is not None and keypoints.conf is not None else None

                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    cls_id = int(clss[j])
                    label = self.names.get(cls_id, f"class_{cls_id}")

                    kpts = []
                    kpt_names = []
                    if kp_data is not None and j < len(kp_data):
                        kp_arr = kp_data[j]
                        num_kpts = len(kp_arr)
                        for k in range(num_kpts):
                            kx, ky = float(kp_arr[k][0]), float(kp_arr[k][1])
                            if kp_conf is not None and j < len(kp_conf) and k < len(kp_conf[j]):
                                v = kp_conf[j][k]
                                if v >= 0.5:
                                    visibility = 2
                                elif v >= 0.1:
                                    visibility = 1
                                else:
                                    visibility = 0
                            else:
                                visibility = 2
                            kpts.append([kx, ky, visibility])
                            kpt_names.append(f"kpt_{k}")

                    shapes.append({
                        "type": "pose",
                        "label": label,
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "keypoints": kpts,
                        "kpt_names": kpt_names,
                        "conf": float(confs[j])
                    })

        elif self.task == "obb":
            obb = result.obb
            if obb is not None and len(obb) > 0:
                xywhr = obb.xywhr.cpu().numpy() if hasattr(obb, 'xywhr') else None
                corners = obb.xyxyxyxy.cpu().numpy() if hasattr(obb, 'xyxyxyxy') else None
                confs = obb.conf.cpu().numpy() if hasattr(obb, 'conf') else np.ones(len(obb))
                clss = obb.cls.cpu().numpy() if hasattr(obb, 'cls') else np.zeros(len(obb))

                for j in range(len(obb)):
                    cls_id = int(clss[j])
                    label = self.names.get(cls_id, f"class_{cls_id}")
                    if corners is not None and j < len(corners):
                        pts = corners[j]
                        points = [[float(pt[0]), float(pt[1])] for pt in pts]
                    else:
                        points = []
                    if xywhr is not None and j < len(xywhr):
                        cx, cy, bw, bh, angle = xywhr[j]
                        rect = [float(cx), float(cy), float(bw), float(bh)]
                        angle_deg = float(angle)
                    else:
                        rect = [0, 0, 0, 0]
                        angle_deg = 0
                    shapes.append({
                        "type": "obb",
                        "label": label,
                        "points": points,
                        "rect": rect,
                        "angle": angle_deg,
                        "conf": float(confs[j]) if j < len(confs) else 1.0
                    })

        return shapes


class YoloBatchWorker(QThread):
    progress = Signal(int, int, str)
    image_done = Signal(str, list)
    finished = Signal(bool, str)
    log = Signal(str)

    def __init__(self, model, task, names, image_paths, output_dir,
                 conf, iou, imgsz, policy,
                 dual_mode=False, mlp_model=None, mlp_conf=0.5):
        super().__init__()
        self.model = model
        self.task = task
        self.names = names
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.policy = policy
        # 双层: YOLO bbox/pose + MobileNet 覆盖 class id；默认关闭不影响旧调用
        self.dual_mode = bool(dual_mode) and mlp_model is not None
        self.mlp_model = mlp_model
        self.mlp_conf = float(mlp_conf)
        self._stop = False
        self._cached_image_path = None
        self._cached_image_bgr = None

    def stop(self):
        self._stop = True

    def _get_image_bgr(self, image_path):
        """缓存当前图的 BGR，避免双层模式下每实例重复 imread。"""
        if image_path == self._cached_image_path and self._cached_image_bgr is not None:
            return self._cached_image_bgr
        import cv2
        img = cv2.imread(image_path)
        self._cached_image_path = image_path
        self._cached_image_bgr = img
        return img

    def _get_kpt_names_from_pose_class(self, label):
        from core.pose_config import PoseConfig
        cfg = PoseConfig()
        pose_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "pose_classes.json")
        if os.path.exists(pose_config_path):
            cfg.load(pose_config_path)
        if label in cfg.categories:
            return cfg.categories[label]["keypoints"]
        return None

    def _convert_result_to_shapes(self, result, image_path):
        shapes = []
        h, w = result.orig_shape if hasattr(result, 'orig_shape') else (0, 0)

        if self.task == "detect":
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    cls_id = int(clss[j])
                    label = self.names.get(cls_id, f"class_{cls_id}")
                    shapes.append({
                        "type": "rectangle",
                        "label": label,
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "conf": float(confs[j])
                    })

        elif self.task == "segment":
            boxes = result.boxes
            masks = result.masks
            if masks is not None and len(masks) > 0:
                xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else None
                confs = boxes.conf.cpu().numpy() if boxes is not None else np.ones(len(masks))
                clss = boxes.cls.cpu().numpy() if boxes is not None else np.zeros(len(masks))
                poly_list = masks.xy if hasattr(masks, 'xy') else []
                for j, poly in enumerate(poly_list):
                    if len(poly) < 3:
                        continue
                    points = [[float(pt[0]), float(pt[1])] for pt in poly]
                    cls_id = int(clss[j]) if xyxy is not None else 0
                    label = self.names.get(cls_id, f"class_{cls_id}")
                    shapes.append({
                        "type": "polygon",
                        "label": label,
                        "points": points,
                        "conf": float(confs[j]) if j < len(confs) else 1.0
                    })
                if boxes is not None and len(boxes) > len(poly_list):
                    for j in range(len(boxes)):
                        if j >= len(poly_list):
                            x1, y1, x2, y2 = xyxy[j]
                            cls_id = int(clss[j])
                            label = self.names.get(cls_id, f"class_{cls_id}")
                            shapes.append({
                                "type": "rectangle",
                                "label": label,
                                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                                "conf": float(confs[j])
                            })

        elif self.task == "pose":
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                kp_data = keypoints.xy.cpu().numpy() if keypoints is not None else None
                kp_conf = None
                if keypoints is not None and hasattr(keypoints, 'conf') and keypoints.conf is not None:
                    kp_conf = keypoints.conf.cpu().numpy()

                image_bgr = None
                if self.dual_mode:
                    image_bgr = self._get_image_bgr(image_path)

                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    cls_id = int(clss[j])
                    label = self.names.get(cls_id, f"class_{cls_id}")

                    # 双层: bbox/kpts 听 YOLO, id 听 MobileNet (保留颜色前缀)
                    if self.dual_mode and image_bgr is not None and kp_data is not None and j < len(kp_data):
                        roi = get_perspective_roi_gray(image_bgr, kp_data[j])
                        armor_id, _mlp_c = classify_roi_with_mlp(
                            self.mlp_model, roi, self.mlp_conf
                        )
                        if armor_id != 8:
                            new_label = merge_yolo_label_with_armor_id(label, armor_id)
                            if new_label != label:
                                label = new_label

                    kpts = []
                    kpt_names = self._get_kpt_names_from_pose_class(label)
                    if kpt_names is None:
                        kpt_names = []
                        if kp_data is not None and j < len(kp_data):
                            num_kpts = len(kp_data[j])
                            for k in range(num_kpts):
                                kpt_names.append(f"kpt_{k}")

                    if kp_data is not None and j < len(kp_data):
                        kp_arr = kp_data[j]
                        for k in range(len(kp_arr)):
                            kx, ky = float(kp_arr[k][0]), float(kp_arr[k][1])
                            if kp_conf is not None and j < len(kp_conf) and k < len(kp_conf[j]):
                                v = kp_conf[j][k]
                                if v >= 0.5:
                                    visibility = 2
                                elif v >= 0.1:
                                    visibility = 1
                                else:
                                    visibility = 0
                            else:
                                visibility = 2
                            kpts.append([kx, ky, visibility])

                    shapes.append({
                        "type": "pose",
                        "label": label,
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "keypoints": kpts,
                        "kpt_names": kpt_names,
                        "conf": float(confs[j])
                    })

        elif self.task == "obb":
            obb = result.obb
            if obb is not None and len(obb) > 0:
                xywhr = obb.xywhr.cpu().numpy() if hasattr(obb, 'xywhr') else None
                corners = obb.xyxyxyxy.cpu().numpy() if hasattr(obb, 'xyxyxyxy') else None
                confs = obb.conf.cpu().numpy() if hasattr(obb, 'conf') else np.ones(len(obb))
                clss = obb.cls.cpu().numpy() if hasattr(obb, 'cls') else np.zeros(len(obb))

                for j in range(len(obb)):
                    cls_id = int(clss[j])
                    label = self.names.get(cls_id, f"class_{cls_id}")
                    if corners is not None and j < len(corners):
                        pts = corners[j]
                        points = [[float(pt[0]), float(pt[1])] for pt in pts]
                    else:
                        points = []
                    if xywhr is not None and j < len(xywhr):
                        cx, cy, bw, bh, angle = xywhr[j]
                        rect = [float(cx), float(cy), float(bw), float(bh)]
                        angle_deg = float(angle)
                    else:
                        rect = [0, 0, 0, 0]
                        angle_deg = 0
                    shapes.append({
                        "type": "obb",
                        "label": label,
                        "points": points,
                        "rect": rect,
                        "angle": angle_deg,
                        "conf": float(confs[j]) if j < len(confs) else 1.0
                    })

        return shapes

    def _save_annotation(self, image_path, shapes):
        json_path = os.path.splitext(image_path)[0] + ".json"
        h, w = 0, 0
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
        except Exception:
            pass

        import json
        annotation = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageHeight": h,
            "imageWidth": w
        }
        for s in shapes:
            shape_entry = {
                "label": s["label"],
                "points": s["points"],
                "group_id": None,
                "shape_type": s["type"],
                "flags": {}
            }
            if s["type"] == "obb":
                shape_entry["rect"] = s["rect"]
                shape_entry["angle"] = s["angle"]
            if s["type"] == "pose":
                shape_entry["keypoints"] = s["keypoints"]
                shape_entry["kpt_names"] = s["kpt_names"]
            annotation["shapes"].append(shape_entry)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

    def run(self):
        total = len(self.image_paths)
        self.log.emit(f"开始批量推理，共 {total} 张图片")
        mode_str = "双层(YOLO+MobileNet)" if self.dual_mode else "单层(YOLO)"
        self.log.emit(
            f"参数: conf={self.conf}, iou={self.iou}, imgsz={self.imgsz}, "
            f"task={self.task}, mode={mode_str}"
        )

        for i, img_path in enumerate(self.image_paths):
            if self._stop:
                self.finished.emit(False, "用户取消")
                return

            filename = os.path.basename(img_path)
            self.progress.emit(i + 1, total, filename)
            self.log.emit(f"[{i+1}/{total}] 推理中: {filename}")

            try:
                results = self.model.predict(
                    img_path,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    verbose=False,
                    device=YoloClient.get_device()
                )
                if not results:
                    self.log.emit(f"  结果为空: {filename}")
                    self.image_done.emit(img_path, [])
                    continue

                shapes = self._convert_result_to_shapes(results[0], img_path)

                json_path = os.path.splitext(img_path)[0] + ".json"
                has_json = os.path.exists(json_path)

                if self.policy == "skip" and has_json:
                    self.log.emit(f"  跳过（已存在）: {filename}")
                elif self.policy == "overwrite" or not has_json or self.policy == "append":
                    if self.policy == "append" and has_json:
                        import json
                        with open(json_path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                        existing_shapes = existing.get("shapes", [])
                        shapes = existing_shapes + shapes
                    # 无检测结果：默认不写空 JSON（避免假「已标注」）；overwrite 才清空
                    if not shapes and self.policy != "overwrite":
                        self.log.emit(f"  无检测，跳过写盘: {filename}")
                    else:
                        self._save_annotation(img_path, shapes)
                        self.log.emit(f"  保存 {len(shapes)} 个标注: {filename}")

                self.image_done.emit(img_path, shapes)

            except Exception as e:
                self.log.emit(f"  推理失败 [{filename}]: {e}")
                self.image_done.emit(img_path, [])

        # 写 classes.txt: 按模型 class id 顺序 (0,1,2,...) 落盘,
        # 保证主程序 load_classes 时 class_list 索引 == 模型 class id,
        # 导出 YOLO 标签的 class id 与训练 data.yaml 一致, 不会因
        # os.listdir 任意顺序导致类别名/ID 错位。
        self._write_classes_txt()

        self.finished.emit(True, f"批量推理完成，共处理 {total} 张图片")

    def _write_classes_txt(self):
        """写 classes.txt：严格按模型 class id 顺序，已有文件不覆盖。"""
        if not self.names:
            return
        try:
            class_file = os.path.join(self.output_dir, "classes.txt")
            if os.path.exists(class_file):
                return
            ordered = [self.names[k] for k in sorted(self.names.keys())]
            with open(class_file, "w", encoding="utf-8") as f:
                f.write("\n".join(ordered) + "\n")
            self.log.emit(f"已生成 classes.txt: {ordered}")
        except Exception as e:
            self.log.emit(f"写 classes.txt 失败: {e}")
