# -*- coding: utf-8 -*-
"""
MLP装甲板分类器 - 支持ONNX和PyTorch格式
自动识别模型格式、输入尺寸、预处理参数
从YOLO Pose关键点提取ROI并进行分类
"""
import os
import time
import json
import numpy as np
import cv2


class MLPClassifier:
    ROI_LIGHT_LEN = 12
    ROI_PERSP_H = 28
    ROI_PERSP_W = 32
    ROI_CROP_W = 16
    ROI_EXPAND_RATIO = 1.3
    ROI_CENTER_RATIO = 0.5

    @classmethod
    def set_roi_params(cls, light_len=None, persp_h=None, persp_w=None, crop_w=None, expand_ratio=None, center_ratio=None):
        if light_len is not None:
            cls.ROI_LIGHT_LEN = light_len
        if persp_h is not None:
            cls.ROI_PERSP_H = persp_h
        if persp_w is not None:
            cls.ROI_PERSP_W = persp_w
        if crop_w is not None:
            cls.ROI_CROP_W = crop_w
        if expand_ratio is not None:
            cls.ROI_EXPAND_RATIO = expand_ratio
        if center_ratio is not None:
            cls.ROI_CENTER_RATIO = center_ratio

    @classmethod
    def load_roi_params_from_settings(cls):
        from PySide6.QtCore import QSettings
        s = QSettings("ArmorLabel", "settings")
        s.beginGroup("Parameters")
        cls.ROI_LIGHT_LEN = int(s.value("MLP_ROI_LIGHT_LEN", 12))
        cls.ROI_PERSP_H = int(s.value("MLP_ROI_PERSP_H", 28))
        cls.ROI_PERSP_W = int(s.value("MLP_ROI_PERSP_W", 32))
        cls.ROI_CROP_W = int(s.value("MLP_ROI_CROP_W", 16))
        cls.ROI_EXPAND_RATIO = float(s.value("MLP_ROI_EXPAND_RATIO", 1.3))
        cls.ROI_CENTER_RATIO = float(s.value("MLP_ROI_CENTER_RATIO", 0.5))
        s.endGroup()

    def __init__(self, model_path, device='cpu'):
        """
        加载MLP分类模型

        Args:
            model_path: 模型文件路径 (.onnx / .pt / .pth)
            device: 推理设备 ('cpu' 或 'cuda')
        """
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path) or '.'
        self.device = device
        self.model = None
        self.session = None
        self.model_type = None
        self.input_size = None
        self.input_name = None
        self.is_nchw = True
        self.input_channels = 3
        self.is_fp16 = False
        self.labels = None
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._load_preprocess_config()
        self._load_labels()
        self.num_classes = len(self.labels) if self.labels else None
        self.model_type = self._detect_format()
        if self.model_type == 'onnx':
            self._load_onnx()
        else:
            self._load_pytorch()

    def _detect_format(self):
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == '.onnx':
            return 'onnx'
        elif ext in ('.pt', '.pth'):
            return 'pytorch'
        raise ValueError(f"不支持的模型格式: {ext}，仅支持 .onnx / .pt / .pth")

    def _load_labels(self):
        label_path = os.path.join(self.model_dir, 'label.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                self.labels = [line.strip() for line in f if line.strip()]
            return

        label_path2 = os.path.join(self.model_dir, 'labels.txt')
        if os.path.exists(label_path2):
            with open(label_path2, 'r', encoding='utf-8') as f:
                self.labels = [line.strip() for line in f if line.strip()]
            return

        self.labels = None

    def _load_preprocess_config(self):
        """
        从 model_dir/preprocess.json 加载预处理参数。

        格式示例:
        {
            "mean": [0.485, 0.456, 0.406],   // ImageNet mean (RGB顺序)
            "std":  [0.229, 0.224, 0.225],  // ImageNet std  (RGB顺序)
            "input_size": [224, 128],         // [W, H] 宽度在前，高度在后
            "channels": 1                      // 通道数 (1=灰度, 3=RGB)
        }
        """
        cfg_path = os.path.join(self.model_dir, 'preprocess.json')
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if 'mean' in cfg:
                    self.mean = np.array(cfg['mean'], dtype=np.float32)
                if 'std' in cfg:
                    self.std = np.array(cfg['std'], dtype=np.float32)
                if 'input_size' in cfg:
                    self.input_size = tuple(cfg['input_size'])
                if 'channels' in cfg:
                    self.input_channels = int(cfg['channels'])
            except Exception:
                pass

    def _load_onnx(self):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("缺少 onnxruntime 库，请运行: pip install onnxruntime")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        input_shape = input_info.shape

        def _parse_dim(dim):
            if isinstance(dim, (int, float)) and dim > 0:
                return int(dim)
            return None

        inp_type = str(input_info.type or '').lower()
        self.is_fp16 = 'float16' in inp_type or ('16' in inp_type and 'float' in inp_type)

        has_dynamic = any(_parse_dim(d) is None for d in input_shape if not isinstance(d, str))

        if has_dynamic and self.input_size is None:
            self.input_size = (224, 224)
            self.input_channels = 3
            self.is_nchw = True
        elif len(input_shape) == 4:
            c1, c4 = int(input_shape[1]), int(input_shape[3])
            if c1 > 3 and c1 != 1:
                raise ValueError(
                    f"不支持的ONNX输入形状 {input_shape}。"
                    f"dim1={c1}不是有效的图像通道数(应为1或3)。"
                    "该模型可能是特征提取MLP而非图像分类器。"
                )
            if c1 in (1, 3):
                self.input_channels = c1
                self.is_nchw = True
                d_h = _parse_dim(input_shape[2])
                d_w = _parse_dim(input_shape[3])
                self.input_size = (d_w or 224, d_h or 224)
            elif c4 in (1, 3):
                self.input_channels = c4
                self.is_nchw = False
                d_h = _parse_dim(input_shape[1])
                d_w = _parse_dim(input_shape[2])
                self.input_size = (d_w or 224, d_h or 224)
        elif len(input_shape) == 3:
            d_h = _parse_dim(input_shape[1])
            d_w = _parse_dim(input_shape[2])
            self.input_size = (d_w or 224, d_h or 224)
            self.input_channels = 3
            self.is_nchw = False
        else:
            self.input_size = (224, 224)
            self.input_channels = 3
            self.is_nchw = True

        if self.input_channels > 3:
            raise ValueError(
                f"ONNX模型输入通道数={self.input_channels}，非图像分类器。"
                "仅支持1通道(灰度)或3通道(RGB)输入。"
            )

        output_info = self.session.get_outputs()[0]
        output_shape = output_info.shape
        if len(output_shape) >= 2:
            num_classes = output_shape[-1]
        else:
            num_classes = output_shape[-1] if len(output_shape) == 1 else output_shape[0]

        if self.labels is None:
            if isinstance(num_classes, int) and num_classes > 0:
                self.labels = [f"class_{i}" for i in range(num_classes)]
            else:
                self.labels = [f"class_{i}" for i in range(self.num_classes or 9)]
        else:
            stored = len(self.labels)
            if isinstance(num_classes, int) and num_classes != stored:
                print(f"[MLPClassifier] 警告: label.txt有{stored}个标签，"
                      f"但模型输出{num_classes}类。已用class_i覆盖。")
                self.labels = [f"class_{i}" for i in range(num_classes)]

    def _load_pytorch(self):
        import torch
        import torch.nn as nn
        from core.safe_torch import torch_load_safe

        checkpoint = torch_load_safe(self.model_path, map_location=self.device)

        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint.to(self.device)
        elif isinstance(checkpoint, dict):
            self.model = self._rebuild_pytorch_model(checkpoint)
        else:
            raise ValueError("无法识别的PyTorch模型格式")

        self.model.eval()

        if self.input_size is None:
            self.input_size = self._detect_pytorch_input_size()

        num_classes = self._detect_pytorch_num_classes()

        if self.labels is None:
            if num_classes:
                self.labels = [f"class_{i}" for i in range(num_classes)]
            else:
                self.labels = [f"class_{i}" for i in range(self.num_classes or 9)]
        else:
            if num_classes and num_classes != len(self.labels):
                print(f"[MLPClassifier] 警告: label.txt有{len(self.labels)}个标签，"
                      f"但模型输出{num_classes}类。已用class_i覆盖。")
                self.labels = [f"class_{i}" for i in range(num_classes)]

    def _rebuild_pytorch_model(self, state_dict):
        import torch.nn as nn

        keys = list(state_dict.keys())
        if not keys:
            raise ValueError("PyTorch state_dict 为空")

        first_key = keys[0]

        if first_key.startswith('model.'):
            prefix = 'model.'
        elif first_key.startswith('classifier.'):
            prefix = 'classifier.'
        else:
            prefix = ''

        if any('features' in k for k in keys):
            try:
                from torchvision.models import mobilenet_v3_small
                model = mobilenet_v3_small(weights=None)

                num_classes = self.num_classes or 9
                first_conv_in_channels = 3
                for k in keys:
                    no_prefix = k.replace(prefix, '')
                    if 'classifier.' in no_prefix and 'weight' in no_prefix:
                        w = state_dict[k]
                        if w.ndim >= 2 and w.shape[0] > 1 and w.shape[0] < 1000:
                            num_classes = w.shape[0]
                            break
                    if 'features.0.0.weight' in no_prefix or ('features.0' in no_prefix and 'weight' in no_prefix):
                        w = state_dict[k]
                        if w.ndim == 4 and w.shape[1] == 1:
                            first_conv_in_channels = 1

                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

                if first_conv_in_channels == 1:
                    old_conv = model.features[0][0]
                    new_conv = nn.Conv2d(1, old_conv.out_channels,
                                       kernel_size=old_conv.kernel_size,
                                       stride=old_conv.stride,
                                       padding=old_conv.padding,
                                       bias=(old_conv.bias is not None))
                    model.features[0][0] = new_conv

                sd = {}
                for k, v in state_dict.items():
                    new_k = k.replace(prefix, '')
                    sd[new_k] = v
                model.load_state_dict(sd)
                return model
            except Exception:
                pass

        if any(k.startswith('fc.') or k.startswith('classifier.0') for k in keys):
            num_features = None
            num_classes = self.num_classes or 9
            for k in keys:
                no_prefix = k.replace(prefix, '')
                if 'fc.' in no_prefix or 'classifier.0' in no_prefix:
                    for dim in state_dict[k].shape:
                        if dim > 10:
                            num_features = dim
                            break
                    if num_features:
                        break
            for k in keys:
                no_prefix = k.replace(prefix, '')
                if 'classifier.' in no_prefix and 'weight' in no_prefix:
                    w = state_dict[k]
                    if w.ndim >= 2 and w.shape[0] > 1 and w.shape[0] < 1000:
                        num_classes = w.shape[0]
                        break
            if num_features:
                hidden = 512
                model = nn.Sequential(
                    nn.Linear(num_features, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden, num_classes)
                )
                sd = {}
                for k, v in state_dict.items():
                    new_k = k.replace(prefix, '')
                    sd[new_k] = v
                model.load_state_dict(sd)
                return model

        if any(k.startswith('conv1.') or k.startswith('fc1.') for k in keys):
            num_classes = None
            flat_dim = None
            for k in keys:
                if 'fc2.' in k and 'weight' in k:
                    num_classes = state_dict[k].shape[0]
                if 'fc1.' in k and 'weight' in k:
                    flat_dim = state_dict[k].shape[1]
            if num_classes is None or flat_dim is None:
                raise ValueError(
                    f"无法从 state_dict 推断新架构参数。num_classes={num_classes}, flat_dim={flat_dim}"
                )
            from torch import nn as nn_torch
            import torch.nn.functional as F
            class NumberClassifier(nn_torch.Module):
                def __init__(self, num_classes=8):
                    super().__init__()
                    self.conv1 = nn_torch.Conv2d(1, 8, kernel_size=3, padding=1)
                    self.conv2 = nn_torch.Conv2d(8, 16, kernel_size=3, padding=1)
                    self.pool = nn_torch.MaxPool2d(2, 2)
                    self.fc1 = nn_torch.Linear(flat_dim, 32)
                    self.dropout = nn_torch.Dropout(0.2)
                    self.fc2 = nn_torch.Linear(32, num_classes)

                def _forward_features(self, x):
                    x = self.pool(F.relu(self.conv1(x)))
                    x = self.pool(F.relu(self.conv2(x)))
                    return x

                def forward(self, x):
                    x = self._forward_features(x)
                    x = x.view(x.size(0), -1)
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x
            model = NumberClassifier(num_classes=num_classes)
            sd = {}
            for k, v in state_dict.items():
                new_k = k.replace(prefix, '')
                sd[new_k] = v
            model.load_state_dict(sd)
            return model

        raise ValueError(
            f"无法从 state_dict 重建模型。第一个 key: {first_key}。"
            "请将模型导出为 ONNX 格式以获得最佳兼容性。"
        )

    def _detect_pytorch_input_size(self):
        import torch
        test_sizes = [(28, 20), (20, 28), (64, 64), (96, 96), (128, 128), (224, 224), (192, 192)]
        for channels in [1, 3]:
            for h, w in test_sizes:
                try:
                    dummy = torch.randn(1, channels, h, w).to(self.device)
                    with torch.no_grad():
                        _ = self.model(dummy)
                    self.input_channels = channels
                    return (w, h)
                except Exception:
                    continue
        return (64, 64)

    def _detect_pytorch_num_classes(self):
        import torch.nn as nn
        candidates = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                if module.out_features > 1 and module.out_features < 1000:
                    candidates.append(module.out_features)
        return candidates[-1] if candidates else None

    def predict(self, image_bgr):
        """
        对单张裁剪后的装甲板图片进行分类推理

        Args:
            image_bgr: OpenCV BGR图像（已裁剪为装甲板区域）
                       也支持已灰度化的2D图像

        Returns:
            (class_name, confidence, inference_ms)
        """
        if image_bgr is None or image_bgr.size == 0:
            return "?", 0.0, 0.0

        input_w, input_h = self.input_size

        if self.input_channels == 1:
            n_ch = image_bgr.shape[2] if image_bgr.ndim == 3 else None
            if n_ch is None:
                img_gray = image_bgr
            elif n_ch == 1:
                img_gray = image_bgr[:, :, 0]
            else:
                img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (input_w, input_h))
            img_float = img_resized.astype(np.float32) / 255.0
            mean_val = float(self.mean[0]) if len(self.mean) >= 1 else 0.485
            std_val = float(self.std[0]) if len(self.std) >= 1 else 0.229
            img_normalized = (img_float - mean_val) / std_val
            img_normalized = img_normalized.astype(np.float32)
            if self.is_nchw:
                input_data = img_normalized[np.newaxis, np.newaxis, :, :]
            else:
                input_data = img_normalized[:, :, np.newaxis]
                input_data = np.expand_dims(input_data, axis=0)
        else:
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (input_w, input_h))
            img_float = img_resized.astype(np.float32) / 255.0
            img_normalized = (img_float - self.mean) / self.std
            if self.is_nchw:
                img_transposed = np.transpose(img_normalized, (2, 0, 1))
                input_data = np.expand_dims(img_transposed, axis=0).astype(np.float32)
            else:
                input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)

        if self.is_fp16:
            input_data = input_data.astype(np.float16)

        t0 = time.time()

        if self.model_type == 'onnx':
            outputs = self.session.run(None, {self.input_name: input_data})
            raw = np.array(outputs[0][0], dtype=np.float32)
            if raw.ndim == 0:
                raw = raw.reshape(1)
            x_max = np.max(raw)
            e_x = np.exp(raw - x_max)
            probabilities = (e_x / np.sum(e_x)).astype(np.float32)
        else:
            import torch
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_data).to(self.device)
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                probabilities = probs.cpu().numpy()[0]

        mlp_ms = (time.time() - t0) * 1000

        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        if 0 <= class_id < len(self.labels):
            class_name = self.labels[class_id]
        else:
            class_name = f"class_{class_id}"

        return class_name, confidence, mlp_ms

    def diagnose(self):
        """诊断 MLP 模型加载和推理配置是否正确。"""
        print(f"\n{'='*50}")
        print(f"MLPClassifier 诊断报告")
        print(f"{'='*50}")
        print(f"模型路径: {self.model_path}")
        print(f"模型类型: {self.model_type}")
        print(f"输入通道: {self.input_channels} {'(灰度)' if self.input_channels == 1 else '(RGB)'}")
        print(f"输入精度: {'float16' if self.is_fp16 else 'float32'}")
        print(f"输入尺寸 (W×H): {self.input_size}")
        print(f"输入格式: {'NCHW' if self.is_nchw else 'NHWC'}")
        print(f"设备: {self.device}")
        print(f"类别数: {len(self.labels)}")
        print(f"类别列表: {self.labels}")
        print(f"Mean: {self.mean}")
        print(f"Std:  {self.std}")

        if self.input_channels == 1:
            synthetic = np.random.randint(50, 200, (self.input_size[1], self.input_size[0]), dtype=np.uint8)
        else:
            synthetic = np.random.randint(50, 200, (self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        name, conf, ms = self.predict(synthetic)
        ch_str = "灰度" if self.input_channels == 1 else "RGB"
        print(f"\n合成图像测试 ({ch_str}随机噪声, {self.input_size[0]}x{self.input_size[1]}):")
        print(f"  预测结果: {name} | 置信度: {conf:.4f} | 推理耗时: {ms:.2f}ms")
        print(f"{'='*50}\n")
        return name, conf, ms

    @staticmethod
    def sort_keypoints(kpts):
        """
        仿照 carcode sortKeypoints: 按 x 分左右，每组按 y 分上下
        输入: [4, 2] 关键点
        输出: [左上, 左下, 右下, 右上]
        """
        kpts_arr = np.array(kpts, dtype=np.float32).reshape(-1, 2)
        if len(kpts_arr) != 4:
            return kpts_arr
        sorted_by_x = kpts_arr[np.argsort(kpts_arr[:, 0])]
        left_points = sorted_by_x[:2]
        right_points = sorted_by_x[2:]
        left_sorted = left_points[np.argsort(left_points[:, 1])]
        right_sorted = right_points[np.argsort(right_points[:, 1])]
        return np.array([left_sorted[0], left_sorted[1], right_sorted[1], right_sorted[0]], dtype=np.float32)

    @staticmethod
    def _get_light_from_contour(cnt):
        """
        仿照 carcode Light::getLightPointsByEllipse
        对轮廓进行椭圆拟合，返回 top 和 bottom 端点
        top = 椭圆4角按y排序后上面两角的中点
        bottom = 椭圆4角按y排序后下面两角的中点
        """
        try:
            ellipse = cv2.fitEllipse(cnt)
        except Exception:
            return None, None
        try:
            ell_pts = cv2.boxPoints(ellipse)
        except Exception:
            return None, None
        ell_pts = np.array(ell_pts, dtype=np.float32).reshape(4, 2)
        sorted_pts = ell_pts[np.argsort(ell_pts[:, 1])]
        top = (sorted_pts[0] + sorted_pts[1]) / 2.0
        bottom = (sorted_pts[2] + sorted_pts[3]) / 2.0
        return top, bottom

    @staticmethod
    def _get_matched_light(traditional_lights, yolo_light_top, yolo_light_bottom):
        """
        仿照 carcode getMatchedLight: 找误差最小的传统灯条
        error = (||候选.top - 灯条.top|| + ||候选.bottom - 灯条.bottom||) / (2 * 灯条.length)
        """
        min_error = 1e9
        best_light = None
        yolo_light_length = np.linalg.norm(yolo_light_bottom - yolo_light_top)
        if yolo_light_length < 1e-6:
            return None, 1e9
        for light in traditional_lights:
            if light['length'] < 1e-6:
                continue
            error = (np.linalg.norm(light['top'] - yolo_light_top) +
                     np.linalg.norm(light['bottom'] - yolo_light_bottom)) / (2.0 * light['length'])
            if error < min_error:
                min_error = error
                best_light = light
        return best_light, min_error

    @staticmethod
    def refine_keypoints_and_clean(image, keypoints_xy, bbox_xyxy):
        """
        仿照 carcode sortKeypoints + getVerifiedPosition + getMatchedLight
        传统视觉修正装甲板关键点

        流程:
        1. sortKeypoints 确保输入 kpts 顺序正确 [左上, 左下, 右下, 右上]
        2. 从 kpts 构造 left_light(kp0, kp1) 和 right_light(kp3, kp2)
        3. 在 bbox ROI 中 HSV 颜色分割 → 找轮廓
        4. 对每个轮廓: fitEllipse → top/bottom → 构造传统 Light
        5. 筛选: 宽高比 0.03~0.3, 倾斜角 < 40°
        6. getMatchedLight: 找最匹配的灯条
        7. 检查两灯条长度比 >= 0.7
        8. 返回修正后 kpts [left.top, left.bottom, right.bottom, right.top]
        9. sortKeypoints 再次确保顺序

        Args:
            image: 原始 BGR 图像
            keypoints_xy: YOLO 输出的 4 个关键点 [4, 2]
            bbox_xyxy: 装甲板边框 [x1, y1, x2, y2]

        Returns:
            (corrected_kpts, lights_info):
                corrected_kpts: 修正后的 4 个关键点 [4, 2]，或 None（修正失败）
                lights_info: 匹配到的灯条信息列表
        """
        if image is None or image.size == 0 or keypoints_xy is None or len(keypoints_xy) < 4:
            return None, []

        h_img, w_img = image.shape[:2]
        kpts = np.array(keypoints_xy, dtype=np.float32).reshape(4, 2)
        kpts = MLPClassifier.sort_keypoints(kpts)

        MIN_LIGHT_RATIO = 0.03
        MAX_LIGHT_RATIO = 0.3
        MAX_LIGHT_ANGLE = 40.0

        bbox = np.array(bbox_xyxy, dtype=np.float32)
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[2]), int(bbox[3])
        bw = (x2 - x1) * 1.8 / 2.0
        bh = (y2 - y1) * 1.8 / 2.0
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        r_x1 = max(int(cx - bw), 0)
        r_y1 = max(int(cy - bh), 0)
        r_x2 = min(int(cx + bw), w_img)
        r_y2 = min(int(cy + bh), h_img)
        if r_x2 <= r_x1 or r_y2 <= r_y1:
            return None, []

        roi = image[r_y1:r_y2, r_x1:r_x2]
        if roi.size == 0:
            return None, []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, np.array([0, 80, 80], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
        mask_red2 = cv2.inRange(hsv, np.array([170, 80, 80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, np.array([100, 80, 80], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8))
        mask_bars = cv2.bitwise_or(mask_red, mask_blue)

        contours, _ = cv2.findContours(mask_bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        traditional_lights = []
        for cnt in contours:
            if len(cnt) < 5:
                continue
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            try:
                rect = cv2.minAreaRect(cnt)
                w_rect, h_rect = rect[1]
                if w_rect < 1 or h_rect < 1:
                    continue
                ratio = min(w_rect, h_rect) / max(w_rect, h_rect)
                if ratio < MIN_LIGHT_RATIO or ratio > MAX_LIGHT_RATIO:
                    continue
                top, bottom = MLPClassifier._get_light_from_contour(cnt)
                if top is None:
                    continue
                top_img = top + np.array([r_x1, r_y1], dtype=np.float32)
                bottom_img = bottom + np.array([r_x1, r_y1], dtype=np.float32)
                light_length = np.linalg.norm(bottom_img - top_img)
                direction = top_img - bottom_img
                tilt_angle = np.abs(np.arctan2(np.abs(direction[0]), np.abs(direction[1])))
                if tilt_angle > MAX_LIGHT_ANGLE * np.pi / 180.0:
                    continue
                traditional_lights.append({
                    'top': top_img,
                    'bottom': bottom_img,
                    'center': (top_img + bottom_img) / 2.0,
                    'length': light_length
                })
            except Exception:
                continue

        if len(traditional_lights) < 2:
            return None, []

        best_left, left_error = MLPClassifier._get_matched_light(traditional_lights, kpts[0], kpts[1])
        best_right, right_error = MLPClassifier._get_matched_light(traditional_lights, kpts[3], kpts[2])

        if best_left is None or best_right is None:
            return None, []

        matched_lights = [best_left, best_right]
        length_ratio = min(best_left['length'], best_right['length']) / max(best_left['length'], best_right['length'])
        if length_ratio < 0.7:
            return None, []

        corrected = np.array([
            best_left['top'],
            best_left['bottom'],
            best_right['bottom'],
            best_right['top']
        ], dtype=np.float32)

        corrected = MLPClassifier.sort_keypoints(corrected)
        lights_info = [(light['top'], light['bottom']) for light in matched_lights]
        return corrected, lights_info

    def extract_armor_roi(image, bbox_xyxy, keypoints_xy, output_size=None, corrected_kpts=None, binarize=True):
        """
        仿照 carcode getPerspectiveROI: 透视变换提取装甲板ROI

        透视变换目标: 灯条只占中间部分 (light_len, h, w)
        然后中心裁剪到 output_size (20x28 for number_classifier.onnx)

        Args:
            image: 原始BGR图像（全图）
            bbox_xyxy: [x1, y1, x2, y2] 装甲板边框（未使用，保留参数兼容性）
            keypoints_xy: shape [4, 2] 四个关键点
            output_size: (w, h) 目标尺寸，默认 (20, 28) 匹配 number_classifier.onnx
            corrected_kpts: 修正后的关键点（来自传统视觉修正）
            binarize: 是否对ROI进行二值化

        Returns:
            roi: 透视变换+裁剪后的灰度二值化图像 (H×W)，或 None
        """
        if image is None or image.size == 0:
            return None

        if keypoints_xy is None or len(keypoints_xy) < 4:
            return None

        kpts = np.array(keypoints_xy, dtype=np.float32).reshape(4, 2)
        if corrected_kpts is not None:
            kpts = np.array(corrected_kpts, dtype=np.float32).reshape(4, 2)

        if len(kpts) != 4:
            return None

        kpts = MLPClassifier.sort_keypoints(kpts)

        LIGHT_LEN = MLPClassifier.ROI_LIGHT_LEN
        H = MLPClassifier.ROI_PERSP_H
        W = MLPClassifier.ROI_PERSP_W
        CROP_W = MLPClassifier.ROI_CROP_W

        if output_size is None:
            dst_w, dst_h = 20, 28
        else:
            dst_w, dst_h = output_size

        top_light_y = (H - LIGHT_LEN) // 2 - 1
        bottom_light_y = top_light_y + LIGHT_LEN

        src_pts = kpts.astype(np.float32)
        dst_pts = np.array([
            [0.0, float(top_light_y)],
            [0.0, float(bottom_light_y)],
            [float(W - 1), float(bottom_light_y)],
            [float(W - 1), float(top_light_y)]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

        crop_x1 = (W - CROP_W) // 2
        crop_x2 = crop_x1 + CROP_W
        roi = warped[:, crop_x1:crop_x2]

        roi = cv2.resize(roi, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        if binarize:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            roi = binary

        return roi

    @staticmethod
    def extract_armor_roi_debug(image, bbox_xyxy, keypoints_xy, output_size=None, corrected_kpts=None, binarize=True):
        """
        调试版本：返回透视变换各阶段图像（仿照 carcode getPerspectiveROI）

        Returns:
            dict with keys: 'before_persp', 'after_persp', 'after_crop', 'after_binary'
            每个value是BGR图像，或None
        """
        result = {
            'before_persp': None,
            'after_persp': None,
            'after_crop': None,
            'after_binary': None,
        }

        if image is None or image.size == 0:
            return result

        if keypoints_xy is None or len(keypoints_xy) < 4:
            return result

        kpts = np.array(keypoints_xy, dtype=np.float32).reshape(4, 2)
        if corrected_kpts is not None:
            use_kpts = np.array(corrected_kpts, dtype=np.float32).reshape(4, 2)
        else:
            use_kpts = kpts
        use_kpts = MLPClassifier.sort_keypoints(use_kpts)

        stage_before = image.copy()

        kpt_names = ['kp0(TL)', 'kp1(BL)', 'kp2(BR)', 'kp3(TR)']
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for i, (pt, col, name) in enumerate(zip(use_kpts, colors, kpt_names)):
            cv2.circle(stage_before, (int(pt[0]), int(pt[1])), 8, col, -1)
            cv2.putText(stage_before, name, (int(pt[0])+10, int(pt[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        for i in range(4):
            next_i = (i + 1) % 4
            cv2.line(stage_before,
                     (int(use_kpts[i,0]), int(use_kpts[i,1])),
                     (int(use_kpts[next_i,0]), int(use_kpts[next_i,1])),
                     (0, 255, 0), 2)

        x1, y1 = int(bbox_xyxy[0]), int(bbox_xyxy[1])
        x2, y2 = int(bbox_xyxy[2]), int(bbox_xyxy[3])
        cv2.rectangle(stage_before, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(stage_before, f'bbox [{x1},{y1},{x2},{y2}]',
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        result['before_persp'] = stage_before

        LIGHT_LEN = MLPClassifier.ROI_LIGHT_LEN
        H = MLPClassifier.ROI_PERSP_H
        W = MLPClassifier.ROI_PERSP_W
        CROP_W = MLPClassifier.ROI_CROP_W

        if output_size is None:
            dst_w, dst_h = 20, 28
        else:
            dst_w, dst_h = output_size

        top_light_y = (H - LIGHT_LEN) // 2 - 1
        bottom_light_y = top_light_y + LIGHT_LEN

        src_pts = use_kpts.astype(np.float32)
        dst_pts = np.array([
            [0.0, float(top_light_y)],
            [0.0, float(bottom_light_y)],
            [float(W - 1), float(bottom_light_y)],
            [float(W - 1), float(top_light_y)]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        result['after_persp'] = warped.copy()

        crop_x1 = (W - CROP_W) // 2
        crop_x2 = crop_x1 + CROP_W
        roi = warped[:, crop_x1:crop_x2].copy()
        result['after_crop'] = roi.copy()

        roi = cv2.resize(roi, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        if binarize:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            roi_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            result['after_binary'] = roi_binary.copy()
            roi_final = binary
        else:
            result['after_binary'] = None
            roi_final = roi

        if len(roi_final.shape) == 2:
            result['final'] = cv2.cvtColor(roi_final, cv2.COLOR_GRAY2BGR)
        else:
            result['final'] = roi_final
        return result

    @staticmethod
    def extract_armor_roi_bbox(image, bbox_xyxy, expand_ratio=None, output_size=None, center_ratio=None, binarize=True):
        """
        从YOLO Pose检测结果中提取装甲板ROI（bbox矩形裁剪 + 中心裁剪 + 二值化）

        Args:
            image: 原始BGR图像（全图）
            bbox_xyxy: [x1, y1, x2, y2] 装甲板边框
            expand_ratio: ROI扩展比例 (1.3 = 从中心扩大30%)
            output_size: (w, h) 目标尺寸
            center_ratio: 保留中心比例 (0.5 = 左右各裁1/4，保留中间50%)
            binarize: 是否对ROI进行二值化

        Returns:
            roi_bgr: 裁剪后的BGR图像，或None（边框无效时）
        """
        if expand_ratio is None:
            expand_ratio = MLPClassifier.ROI_EXPAND_RATIO
        if center_ratio is None:
            center_ratio = MLPClassifier.ROI_CENTER_RATIO

        if image is None or image.size == 0:
            return None

        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * expand_ratio / 2.0
        bh = (y2 - y1) * expand_ratio / 2.0

        x1e = max(int(cx - bw), 0)
        y1e = max(int(cy - bh), 0)
        x2e = min(int(cx + bw), w_img)
        y2e = min(int(cy + bh), h_img)

        if x2e <= x1e or y2e <= y1e:
            return None

        roi = image[y1e:y2e, x1e:x2e]
        if roi.size == 0:
            return None

        if center_ratio < 1.0:
            h_roi, w_roi = roi.shape[:2]
            margin = int(w_roi * (1.0 - center_ratio) / 2.0)
            left = margin
            right = w_roi - margin
            if right > left:
                roi = roi[:, left:right, :]
            if roi.size == 0:
                return None

        if binarize:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            roi = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if output_size is not None:
            roi = cv2.resize(roi, output_size, interpolation=cv2.INTER_LINEAR)

        return roi
