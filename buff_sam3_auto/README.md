# buff_sam3_auto

能量机关（Buff）自动标注可复用模块：

**YOLO ONNX bbox → SAM3 双轮廓（圆盘 + R）→ 极轴几何五点**

## 流程

```
图像
 └─ OrtBBoxDetector (包内 yolo11n_buff.onnx)
      └─ 每个 ROI
           ├─ SAM3: circle 轮廓（dart board）
           ├─ SAM3: R 轮廓
           └─ solve_five_keypoints(R质心, 圆盘轮廓)
                ├─ 极点 = R 质心          → kp0
                ├─ 最近点 / 最远点        → kp1 / kp3（极轴/长轴）
                └─ 垂直最长弦两端         → kp2 / kp4（短轴）
```

## 安装依赖

```text
numpy
opencv-python
onnxruntime-gpu   # 或 onnxruntime
torch
Pillow
# 另需可用的 facebook/sam3 源码或安装包
```

## 快速使用

```python
import cv2
from buff_sam3_auto import BuffAutoStrategy, default_onnx_path

strategy = BuffAutoStrategy(
    onnx_path=default_onnx_path(),          # 可省略，默认包内 ONNX
    sam_checkpoint="D:/path/to/sam3.pt",
    device="cuda",
)
image = cv2.imread("frame.jpg")
annotations, debug = strategy.annotate(image)
lines = BuffAutoStrategy.to_yolo_lines(annotations)
```

命令行批处理：

```bash
python -m buff_sam3_auto.cli ^
  --images D:/data/images ^
  --sam D:/weights/sam3.pt ^
  --out D:/data/labels
```

## 与标注工具集成

`settings/annotator.yaml` 中：

```yaml
buff_strategy:
  enabled: true
  model: buff_sam3_auto/weights/yolo11n_buff.onnx
  sam_model: path/to/sam3.pt
```

`labelme.py` 通过 `from buff_sam3_auto import BuffAutoStrategy` 调用本模块。

## 仅几何复用（无模型）

```python
from buff_sam3_auto import select_circle_contour, solve_five_keypoints
```
