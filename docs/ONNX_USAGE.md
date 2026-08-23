# ONNX 输出使用说明

## 模型信息

- 模型: `best.pt` -> `best.onnx` (YOLOv12n-Pose)
- 任务: `pose` (关键点检测)
- 类别: `0=RT`(红方机关), `1=BT`(蓝方机关)
- 关键点: 5 个, 每关键点 `[x, y, vis]`

---

## ONNX 输入

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `images` | `[1, 3, 640, 640]` | float32 | RGB, 值域 [0,1], letterbox 到 640 |

## ONNX 输出

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `output0` | `[1, 21, 8400]` | float32 | 已解码的 8400 个候选 |

### 21 个通道布局

```
索引     内容
───────────────────────────────────────────
0,1      cx, cy           bbox 中心 (640 空间坐标)
2,3      w, h             bbox 宽高 (640 空间坐标)
4        类别 RT 得分      已 sigmoid
5        类别 BT 得分      已 sigmoid
6,7      kpt0_x, kpt0_y   关键点0 (640 空间坐标)
8        kpt0_vis          关键点0 可见性 (已 sigmoid)
9,10     kpt1_x, kpt1_y   关键点1
11       kpt1_vis
12,13    kpt2_x, kpt2_y   关键点2
14       kpt2_vis
15,16    kpt3_x, kpt3_y   关键点3
17       kpt3_vis
18,19    kpt4_x, kpt4_y   关键点4
20       kpt4_vis
```

8400 来自三个检测层: `80x80 + 40x40 + 20x20` (strides = [8, 16, 32])



## 数据流向总结

```
原图 (WxH)
   │ letterbox: resize 保持宽高比, pad 右下到 640x640
   ▼
[1, 3, 640, 640]  归一化 RGB
   │ ONNX 内部已融合 DFL/解码/sigmoid
   ▼
[1, 21, 8400]
   │ 0:2  -> cx,cy (640 空间)
   │ 2:4  -> w,h   (640 空间)
   │ 4:6  -> cls (已 sigmoid)
   │ 6:21 -> 5 关键点 (已解码, 640 空间, vis 已 sigmoid)
   │
   │ NMS (xywh->xyxy, 过滤, iou=0.7)
   │ 坐标缩放: x_640 / scale -> x_orig
   ▼
最终结果 (原图坐标)
```

## 注意事项

1. **letterbox 必须左上对齐**: `cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, ...)`, 即 上=左=0, 下=pad_h, 右=pad_w。这样 640 坐标除 scale 就直接是原图坐标。
2. **ONNX 输出是 xywh 格式**, NMS 前要转成 xyxy。
3. **vis 阈值**: vis>=0.5 判为可见(2), vis<0.1 判为不可见(0), 中间判为遮挡(1)。
