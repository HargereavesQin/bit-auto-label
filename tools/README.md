# `tools/` — Standalone Tool 入口

每个文件都是**独立可启动**的 `QMainWindow`,从根目录运行 `python tools/<name>.py`。

启动方式:
```bash
conda activate sam3
python tools/main_dataset_tool.py    # 数据集处理 (YOLO 划分 / 转换 / U-Net)
python tools/mlp_train.py            # MLP / MobileNetV3 数字分类训练
python tools/mlp_demo.py             # MLP 分类推理演示
python tools/yolo_train.py           # YOLO 训练 UI
python tools/yolo_inference.py       # YOLO 推理查看器 (图片/视频)
python tools/yolo_auto_annotate.py   # YOLO 批量自动标注
python tools/video_frame_extractor.py# 视频抽帧
python tools/armor_data_extractor.py # 装甲板数据提取
```

**约定**:
- 每个文件顶部都会注入 `sys.path` 指向项目根, 所以 `from core.xxx import ...` 仍然有效
- 视觉风格统一走 `ui/style_editorial.qss` (编辑风暖灰 + 香槟金)
- 通用基类在 `core/base_tool_window.py` (header / status / log / config 持久化已收敛)
- 旧的根目录 `python main_dataset_tool.py` 启动方式**已弃用**

添加新 tool: 复制任意一个文件,继承 `BaseToolWindow`,只写 `build_ui()` 即可。
