# `scripts/` — 常用批处理脚本（非 GUI 主程序）

主功能入口是 `main.py` 与 `tools/` 下的窗口工具。  
本目录只保留与 **YOLO / MLP / 层级标注** 相关的可复用 CLI；一次性本机流水线（RM26 / lazer711 / pillar / ROS2 等）已移除，勿再入库。

| 脚本 | 用途 |
|---|---|
| `03_convert_and_split_yolo.py` | JSON/XML → YOLO 转换并划分（CLI；GUI 可用 `tools/main_dataset_tool.py`） |
| `yolo_2_mlp.py` | YOLO-Pose 数据集 → MLP 分类 ROI 数据集 |
| `mlp_sort_roi.py` | 按 MLP 预测整理 ROI 目录 |
| `augment_for_mlp.py` | 为 MLP 生成增强样本 |
| `test_mlp_conv_pt.py` | 检查 MLP `.pt` 权重能否加载 |
| `merge_pose_json_into_txt.py` | Pose JSON 关键点合并进 YOLO txt |
| `remap_yolo_to_hier_ids.py` | YOLO class_id 重映射为层级 14 类 id |
| `hierarchical_train.py` | 层级 DecoupledEdgeYOLO 训练循环（`tools/hierarchical_train.py` GUI 会调用） |
| `hierarchical_paths.py` | 层级训练默认路径配置（请改成你的本机/相对路径） |
| `env.local.example` | `run.sh` 本地 conda 覆盖示例（复制为 `env.local`，已 gitignore） |

新需求优先做进 `tools/`（继承 `core/base_tool_window.py`），不要再往本目录堆一次性脚本。
