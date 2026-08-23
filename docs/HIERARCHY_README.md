# 层级标注 / Hierarchical Annotation

为 RM 装甲板数据集做 14 类兵种 + L2 子类标注，配套 DecoupledEdgeYOLO 训练。
**与 cpp 推理框架对齐**：14 个 class_id 不变，颜色已并入 class_id。

---

## 类别方案（14 类）

| id | 名称 | 阵营 | L2 子类 | 关键点 |
|----|------|------|---------|------|
| 0  | BG   | 蓝方哨兵 | — | 4 角点 |
| 1  | B1   | 蓝方 | — | 4 角点 |
| 2  | B2   | 蓝方 | — | 4 角点 |
| 3  | B3   | 蓝方 | — | 4 角点 |
| 4  | B4   | 蓝方 | — | 4 角点 |
| 5  | BO   | 蓝方前哨 | up/mid/down | 4 角 + L2 |
| 6  | BB   | 蓝方基地 | — | 4 角点 |
| 7  | RG   | 红方哨兵 | — | 4 角点 |
| 8  | R1   | 红方 | — | 4 角点 |
| 9  | R2   | 红方 | — | 4 角点 |
| 10 | R3   | 红方 | — | 4 角点 |
| 11 | R4   | 红方 | — | 4 角点 |
| 12 | RO   | 红方前哨 | up/mid/down | 4 角 + L2 |
| 13 | RB   | 红方基地 | — | 4 角点 |

- **颜色已编码进 class_id**（B*=蓝，R*=红），不再单独占关键点。
- **L2 只对 BO(5) / RO(12) 两个兵种**生效，3 子类 `up/mid/down`。

## 关键点（5 个，Virtual Keypoint Multiplexing）

| KP | 含义 |
|----|------|
| 0-3 | 四角点 TL/TR/BL/BR（物理点） |
| 4   | L2 子类（虚拟点，visibility 编码：0=none 1=up 2=mid 3=down，坐标复用 bbox 中心） |

- `center` 不单独存（可由 4 角算出），减少冗余。
- L2 虚拟点的 `x,y` 没有物理意义，仅 `visibility` 用于编码子类。

---

## 一、标注

### 1. 进入

```bash
conda activate sam3
python main.py
```

菜单栏 **工具 → 层级标注**，右侧出现层级面板，自动切到 Pose 模式。

### 2. 选类别

- **机器人兵种**：下拉选 0-13（含 `BG` 蓝哨兵 / `RG` 红哨兵）
  - 只有 `BO` / `RO` 显示 L2 子类区，其他兵种 L2 区隐藏
- **L2 子类**（仅 BO/RO）：单选 `无 / up / mid / down`

### 3. 标一个实例

1. 在图上 **拖拽** 画 bbox
2. 依次点击 **4 个角点**：`TL → TR → BL → BR`
3. 若当前兵种为 `BO`/`RO`：放完第 4 个点后，**KP4（L2 子类）自动放在 bbox 中心**，visibility 带上你选的 L2 码；其余兵种（含 BG/RG）仅 4 角点

### 4. 调整

| 操作 | 效果 |
|---|---|
| 右键关键点 | 切换可见性 0→1→2→0（手动改 vis 编码） |
| `W` / `S` | 上 / 下一个兵种 |
| 右键实例列表项 | 删除该实例 |

### 5. 编辑兵种定义

层级面板 → **编辑层级定义**，可改兵种名、勾选「有 L2 子类」并填 `up,mid,down`。

### 6. 保存

`Ctrl+S` 或切图时自动保存。JSON 里 `shape_type = hierarchy_pose`；BO/RO 含 5 个 keypoints，其余兵种为 4 角点。

---

## 二、导出 YOLO-Pose 数据集

**工具 → 数据集处理工具**，模式选 `TO_YOLO_POSE`：

- 自动识别 `hierarchy_pose` 与 `pose` 两种类型
- 统计最大关键点数 → 写入 `data.yaml` 的 `kpt_shape: [5, 3]`
- 输出 `images/{train,val}` + `labels/{train,val}` + `data.yaml`

每行 label 格式：

```
class_id cx cy w h  kpt0_x kpt0_y kpt0_v  ...  kpt4_x kpt4_y kpt4_v
```

- `class_id`：0-13（0=BG 蓝哨兵，7=RG 红哨兵）
- KP4：仅 BO/RO 有；vis = L2 子类码（0=none 1=up 2=mid 3=down）；其余兵种为 4 角点

---

## 三、训练 DecoupledEdgeYOLO

### 方式 A：GUI

```bash
python tools/hierarchical_train.py
```

填数据集目录、调超参、点 **开始训练**。

### 方式 B：脚本

```bash
python scripts/hierarchical_train.py
```

超参在 `scripts/hierarchical_paths.py` 里改。

### 模型结构

```
yolov10n backbone + neck (无需 .pt 预训练)
        ├─ PnP Pose Head  → bbox(14类) + 4 角点
        └─ L2 SubLevel Head → none/up/mid/down  (输入 detach, stop-gradient)
```

- **bbox 分类头输出 14 类**（颜色 + 兵种都在 class_id 里）
- **L2 头输入 `.detach()`**：L2 损失只更新 L2 头自身，不回传 backbone，避免细粒度分级污染主干
- L2 只在 BO/RO 样本上有非 none 监督，其他兵种 L2 恒 none

权重输出到 `runs/hierarchical_v10n/weights/{best,last}.pt`。

---

## 四、关键文件

| 文件 | 作用 |
|---|---|
| `config/hierarchical_classes.json` | 14 类 + L2 子类定义 |
| `core/hierarchical_config.py` | 配置加载 + KP 索引常量 |
| `ui/hierarchical_panel.py` | 右侧标注面板 |
| `ui/hierarchical_config_dialog.py` | 兵种定义编辑对话框 |
| `core/hierarchical_model.py` | DecoupledEdgeYOLO 模型 |
| `scripts/hierarchical_train.py` | 自定义训练循环 |
| `tools/hierarchical_train.py` | 训练 GUI |
