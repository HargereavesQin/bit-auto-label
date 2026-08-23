# 修复任务清单

- [x] 问题1: 位姿模式下强制关闭 SAM，跳过图片推理（`_set_mode` + `on_file_selected` + `update_model_status`）
- [x] 问题2: 右侧位姿类别列表不显示（`_enter_pose_mode` 中确保面板创建和刷新）
- [x] 问题3: 位姿标注改为「空格新建实例 → 拖拽bbox → 点击添加关键点」（`canvas.py` + `keyPressEvent`）
- [x] 问题4: 支持多个位姿实例（多类别/单类别多实例）（`canvas.py` 中 `pose_active` 状态控制）
- [x] 问题5: 快速浏览图片时 SAM 导致卡崩溃 - 禁止 SAM 加载中切换图片（`sam_loading` 标志）
- [x] 问题6: 统计功能区分 JSON/YOLO/XML 标注格式（`statistics.py` + `statistics_dialog.py`）
- [x] 测试验证功能（语法检查通过）

## 历史修复记录

### 问题7: RM Armor Auto 全自动模式下多装甲板标注失败

**现象**: 第一张单装甲板图片全自动标注成功，第二张双装甲板图片只有第一个装甲板被标注，第二个完全跳过。

**根因**: `_handle_full_auto_light_bar_results` 中，所有灯条来自全图检测结果（14个），而非按数字中心就近匹配。当两个装甲板靠近时，装甲板A的数字可能被装甲板B的灯条错误匹配，导致候选列表为空或质量差，在垂直度/平行度阈值检查处被静默 `continue` 跳过。

**修复** (`core/rm_armor_auto.py`):
1. `_handle_full_auto_number_results`: 将 `_full_auto_numbers` 从 `[(cx, cy)]` 扩展为 `[(cx, cy, bw, bh)]`，存储数字边界框尺寸
2. `_handle_full_auto_light_bar_results`: 新增 `search_radius = max(num_w, num_h) * 5` 邻近过滤器，只保留距数字中心 `5倍bbox尺寸` 半径内的灯条；添加调试打印说明每个装甲板被跳过的原因（半径内灯条不足/无左右候选/椭圆拟合失败/垂直度偏差/平行度不足）
3. 以上改动不影响单装甲板场景（搜索半径仍然有效）

**调试方法**: 运行后查看控制台输出，格式为 `[rm_auto] number at (x,y): reason, skip`

### 功能8: RM Armor Auto 全自动可视化覆盖层

**需求**: 点击全自动标注后，以虚线形式可视化显示：搜索半径范围、识别到的数字位置，5秒后自动消失或切换图片时消失。

**修改文件**:
- `core/rm_armor_auto.py`: 新增 `full_auto_regions_ready = Signal(list)` 信号，在数字检测完成后、请求灯条推理前发出
- `core/canvas.py`: 新增 `rm_overlay_items` 列表和 `clear_rm_overlays()` 方法，在 `load_image` 中调用清除
- `main.py`: 新增 `rm_overlay_timer` (QTimer，单次5秒)，连接 `full_auto_regions_ready` 到 `_on_full_auto_regions_ready` 处理器，在其中绘制橙色虚线圆圈（搜索半径）、绿色虚线矩形（数字框）、"数字"文本标签；在 `_on_full_auto_requested` 和 `on_file_selected` 中清除旧覆盖层并停止计时器

### 功能9: RM Armor Auto 全自动选灯条策略优化

**问题**: 选灯条策略为"离数字最近的左右各一个"（x坐标最大/最小），导致误判的水平特征被优先选中，真竖直灯条反而被跳过。日志显示偏差达79-88°（近似水平），远超过50°阈值。

**修复** (`core/rm_armor_auto.py`):
1. 搜索半径：5→3（减少误判干扰）
2. 新增 `_pick_best_bar(candidates)` 方法：对每个候选拟合椭圆、计算偏差，选偏差最小的（最竖直的）作为最佳灯条
3. `_handle_full_auto_light_bar_results` 中：替换 `max(left_candidates)` / `min(right_candidates)`（按x坐标最近）为 `_pick_best_bar(left_candidates)` / `_pick_best_bar(right_candidates)`（按偏差最小）

### 功能10: RM Armor Auto 全自动进一步优化

**问题1**: 搜索半径3倍仍然偏大，部分场景引入过多误判灯条；ROI仅包含灯条端点未包含数字框，导致bbox不够贴合。

**问题2**: 关键点数量问题需调试确认。

**修复**:
1. `core/rm_armor_auto.py`: 搜索半径 3→2（line 336, 346）；ROI计算新增数字bbox覆盖 `all_x.extend([num_cx - num_w/2, num_cx + num_w/2])` 和 `all_y.extend([num_cy - num_h/2, num_cy + num_h/2])`
2. `main.py`: `_on_full_auto_step_placed` 中添加关键点调试打印 `[full_auto] placing N keypoints` 和每个关键点的坐标

### 功能11: RM Armor Auto 全自动选灯条策略改为距离优先

**问题1**: kpt[2]和kpt[3]坐标完全相同——右侧灯条椭圆拟合端点退化（两点距离<5像素），`_fit_ellipse_endpoints`返回重合端点，未做检查。

**问题2**: `_pick_best_bar`选"最竖直的"而非"最近的"，导致更远的真灯条被选中，更近的假灯条反而被跳过。

**修复** (`core/rm_armor_auto.py`):
1. `_pick_best_bar` → `_pick_closest_valid_bar(candidates, ref_x, ref_y)`：按距数字中心距离排序（近→远），对每个候选检查椭圆拟合有效性、端点距离>5像素、偏差<阈值，返回第一个合格者；参照半自动模式"最近优先"策略
2. `_handle_full_auto_light_bar_results`: 调用改为 `_pick_closest_valid_bar(left_candidates, num_cx, num_cy)`，移除单独的偏差检查（已内嵌），保留平行度检查，新增左右bar各自的失败原因打印

### 功能12: `_fit_ellipse_endpoints` 端点计算改为数学精确值

**问题**: 原实现用 `cv2.ellipse2Poly` 生成72点多边形 + 遍历找最远对 + `int()` 截断角度/坐标，导致三重精度损失。小灯条（细长条）的椭圆拟合后，多边形采样的"最远对"可能落在同一像素上，产生重合端点。

**修复** (`core/rm_armor_auto.py` `_fit_ellipse_endpoints`):
- 移除 `cv2.ellipse2Poly` 和最远对搜索
- 改为直接从椭圆参数数学计算长轴端点：`pt1 = (cx + dx, cy + dy)`，`pt2 = (cx - dx, cy - dy)`
- `semi_a = max(size_w, size_h) / 2`，根据哪个轴更大确定角度偏移
- 半自动模式也调用同一函数，一并受益

### 功能13: RM Armor Auto 调试打印增强 + SAM 提示词修正

**问题**: kpt[2]和kpt[3]仍然重合，需要更详细的运行时诊断信息；SAM文字提示"light bar"可能不如"light"有效。

**修改** (`core/rm_armor_auto.py`):
1. `_fit_ellipse_endpoints` 新增调试打印：椭圆参数(center, size, angle)和计算出的端点坐标及距离
2. `_pick_closest_valid_bar` 新增调试打印：每个候选被跳过的主要原因(fitEllipse失败/端点距离过小/偏差超阈值)
3. `_handle_full_auto_light_bar_results` 新增调试打印：成功选中左右灯条后的端点坐标和偏差值；**新增 emit 前的原始 kpts 数据打印** (`l_top/l_bot/r_bot/r_top` 坐标 + `kpts` 列表完整内容)
4. SAM提示词 `"light bar"` → `"light"`：6处（`_on_text_result` 调度条件 x2，`_handle_number_first_results` 请求 x1，`_handle_full_auto_number_results` 请求 x1，半自动 `_handle_light_bar_results`不受影响仍用"light bar"作为相位名）
5. `main.py` `_on_full_auto_step_placed`: 新增 `kpt_count`/`kpt_names` 调试打印 + 每次 `add_keypoint` 的返回值(`ok=True/False`)和当前 keypoint 总数打印

**调试方法**: 运行后查看控制台输出，应能看到：
- `[rm_auto] EMIT kpts list: [[...],[...],[...],[...]]`
- `[full_auto] kpt_count=N kpt_names=[...]`
- `add_keypoint(x,y) v=N: ok=True/False total=N` (每关键点一条)
- 重点关注 kpt_count 是否为 0（导致全部 ok=False）

### 功能14: 修复 kpt[2]==kpt[3] bug + 灯条长度过滤

**Bug 根因**: `_handle_full_auto_light_bar_results` 第389行 `r_bot` 赋值存在笔误：
```python
# 错误代码
r_bot = right_endpoints[1] if right_endpoints[0][1] < right_endpoints[1][1] else right_endpoints[1]
# 两个分支都用了 right_endpoints[1]！
```
导致当条件为 False 时，`r_bot` 被错误赋值为 `right_endpoints[1]`（应该赋值为 `right_endpoints[0]`），与 `r_top` 相同。

**修复** (`core/rm_armor_auto.py`):
1. 第386-392行：替换三元表达式为 `sorted()` 写法，消除潜在逻辑歧义：
   ```python
   left_sorted = sorted(left_endpoints, key=lambda p: p[1])
   right_sorted = sorted(right_endpoints, key=lambda p: p[1])
   l_top, l_bot = left_sorted[0], left_sorted[1]
   r_top, r_bot = right_sorted[0], right_sorted[1]
   ```
2. `_pick_closest_valid_bar` 新增 `min_length` 参数（默认0表示不限制）：
   ```python
   def _pick_closest_valid_bar(self, candidates, ref_x, ref_y, min_length=0):
       ...
       if min_length > 0 and dist < min_length:
           print(f"  [rm_auto] bar ({bcx:.1f},{bcy:.1f}): endpoint dist {dist:.2f}px < min_length {min_length:.2f}px, skip")
           continue
   ```
3. `_handle_full_auto_light_bar_results` 调用处：计算 `number_ref_dist = max(num_w, num_h) * BAR_LENGTH_MIN_RATIO`，传递给左右灯条筛选：
   ```python
   number_ref_dist = max(num_w, num_h) * self.BAR_LENGTH_MIN_RATIO
   left_result = self._pick_closest_valid_bar(left_candidates, num_cx, num_cy, min_length=number_ref_dist)
   right_result = self._pick_closest_valid_bar(right_candidates, num_cx, num_cy, min_length=number_ref_dist)
   ```
   即：灯条端点距离 < 数字bbox长边×BAR_LENGTH_MIN_RATIO → 丢弃该灯条

**可调参数**: `core/rm_armor_auto.py` 第30行 `BAR_LENGTH_MIN_RATIO`，默认 `0.5`（50%）。调大则更严格（要求灯条更长才保留），调小则更宽松。

**调试方法**: 运行后查看控制台，新过滤会打印：
- `[rm_auto] bar (x,y): endpoint dist Npx < min_length Mpx, skip`

### Bug修复: SEARCH_RADIUS_RATIO 硬编码 * 2

**问题**: `core/rm_armor_auto.py:338` 区域可视化叠加层的搜索半径硬编码为 `* 2`，即使设置了 `SEARCH_RADIUS_RATIO=1.1` 也不生效。

**修复**: 将 `max(bw, bh) * 2` 改为 `max(bw, bh) * self.SEARCH_RADIUS_RATIO`

### 功能15: 批量全自动标注所有图片

**需求**: 一键对目录中所有图片进行全自动装甲板标注，提供三种模式选择。

**新增文件/修改**:
- `core/rm_armor_auto.py`: 新增信号 `full_auto_finished = Signal()`，在 `_reset_full_auto()` 中发射
- `ui/pose_panel.py`: 新增信号 `batch_auto_requested`，新增按钮"批量全标注"（红色）
- `main.py`: 批量状态机实现
  - 新增状态变量：`_batch_mode`, `_batch_images`, `_batch_index`, `_batch_total`, `_batch_policy`
  - `_on_batch_auto_requested()`: 弹出选项对话框（覆盖全部/跳过已标注/补全添加）
  - `_batch_process_next()`: 切换到下一张图片，删除已有json（覆盖模式）
  - `_on_batch_auto_finished()`: 保存当前图片，移动到下一张
  - 修改 `_on_sam_image_loaded()`: 批量模式下 SAM 图片加载完成后自动触发 `request_full_auto`

**三种模式**:
- 覆盖全部: 删除已有 .json，每张图片重新标注
- 跳过已标注: 跳过已有 .json 的图片
- 补全添加: 保留已有标注，在其上追加新的标注

**调试方法**: 点击"批量全标注"按钮，选择模式，等待批量处理完成

### 功能16: 视频抽帧工具

**新增文件**: `video_frame_extractor.py` - 独立窗口工具

**功能**:
- 双帧预览：左侧预览起始帧，右侧预览结束帧，拖动滑块实时更新
- 时间轴双滑块：自定义 RangeSlider 组件，同时设置起始/结束范围
- 时间范围选择：可通过数字输入或直接拖动滑块设定
- 抽帧间隔：支持小数秒（如 0.5 秒）
- 预计输出：实时显示预计抽帧数量
- 进度条：显示抽帧进度

**集成**:
- `ui/main_window.py`: 工具栏新增"视频抽帧"按钮
- `main.py`: 导入 `VideoFrameExtractorWindow`，新增 `open_video_extractor()` 方法

**使用方法**: 工具栏点击"视频抽帧" → 选择视频 → 拖动双滑块选择范围 → 设置间隔 → 开始抽帧

**命名格式**: `{prefix}_{frame_index:06d}.{format}`（如 `video_000123.jpg`）
