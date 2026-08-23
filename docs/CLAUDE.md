# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: ArmorLabel

A PySide6 + SAM3 annotation tool for **RoboMaster (RM) armor-plate datasets** with YOLO-pose support. Forked from [luohuabuxiema/LuoHuaLabel](https://github.com/luohuabuxiema/LuoHuaLabel), extended with semi-auto / full-auto armor annotation, video frame extraction, and a dataset-processing toolchain.

Domain language: "buff" / "机关" = sentry/turret targets; "armor" = numbered armor plate with 4 light-bar corners; "kpts" = keypoints (light-bar tips).

## Environment

> **Always activate the conda env first** — PyTorch + CUDA + sam3 are pinned to a specific build.
> ```bash
> conda activate sam3
> ```

- Python 3.11, PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 9.19.0 (see `requirements.txt`).
- SAM3 is installed from GitHub (`git+https://github.com/facebookresearch/sam3.git`); the `sam3/` directory in-repo is the vendored source for offline use.
- Weights live in `models/` (gitignored): `sam3.pt` (~3.4 GB) is required for SAM-assisted annotation; `best.pt` / `best.onnx` is the trained YOLOv12n-Pose model.
- No test framework, linter, or type checker is configured — verify changes by launching the app.

## Common commands

```bash
conda activate sam3
python main.py                          # main annotation tool
python tools/main_dataset_tool.py       # dataset processing (YOLO split/convert, UNet masks)
python tools/video_frame_extractor.py   # standalone video frame extractor
python tools/yolo_auto_annotate.py      # YOLO batch auto-annotate
python tools/yolo_train.py              # YOLO training UI (50+ params)
python tools/yolo_inference.py          # YOLO inference
python tools/mlp_train.py               # MLP classifier training
python tools/mlp_demo.py                # MLP classifier demo
python tools/armor_data_extractor.py    # armor data extraction
# 常用 CLI（详见 scripts/README.md）
python scripts/03_convert_and_split_yolo.py
python scripts/yolo_2_mlp.py
```

## Architecture

**Entry points** are top-level scripts; `main.py` wires `ui/main_window.Ui_MainWindow` to `core/canvas.Canvas` plus the standalone tool windows. The app uses Qt signals extensively — most cross-module communication is `QObject` signals, not direct calls.

### Core layers

- `main.py` — `MainWindow`: event wiring, undo/redo stacks (configurable depth), format I/O dispatch, batch-annotation state machine. Holds global flags (`sam_loading`, `pose_mode_active`, `_batch_mode`, …).
- `core/canvas.py` — `Canvas(QGraphicsScene)`. Mouse events, SAM hover preview, crosshair cursor, OBB handles. Defines the `CanvasMode` enum (RECT/POLY/POINT/RBOX/POSE).
- `core/shapes.py` — `RectShape`, `PolyShape`, `PointShape`, `RotatedRectShape`. OBB uses polar-coord rotation: `angle_deg = math.degrees(math.atan2(dy, dx)); self.setRotation(angle_deg + 90)`. All 4 corners are clamped to image bounds (intentional "wall-sliding").
- `core/pose_shapes.py` — `PoseShape` (bbox + per-keypoint `KeypointItem`). Right-click on a keypoint cycles visibility `0→1→2→0` (unlabeled → occluded → visible).
- `core/sam_client.py` — `SAMClient` orchestrates three `QThread` workers (`ModelLoadWorker`, `ImageLoadWorker`, `SamInferenceWorker`). The `sam_loading` flag blocks image switching during model/image load — touching this without coordination will crash the app.
- `core/rm_armor_auto.py` — RM semi-auto: SAM text prompts `"number"` then `"light"`, filters bars with `cv2.fitEllipse`, validates left/right bar geometry, then shrinks bbox and places 4 corner keypoints.
- `core/rm_buff_merge_auto.py` + `core/buff_merge_geometry.py` — full-auto buff/sentry annotation (merges detected number + adjacent light bars into a single `PoseShape`).
- `core/exporter.py` — JSON (LabelMe-style) / YOLO (det/seg/OBB/pose) / Pascal VOC XML. OBB JSON stores `rect` `[cx,cy,w,h]` + `angle`; pose JSON stores `keypoints` + `kpt_names`.
- `core/yolo_client.py` — YOLO inference wrapper used by the auto-annotate and inference tools.
- `core/mlp_classifier.py` + `tools/mlp_train.py` / `tools/mlp_demo.py` — secondary MLP classifier pipeline (armor number recognition).
- `core/pose_config.py` + `config/pose_classes.json` — pose skeleton definitions. The `BT` category defines 5 keypoints; the JSON is edited via `ui/pose_config_dialog.py`.

### Standalone windows

Each is a separate `QMainWindow` launched from the menu (`工具` / `设置`) and runs long tasks in a `QThread` worker so the UI stays responsive.
All 8 tool entry points live in `tools/` (统一基类 `core/base_tool_window.py` 提供 header / status dot / log console / config 持久化).

- `tools/main_dataset_tool.py` — 6 modes: `YOLO_SPLIT`, `TO_YOLO_DET`, `TO_YOLO_SEG`, `TO_YOLO_OBB`, `TO_YOLO_POSE`, `UNET`. Default split ratios 0.7/0.2/0.1. Always writes a `data.yaml` with train/val/test paths.
- `tools/video_frame_extractor.py` — dual-frame preview + `RangeSlider` timeline, 0.001s interval precision, cancelable background extraction. Output naming: `{prefix}_{frame_index:06d}.{format}`.
- `ui/dataset_fusion_dialog.py` + `core/dataset_fusion.py` — multi-dataset fusion.
- `ui/batch_rename_dialog.py` — batch filename renaming.
- `tools/yolo_train.py` — YOLO training window exposing all 50+ ultralytics params (see `yolo_train_configs/train.yaml` for an example config: pose weights, `pose: 12.0`, `kobj: 1.0`).

### Pose annotation workflow

Press **U** to enter Pose mode. The right panel (`ui/pose_panel.py`) shows the active pose class and instance list. Four sub-modes (all work without pressing Space):

| Sub-mode | SAM | Semi-auto | Flow |
|---|---|---|---|
| Manual | off | off | Drag bbox → click to add keypoints |
| SAM assist | on | off | Hover preview, click to confirm bbox |
| Semi-auto (RM) | off | on | Drag bbox → auto-detect number + light bars → shrink → place kpts |
| SAM + semi-auto | on | on | Click → SAM detects bbox → semi-auto runs → kpts placed |

**Batch full-auto-annotate** (red button in pose panel): three policies — `overwrite` / `skip` / `append`. The state machine in `main.py` drives image switching, waits for SAM feature extraction, runs full-auto, saves, and advances. The pose workflow uses **W/S** to switch pose category.

### SAM3 inference pipeline (`core/sam_client.py`)

1. `ModelLoadWorker` loads `models/sam3.pt` in a background thread; sets `sam_loading = True` while loading.
2. `ImageLoadWorker` calls `Sam3Processor.set_image()` per image switch.
3. `SamInferenceWorker` runs a `queue.Queue(maxsize=1)` — point prompts (mouse hover/click) and text prompts (full-image batch detection) share the same bus.
4. `sam_loading` flag is the only crash guard — never bypass it.

Prompt types: point (hover/click) and text (full-image). Point mode in the canvas automatically disables SAM (see `CanvasMode.POINT`).

## YOLO-Pose model (`models/best.onnx`)

`ONNX_USAGE.md` documents the exported model. Key points:

- Input: `[1, 3, 640, 640]` float32, letterbox to 640 with **top-left alignment** (`cv2.copyMakeBorder` with top=left=0).
- Output: `[1, 21, 8400]` — 8400 anchors from strides `[8, 16, 32]` (`80x80 + 40x40 + 20x20`).
- Channel layout: `0-3` cx,cy,w,h (640-space); `4-5` sigmoid class scores (`RT`, `BT`); `6-20` five `[x, y, vis]` keypoints (decoded, 640-space, vis already sigmoid'd).
- Output is `xywh` — convert to `xyxy` before NMS. Visibility threshold: `>=0.5` → visible (2), `<0.1` → not visible (0), else occluded (1).
- Coordinate scaling: `x_640 / scale` → original-image coords (works only because letterbox is top-left aligned).

## Settings & persistence

Two separate persistence layers — they do not share keys:

1. `QSettings("ArmorLabel", "settings")` → `~/.config/ArmorLabel/settings.ini` — global app params (RM thresholds, SAM device, YOLO device, undo depth, snap threshold). Edited via `ui/settings_dialog.py`. **SAM device changes need a restart; YOLO device is hot-reloadable.**
2. `QSettings("LuoHuaLabel", "settings")` → `~/.config/LuoHuaLabel/settings.ini` — older global params (overlay timeout, etc.). `main.py` reads from this namespace.
3. Standalone tool windows auto-save last-used settings to `~/.config/LuoHuaLabel/tool_configs.yaml` on close and restore on open (`core/tool_config.py`).

### Key tunable parameters

| Param | Default | Effect |
|---|---|---|
| `RM_VERTICAL_ANGLE_THRESHOLD` | 50.0° | Max deviation from vertical for a valid light bar |
| `RM_PARALLEL_THRESHOLD` | 45.0° | Max angle difference between left/right bars |
| `RM_BAR_LENGTH_MIN_RATIO` | 0.5 | Min bar length = `max(num_w,num_h) * ratio` |
| `RM_SEARCH_RADIUS_RATIO` | 1.5 | Bar search radius around number bbox |
| `RM_ROI_PADDING` | 10 px | Padding around armor ROI bbox |
| `RM_OVERLAY_TIMEOUT` | 1000 ms | RM full-auto overlay display duration |
| `MAX_HISTORY_STEPS` | 20 | Undo/redo stack depth |
| `SNAP_THRESHOLD` | 8 px | Polygon vertex edge-snap distance |
| `YOLO_CONF` / `YOLO_IOU` / `YOLO_IMGSZ` | 0.25 / 0.7 / 640 | YOLO inference thresholds |

## Conventions & gotchas

- **Conda env is mandatory** — sam3 won't import otherwise.
- Empty annotations produce **empty files on save** (intentional, to mark "reviewed, nothing here").
- OBB corners are **clamped to image bounds** during drag — resist the urge to "fix" the wall-sliding; it's collision handling.
- Point mode in canvas **disables SAM** (point prompts are mode-disabled, not just hidden).
- Image switching is **blocked while SAM is loading** (`sam_loading` guard in `main.py:55`).
- 4K+ images with hundreds of polygon masks will see pan/zoom FPS degrade — this is the cost of `QGraphicsScene` rendering, not a bug.
- `sam3/` is a vendored copy of the upstream repo; treat as read-only source — changes there should go upstream.
- `ultralytics/` is also vendored; `ultralytics/utils/metrics.py` is modified in-tree (see git status) — review before bumping.
- `new_buff/`, `runs/`, `stored_weights/`, `datasets/`, `origin_data/`, and `*.jpg/*.png/*.mp4/*.avi` are all gitignored data/output directories — never commit artifacts.

## Where to make changes

- New canvas mode → add to `CanvasMode` in `core/canvas.py`, wire in `main.py:_set_mode`, add export branch in `core/exporter.py`.
- New shape type → subclass in `core/shapes.py`, register a `ShapeSpec` in `core/shape_registry.py:ensure_default_specs()` — `main.py` and `core/exporter.py` 不再需要改动. (架构基建任务 #26 之后)
- New armor heuristic → `core/rm_armor_auto.py` (semi-auto) or `core/rm_buff_merge_auto.py` (full-auto); mirror params in `ui/settings_dialog.py` and `config/pose_classes.json` if the skeleton changes.
- New dataset-processing mode → add branch in `tools/main_dataset_tool.py` and the worker enum.
- New standalone tool → 继承 `core/base_tool_window.BaseToolWindow`, 只写 `build_ui()`, 启动方式 `python tools/<name>.py`. (架构基建任务 #25 之后)
- 公共基础: 6 个基建模块 `core/constants.py` / `core/event_bus.py` / `core/settings.py` / `core/worker.py` / `core/base_tool_window.py` / `core/shape_registry.py`
