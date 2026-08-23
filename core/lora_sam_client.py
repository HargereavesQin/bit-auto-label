# -*- coding: utf-8 -*-
"""
独立 LoRA-SAM 客户端 — 与 core/sam_client.py 完全解耦。

仅支持文本提示 grounding（与训练时 enable_inst_interactivity=False 一致）。
加载流程: sam3.pt → PEFT adapter → merge_and_unload → Sam3Processor。
"""
import os as _os
import queue

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from PySide6.QtCore import QObject, QSettings, QThread, Signal

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False

_settings = QSettings("ArmorLabel", "settings")
_settings.beginGroup("Parameters")
DEVICE = str(_settings.value("SAM_DEVICE", "cuda"))
_settings.endGroup()

if DEVICE not in ("cpu",) and not DEVICE.startswith("cuda"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _bpe_path():
    return _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "../sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    )


def _collect_lora_target_modules(model):
    """与 lora_train.inject_lora 保持一致的 target 扫描。"""
    target_modules = []
    for name, module in model.named_modules():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj"]):
            if isinstance(module, nn.Linear):
                target_modules.append(name)
    return list(set(target_modules))


class LoraImageLoadWorker(QThread):
    finished = Signal(bool, str)

    def __init__(self, image_path, processor):
        super().__init__()
        self.image_path = image_path
        self.processor = processor
        self._result_state = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                self.finished.emit(False, "已取消")
                return
            pil_img = Image.open(self.image_path).convert("RGB")
            dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
            with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=dtype):
                self._result_state = self.processor.set_image(pil_img)
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))

    def get_result_state(self):
        return self._result_state


class LoraModelLoadWorker(QThread):
    loaded = Signal(object, object, bool, str)

    def __init__(self, checkpoint_path, lora_adapter_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.lora_adapter_path = lora_adapter_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                self.loaded.emit(None, None, False, "已取消")
                return
            if not HAS_SAM3:
                self.loaded.emit(None, None, False, "SAM3 模块未安装，请先安装 sam3 包")
                return
            if not self.lora_adapter_path or not _os.path.isdir(self.lora_adapter_path):
                self.loaded.emit(None, None, False, f"LoRA adapter 路径无效: {self.lora_adapter_path}")
                return
            adapter_file = _os.path.join(self.lora_adapter_path, "adapter_model.safetensors")
            if not _os.path.isfile(adapter_file):
                self.loaded.emit(None, None, False, f"未找到 adapter_model.safetensors: {adapter_file}")
                return

            model = build_sam3_image_model(
                bpe_path=_bpe_path(),
                checkpoint_path=self.checkpoint_path,
                enable_inst_interactivity=False,
            )
            model.to(DEVICE)

            # 优先用 adapter_config.json 还原 LoRA；失败则按训练脚本规则手动注入
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.lora_adapter_path)
            except Exception as peft_err:
                print(f"[LoraSam] PeftModel.from_pretrained 失败，回退手动注入: {peft_err}")
                from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
                from safetensors.torch import load_file

                target_modules = _collect_lora_target_modules(model)
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules=target_modules,
                    modules_to_save=["mask_decoder"],
                    bias="none",
                )
                model = get_peft_model(model, lora_config)
                state_dict = load_file(adapter_file)
                set_peft_model_state_dict(model, state_dict)

            model = model.merge_and_unload()
            model.eval()

            processor = Sam3Processor(model, device=DEVICE)
            self.loaded.emit(model, processor, True, "LoRA-SAM 模型加载成功")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.loaded.emit(None, None, False, f"LoRA-SAM 加载失败: {e}")


class LoraSamInferenceWorker(QThread):
    """仅文本提示推理（与训练路径一致）。"""
    text_result_ready = Signal(list, str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.inference_state = None
        self.task_queue = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            try:
                prompt_text = self.task_queue.get(timeout=0.05)
                if not self.model or not self.processor or not self.inference_state:
                    self.text_result_ready.emit([], prompt_text)
                    continue

                dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
                with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=dtype):
                    out_state = self.processor.set_text_prompt(
                        prompt=prompt_text, state=self.inference_state
                    )
                    masks = out_state.get("masks", [])
                    scores = out_state.get("scores", [])
                    boxes = out_state.get("boxes", [])

                    results = []
                    if len(masks) > 0:
                        for i in range(len(masks)):
                            mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                            mask_np = np.squeeze(mask_np)
                            score_val = float(
                                scores[i].cpu() if torch.is_tensor(scores[i]) else scores[i]
                            )
                            box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]
                            if box.ndim > 1:
                                box = box.squeeze()
                            x1, y1, x2, y2 = box
                            rect_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                            mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(
                                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            poly_pts = []
                            rect_obb = []
                            if contours:
                                largest = max(contours, key=cv2.contourArea)
                                epsilon = 0.002 * cv2.arcLength(largest, True)
                                approx = cv2.approxPolyDP(largest, epsilon, True)
                                poly_pts = approx.reshape(-1, 2).tolist()
                                obb = cv2.minAreaRect(largest)
                                rect_obb = [
                                    float(obb[0][0]), float(obb[0][1]),
                                    float(obb[1][0]), float(obb[1][1]),
                                    float(obb[2]),
                                ]
                            if poly_pts:
                                results.append({
                                    "poly_pts": poly_pts,
                                    "rect": rect_xywh,
                                    "obb": rect_obb,
                                    "score": score_val,
                                })
                    self.text_result_ready.emit(results, prompt_text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LoraSam] 推理错误: {e}")

    def request_text_inference(self, prompt_text):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                pass
        self.task_queue.put(prompt_text)

    def stop(self):
        self.running = False
        if not self.wait(10000):
            print("LoraSamInferenceWorker: 推理线程 10s 内未退出（长任务进行中），继续退出流程")


class LoraSamClient(QObject):
    model_status_changed = Signal(bool, str)
    text_result_ready = Signal(list, str)
    image_load_finished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.processor = None
        self.inference_worker = LoraSamInferenceWorker()
        self.inference_worker.text_result_ready.connect(self.text_result_ready)
        self.inference_worker.start()
        self.load_worker = None
        self._image_load_worker = None

    def load_model_async(self, checkpoint_path, lora_adapter_path):
        if self.load_worker is not None and self.load_worker.isRunning():
            self.model_status_changed.emit(False, "LoRA 模型正在加载中，请勿重复触发...")
            return
        self.model_status_changed.emit(False, "正在后台加载 LoRA-SAM 模型…")
        self.load_worker = LoraModelLoadWorker(checkpoint_path, lora_adapter_path)
        self.load_worker.loaded.connect(self._on_model_loaded)
        self.load_worker.start()

    def _on_model_loaded(self, model, processor, success, msg):
        if success:
            self.model = model
            self.processor = processor
            self.inference_worker.model = model
            self.inference_worker.processor = processor
        self.model_status_changed.emit(success, msg)

    def set_image_async(self, image_path, token=0):
        """异步提取图片特征。快速切图时不 terminate，排队最新路径。"""
        if not self.processor:
            self.image_load_finished.emit(False, "LoRA-SAM 处理器未就绪")
            return
        self._want_path = image_path
        self._want_token = token
        if self._image_load_worker is not None and self._image_load_worker.isRunning():
            return
        self._spawn_image_worker(image_path, token)

    def _spawn_image_worker(self, image_path, token):
        self._active_token = token
        self._image_load_worker = LoraImageLoadWorker(image_path, self.processor)
        self._image_load_worker.finished.connect(self._on_image_load_finished)
        self._image_load_worker.start()

    def _on_image_load_finished(self, success, msg):
        finished = self.sender()
        if finished is not None and finished is not self._image_load_worker:
            return
        # 加载期间又请求了更新的路径 → 继续加载最新，不向 UI 发中间结果
        want_path = getattr(self, "_want_path", None)
        want_token = getattr(self, "_want_token", 0)
        active_token = getattr(self, "_active_token", 0)
        if want_path and (
            want_token != active_token
            or (finished and getattr(finished, "image_path", None) != want_path)
        ):
            self._spawn_image_worker(want_path, want_token)
            return
        if success and finished is not None:
            old_state = self.inference_worker.inference_state
            self.inference_worker.inference_state = finished.get_result_state()
            if old_state is not None:
                del old_state
            self._image_load_count = getattr(self, "_image_load_count", 0) + 1
            if self._image_load_count % 8 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.image_load_finished.emit(success, msg)

    def request_text_inference(self, prompt_text):
        if self.model:
            self.inference_worker.request_text_inference(prompt_text)

    def unload_model(self):
        self.inference_worker.inference_state = None
        self.inference_worker.model = None
        self.inference_worker.processor = None
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        self.inference_worker.stop()
        for worker in (self.load_worker, self._image_load_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                if not worker.wait(30000):
                    print(f"{type(worker).__name__}: 等待 30s 未结束，放弃等待（可能造成退出告警）")
