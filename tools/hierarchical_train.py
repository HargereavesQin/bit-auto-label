# -*- coding: utf-8 -*-
"""层级标注训练 GUI (DecoupledEdgeYOLO + yolov10n).

继承 BaseToolWindow, 在后台 QThread 跑 scripts.hierarchical_train.main().
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal

from core.base_tool_window import BaseToolWindow
from ui.icons import Icon, Icons, C


class HierTrainWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        self._stop = False
        # 把参数写回 hierarchical_paths 模块 (运行时覆盖)
        try:
            import importlib
            import scripts.hierarchical_paths as P
            for k, v in self.params.items():
                if hasattr(P, k):
                    setattr(P, k, v)
            importlib.reload(P)
        except Exception as e:
            self.finished_signal.emit(False, f"参数加载失败: {e}")
            return

        # 重定向 stdout 到 log signal
        old_stdout = sys.stdout
        class _StdCapture:
            def write(self_, text):
                if text and text.strip():
                    self_.log_signal.emit(text.rstrip())
                if not self._stop:
                    old_stdout.write(text)
            def flush(self_):
                old_stdout.flush()
        sys.stdout = _StdCapture()
        try:
            from scripts.hierarchical_train import main as train_main
            train_main()
            self.finished_signal.emit(True, "训练完成")
        except KeyboardInterrupt:
            self.finished_signal.emit(False, "用户停止训练")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"训练失败: {e}")
        finally:
            sys.stdout = old_stdout


class HierarchicalTrainWindow(BaseToolWindow):
    TOOL_NAME = "hierarchical_train"
    WINDOW_TITLE = "层级标注训练  /  HIERARCHICAL  TRAIN"

    def build_ui(self):
        self.setWindowIcon(Icon.icon(Icons.POSE, size=24, color=C.AMBER))
        self.setMinimumSize(620, 600)

        self.add_section("数据集  /  DATASET", self._build_data_form())
        self.add_section("模型  /  MODEL", self._build_model_form())
        self.add_section("训练超参  /  HYPERPARAMS", self._build_hyper_form())
        self._build_action_row()
        self.add_log_console(min_height=180)

        self._worker: HierTrainWorker | None = None

    # ---------------------------------------------------------
    def _build_data_form(self) -> QWidget:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("数据集目录:"))
        self.data_dir_edit = QLineEdit(str(self._default_data_dir()))
        btn = QPushButton("浏览")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self._pick_data_dir)
        row.addWidget(self.data_dir_edit, 1)
        row.addWidget(btn)
        layout.addLayout(row)

        hint = QLabel("目录结构需含 images/train, images/val, labels/train, labels/val + data.yaml")
        hint.setProperty("role", "caption")
        layout.addWidget(hint)
        return box

    def _build_model_form(self) -> QWidget:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setSpacing(6)
        layout.addWidget(QLabel("骨干: yolov10n.yaml (vendored, 无需 .pt 预训练)"))
        layout.addWidget(QLabel("结构: DecoupledEdgeYOLO (backbone+neck + PnP Pose + L2 SubLevel)"))
        layout.addWidget(QLabel("14 类: 0:BG 1-6:蓝方 7-13:红方 (颜色并入 class_id)"))
        layout.addWidget(QLabel("Stop-Gradient: L2 头输入 .detach(), 梯度不回传 backbone"))
        layout.addWidget(QLabel("L2 只对 BO(5)/RO(12) 生效: up/mid/down"))
        return box

    def _build_hyper_form(self) -> QWidget:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        grid = []
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(150)
        grid.append(("Epochs", self.epochs_spin))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        grid.append(("Batch", self.batch_spin))

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        grid.append(("Imgsz", self.imgsz_spin))

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(1e-6, 1e-1)
        self.lr_spin.setValue(1e-3)
        grid.append(("LR", self.lr_spin))

        self.device_edit = QLineEdit("0")
        grid.append(("Device", self.device_edit))

        self.amp_chk = QCheckBox("AMP (混合精度)")
        self.amp_chk.setChecked(True)
        layout.addWidget(self.amp_chk)

        for label_text, w in grid:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text + ":"))
            row.addWidget(w, 1)
            layout.addLayout(row)
        return box

    def _build_action_row(self):
        row = QHBoxLayout()
        row.setSpacing(10)
        self.start_btn = QPushButton("  开始训练")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setProperty("role", "primary")
        self.start_btn.clicked.connect(self._on_start)
        row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("  停止")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setProperty("role", "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self.stop_btn)
        row.addStretch()
        self.add_layout(row)

    # ---------------------------------------------------------
    def _default_data_dir(self) -> Path:
        from scripts.hierarchical_paths import DATASET_ROOT
        return DATASET_ROOT

    def _pick_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择层级数据集目录", self.data_dir_edit.text())
        if path:
            self.data_dir_edit.setText(path)

    def _on_start(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "训练正在进行中")
            return
        params = {
            "EPOCHS": self.epochs_spin.value(),
            "BATCH_SIZE": self.batch_spin.value(),
            "TRAIN_IMGSZ": self.imgsz_spin.value(),
            "LR": self.lr_spin.value(),
            "DEVICE": self.device_edit.text().strip(),
            "AMP": self.amp_chk.isChecked(),
            "DATASET_ROOT": Path(self.data_dir_edit.text()),
        }
        # 派生路径
        root = params["DATASET_ROOT"]
        params["TRAIN_IMAGES"] = root / "images/train"
        params["VAL_IMAGES"] = root / "images/val"
        params["TRAIN_LABELS"] = root / "labels/train"
        params["VAL_LABELS"] = root / "labels/val"
        params["DATA_YAML"] = root / "data.yaml"

        self._worker = HierTrainWorker(params)
        self._worker.log_signal.connect(self.log)
        self._worker.finished_signal.connect(self._on_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.set_status("训练中", "working")
        self.log(f"启动训练: {params}")
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
            self.log("已发送停止信号…")

    def _on_finished(self, ok: bool, msg: str):
        self.log(msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.set_status("完成" if ok else "错误", "done" if ok else "error")
        if not ok:
            QMessageBox.critical(self, "训练失败", msg)

    # ---------------------------------------------------------
    def load_tool_config(self, cfg: dict):
        if not cfg:
            return
        if cfg.get("data_dir"):
            self.data_dir_edit.setText(cfg["data_dir"])
        if cfg.get("epochs"):
            self.epochs_spin.setValue(int(cfg["epochs"]))
        if cfg.get("batch"):
            self.batch_spin.setValue(int(cfg["batch"]))
        if cfg.get("imgsz"):
            self.imgsz_spin.setValue(int(cfg["imgsz"]))
        if cfg.get("lr"):
            self.lr_spin.setValue(float(cfg["lr"]))
        if cfg.get("device"):
            self.device_edit.setText(cfg["device"])
        if cfg.get("amp") is not None:
            self.amp_chk.setChecked(bool(cfg["amp"]))

    def dump_tool_config(self) -> dict:
        return {
            "data_dir": self.data_dir_edit.text(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "lr": self.lr_spin.value(),
            "device": self.device_edit.text(),
            "amp": self.amp_chk.isChecked(),
        }


def main():
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = HierarchicalTrainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
