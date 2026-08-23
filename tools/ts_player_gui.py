#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS 视频播放器 GUI (PySide6).

RGB 增益滑块 + YOLO 检测框 (pt/onnx/engine) + 可拖动进度条 + 按钮操作.

用法:
    python tools/ts_player_gui.py
    python tools/ts_player_gui.py --model /path/best.pt --conf 0.35
"""
import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSlider,
    QVBoxLayout, QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ts_player_rgb import build_detector  # noqa: E402

DEFAULT_VIDEO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "camera_20260805_105958_part001.ts",
)


class PlayerWindow(QMainWindow):
    def __init__(self, args) -> None:
        super().__init__()
        self.setWindowTitle("TS Player GUI")
        self.resize(1280, 900)

        self.save_path = args.save
        self.conf = args.conf
        self.iou = args.iou
        self.det = None

        self.cap = None
        self.fps = 25.0
        self.total = 0
        self.idx = 0
        self.frame = None          # 原始 BGR
        self.vis_rgb = None        # 渲染用 RGB
        self.playing = False
        self._seek_dragging = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self._build_ui(args)

        if args.video and os.path.isfile(args.video):
            self._open_video(args.video)
        if args.model:
            self._load_model(args.model, args.device, args.imgsz)

    # ---------------- UI ----------------

    def _build_ui(self, args) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.canvas = QLabel("打开视频或启动播放…", self)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background:#111; color:#888;")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(420)
        root.addWidget(self.canvas, 1)

        # 进度条
        seek_row = QHBoxLayout()
        self.seek = QSlider(Qt.Horizontal, self)
        self.seek.setEnabled(False)
        self.seek.sliderPressed.connect(self._on_seek_pressed)
        self.seek.sliderReleased.connect(self._on_seek_released)
        self.seek.sliderMoved.connect(self._on_seek_moved)
        seek_row.addWidget(self.seek, 1)
        self.frame_lbl = QLabel("0/0", self)
        self.frame_lbl.setMinimumWidth(90)
        self.frame_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seek_row.addWidget(self.frame_lbl)
        root.addLayout(seek_row)

        # RGB 滑块
        gain_row = QHBoxLayout()
        self.gain_sliders = {}
        for name, init in (("R", args.init_r), ("G", args.init_g), ("B", args.init_b)):
            gain_row.addWidget(QLabel(f"<b>{name}</b>", self))
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(0, 200)
            slider.setValue(int(init * 100))
            slider.valueChanged.connect(lambda _v, n=name: self._on_gain_changed(n))
            self.gain_sliders[name] = slider
            gain_row.addWidget(slider, 1)
            val = QLabel(f"{init:.2f}", self)
            val.setMinimumWidth(42)
            gain_row.addWidget(val)
            self.__dict__[f"val_{name}"] = val
        root.addLayout(gain_row)

        # 按钮行
        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("打开视频", self)
        self.btn_open.clicked.connect(self._choose_video)
        btn_row.addWidget(self.btn_open)

        self.btn_model = QPushButton("加载模型", self)
        self.btn_model.clicked.connect(self._choose_model)
        btn_row.addWidget(self.btn_model)

        self.btn_play = QPushButton("▶ 播放", self)
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_play)
        btn_row.addWidget(self.btn_play)

        self.btn_back = QPushButton("⏪ -1s", self)
        self.btn_back.clicked.connect(lambda: self._step_sec(-1))
        btn_row.addWidget(self.btn_back)
        self.btn_fwd = QPushButton("+1s ⏩", self)
        self.btn_fwd.clicked.connect(lambda: self._step_sec(+1))
        btn_row.addWidget(self.btn_fwd)

        self.chk_detect = QCheckBox("检测框", self)
        self.chk_detect.setChecked(True)
        self.chk_detect.setEnabled(False)
        self.chk_detect.stateChanged.connect(lambda _s: self._render())
        btn_row.addWidget(self.chk_detect)

        btn_row.addWidget(QLabel("conf", self))
        self.spin_conf = QDoubleSpinBox(self)
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(self.conf)
        self.spin_conf.valueChanged.connect(lambda v: setattr(self, "conf", float(v)))
        btn_row.addWidget(self.spin_conf)

        btn_row.addWidget(QLabel("iou", self))
        self.spin_iou = QDoubleSpinBox(self)
        self.spin_iou.setRange(0.05, 0.99)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(self.iou)
        self.spin_iou.valueChanged.connect(lambda v: setattr(self, "iou", float(v)))
        btn_row.addWidget(self.spin_iou)

        self.btn_save = QPushButton("保存 RGB", self)
        self.btn_save.clicked.connect(self._save_gains)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.status = QLabel("就绪", self)
        self.status.setStyleSheet("color:#555;")
        root.addWidget(self.status)

        self.setCentralWidget(central)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_S), self, activated=self._save_gains)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._step_sec(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._step_sec(+1))

    # ---------------- 视频 ----------------

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", os.path.dirname(DEFAULT_VIDEO),
            "Video (*.ts *.mp4 *.avi *.mkv);;All (*)")
        if path:
            self._open_video(path)

    def _open_video(self, path: str) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频: {path}")
            self.cap = None
            return
        self.video_path = path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.idx = 0
        self.seek.setRange(0, max(0, self.total - 1))
        self.seek.setEnabled(True)
        self.btn_play.setEnabled(True)
        if not self.save_path:
            stem = os.path.splitext(os.path.basename(path))[0]
            self.save_path = f"rgb_gain_{stem}.json"
        self.status.setText(
            f"{os.path.basename(path)}  fps={self.fps:.1f}  frames={self.total}")
        self._seek_to(0)
        self.toggle_play(force=True)

    def _next_frame(self) -> None:
        ok, f = self.cap.read()
        if not ok:
            self.toggle_play(force=False)
            self.status.setText("播放结束")
            return
        self.frame = f
        self.idx += 1
        self._render()
        if not self._seek_dragging:
            self.seek.blockSignals(True)
            self.seek.setValue(self.idx)
            self.seek.blockSignals(False)

    def _seek_to(self, pos: int) -> None:
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, f = self.cap.read()
        if ok:
            self.frame = f
            self.idx = pos
            self._render()

    def _step_sec(self, sec: int) -> None:
        if self.cap is None:
            return
        target = int(np.clip(self.idx + int(sec * self.fps), 0, max(0, self.total - 1)))
        self._seek_to(target)

    # ---------------- 进度条回调 ----------------

    def _on_seek_pressed(self) -> None:
        self._seek_dragging = True

    def _on_seek_moved(self, pos: int) -> None:
        self._seek_to(pos)

    def _on_seek_released(self) -> None:
        self._seek_dragging = False
        self._seek_to(self.seek.value())

    # ---------------- 增益 / 渲染 ----------------

    def _gains(self) -> tuple[float, float, float]:
        return tuple(self.gain_sliders[n].value() / 100.0 for n in ("R", "G", "B"))

    def _on_gain_changed(self, name: str) -> None:
        getattr(self, f"val_{name}").setText(f"{self.gain_sliders[name].value() / 100.0:.2f}")
        self._render()

    def _render(self) -> None:
        if self.frame is None:
            return
        r, g, b = self._gains()
        f = self.frame.astype(np.float32)
        f[:, :, 0] *= b
        f[:, :, 1] *= g
        f[:, :, 2] *= r
        vis = np.clip(f, 0, 255).astype(np.uint8)

        n_det = -1
        if self.det is not None and self.chk_detect.isChecked():
            from onnx_auto_annotate import draw
            dets = self.det.infer(vis, self.conf, self.iou, 50)
            vis = draw(vis, dets, self.det.names, self.det.num_kpts)
            n_det = len(dets)

        det_txt = f"  dets={n_det}" if n_det >= 0 else ""
        cv2.putText(vis, f"[{self.idx}/{self.total or '?'}] R={r:.2f} G={g:.2f} B={b:.2f}{det_txt}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        self.vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self.frame_lbl.setText(f"{self.idx}/{self.total or '?'}")
        self._show_pixmap()

    def _show_pixmap(self) -> None:
        if self.vis_rgb is None:
            return
        h, w, _ = self.vis_rgb.shape
        img = QImage(self.vis_rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(img).scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas.setPixmap(pix)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._show_pixmap()

    # ---------------- 模型 ----------------

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重", "",
            "YOLO weights (*.pt *.onnx *.engine);;All (*)")
        if path:
            self._load_model(path, "cuda", 640)

    def _load_model(self, path: str, device: str, imgsz: int) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.det = build_detector(path, device, imgsz)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "模型加载失败", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.chk_detect.setEnabled(True)
        self.status.setText(
            f"模型 {os.path.basename(path)}  backend={self.det.backend}  names={self.det.names}")
        self._render()

    # ---------------- 播放控制 / 保存 ----------------

    def toggle_play(self, force: bool | None = None) -> None:
        if self.cap is None:
            return
        self.playing = (not self.playing) if force is None else force
        if self.playing:
            self.timer.start(max(1, int(round(1000.0 / self.fps))))
            self.btn_play.setText("⏸ 暂停")
        else:
            self.timer.stop()
            self.btn_play.setText("▶ 播放")

    def _save_gains(self) -> None:
        r, g, b = self._gains()
        payload = {
            "video": os.path.basename(getattr(self, "video_path", "")),
            "r_gain": round(r, 3),
            "g_gain": round(g, 3),
            "b_gain": round(b, 3),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.save_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        self.status.setText(f"[save] {self.save_path}  R={r:.3f} G={g:.3f} B={b:.3f}")
        print(f"[save] {self.save_path}  R={r:.3f} G={g:.3f} B={b:.3f}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def main() -> None:
    ap = argparse.ArgumentParser(description="TS player GUI (PySide6)")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--save", default=None)
    ap.add_argument("--model", default=None, help="YOLO 权重: .pt / .onnx / .engine")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--init-r", type=float, default=1.0)
    ap.add_argument("--init-g", type=float, default=0.7)
    ap.add_argument("--init-b", type=float, default=1.0)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = PlayerWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
