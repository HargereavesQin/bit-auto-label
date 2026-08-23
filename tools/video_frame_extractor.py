# -*- coding: utf-8 -*-
"""
@File ：video_frame_extractor.py
@Motto :学习新思想，争做新青年
视频抽帧工具 - 支持双帧预览和时间范围选择
"""

import os, sys
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import cv2
import numpy as np
import subprocess
import shutil
import re
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QFrame, QGridLayout,
    QSlider, QProgressBar, QStyle, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPoint
from PySide6.QtGui import QFont, QImage, QPixmap, QPainter, QColor, QPen


def _ffmpeg_available() -> bool:
    """检查系统是否有 ffmpeg."""
    return shutil.which("ffmpeg") is not None


def _ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _is_transport_stream(path: str) -> bool:
    """是否为 MPEG-TS（按帧 seek 极不可靠）。"""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".ts", ".mts", ".m2ts")


def _probe_duration_ffprobe(path: str) -> float:
    """用 ffprobe 读容器时长（秒）；失败返回 0。"""
    if not _ffprobe_available():
        return 0.0
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        ).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def _probe_duration_opencv_scan(path: str, fps_hint: float = 0.0) -> float:
    """OpenCV 顺序读到末尾估时长（慢，仅作 ffprobe 失败回退）。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 25.0
    n = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        n += 1
        if n > 500_000:
            break
    cap.release()
    if fps <= 0:
        fps = 25.0
    return n / fps if n > 0 else 0.0


def _jpeg_quality_to_ffmpeg_q(quality: int) -> int:
    """OpenCV JPEG quality (1-100) → ffmpeg -q:v (2=best, 31=worst)."""
    quality = max(1, min(100, int(quality)))
    return max(2, 31 - int(quality * 29 / 100))


# =========================================================
# 加载编辑风样式 (ui/style_editorial.qss)
# =========================================================
def _load_editorial_qss() -> str:
    qss_path = os.path.join(_ROOT, "ui", "style_editorial.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        return f.read()


EDITORIAL_QSS = _load_editorial_qss()

class RangeSlider(QWidget):
    """双滑块时间范围选择器"""
    range_changed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
        self.setCursor(Qt.PointingHandCursor)

        self._start_ratio = 0.0
        self._end_ratio = 1.0
        self._dragging = None
        self._handle_width = 12
        self._padding = 8

        self.setMinimumSize(400, 40)

    def set_range(self, start_ratio, end_ratio):
        self._start_ratio = max(0.0, min(start_ratio, 1.0))
        self._end_ratio = max(0.0, min(end_ratio, 1.0))
        if self._start_ratio > self._end_ratio:
            self._start_ratio = self._end_ratio
        self.update()
        self.range_changed.emit(self._start_ratio, self._end_ratio)

    def get_range(self):
        return self._start_ratio, self._end_ratio

    def mousePressEvent(self, event):
        pos = event.pos().x()
        handle = self._hit_test(pos)
        if handle:
            self._dragging = handle
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            event.ignore()
            return
        x = event.pos().x()
        ratio = self._pos_to_ratio(x)
        if self._dragging == "start":
            self._start_ratio = max(0.0, min(ratio, self._end_ratio - 0.01))
        else:
            self._end_ratio = min(1.0, max(ratio, self._start_ratio + 0.01))
        self.update()
        self.range_changed.emit(self._start_ratio, self._end_ratio)

    def mouseReleaseEvent(self, event):
        self._dragging = None
        event.ignore()

    def _hit_test(self, x):
        w = self.width()
        track_left = self._padding + self._handle_width // 2
        track_right = w - self._padding - self._handle_width // 2
        track_width = track_right - track_left
        if track_width <= 0:
            return None

        start_x = track_left + self._start_ratio * track_width
        end_x = track_left + self._end_ratio * track_width

        if abs(x - start_x) <= self._handle_width:
            return "start"
        if abs(x - end_x) <= self._handle_width:
            return "end"
        return None

    def _pos_to_ratio(self, x):
        w = self.width()
        track_left = self._padding + self._handle_width // 2
        track_right = w - self._padding - self._handle_width // 2
        ratio = (x - track_left) / (track_right - track_left)
        return max(0.0, min(1.0, ratio))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        track_left = self._padding + self._handle_width // 2
        track_right = w - self._padding - self._handle_width // 2
        track_top = h // 2 - 3
        track_bottom = h // 2 + 3
        track_width = track_right - track_left

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(80, 80, 80))
        painter.drawRect(track_left, track_top, track_width, 6)

        start_x = track_left + self._start_ratio * track_width
        end_x = track_left + self._end_ratio * track_width
        painter.setBrush(QColor(64, 158, 255))
        painter.drawRect(int(start_x), track_top, int(end_x - start_x), 6)

        for xp, color in [(start_x, QColor(64, 158, 255)), (end_x, QColor(255, 140, 0))]:
            painter.setBrush(color)
            painter.drawEllipse(int(xp) - 8, track_top - 5, 16, 16)


class FrameExtractorWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finish_signal = Signal(bool, str)

    def __init__(self, video_path, output_dir, prefix, start_time, end_time,
                 interval, fmt, frame_start, frame_end, jpeg_quality=95, scale_width=0,
                 sequential=False):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.prefix = prefix
        self.start_time = start_time
        self.end_time = end_time
        self.interval = interval
        self.fmt = fmt
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.jpeg_quality = jpeg_quality
        self.scale_width = scale_width
        self.sequential = sequential or _is_transport_stream(video_path)
        self._stop = False

    def stop(self):
        self._stop = True

    def log(self, msg):
        self.log_signal.emit(msg)

    def _imwrite_params(self):
        if self.fmt == "jpg":
            return [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)]
        return [cv2.IMWRITE_PNG_COMPRESSION, 6]

    def _maybe_scale(self, frame):
        if self.scale_width > 0 and frame.shape[1] > self.scale_width:
            h, w = frame.shape[:2]
            new_h = int(h * self.scale_width / w)
            return cv2.resize(frame, (self.scale_width, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def run(self):
        try:
            if self.sequential:
                self._run_sequential()
            else:
                self._run_seek()
        except Exception as e:
            self.finish_signal.emit(False, str(e))

    def _run_seek(self):
        """普通容器：按帧号 seek（快）。"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finish_signal.emit(False, f"无法打开视频: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.log(f"[OpenCV/seek] 视频: {fps:.1f} FPS, 共 {total_frames} 帧")
        self.log(f"[OpenCV/seek] 范围: 第 {self.frame_start}~{self.frame_end} 帧, 间隔 {self.interval:.3f}s")

        os.makedirs(self.output_dir, exist_ok=True)
        step = max(1, int(self.interval * fps))
        target_frames = list(range(self.frame_start, self.frame_end + 1, step))
        self.log(f"[OpenCV/seek] 预计输出: {len(target_frames)} 帧")

        params = self._imwrite_params()
        extracted = 0
        failed = 0
        for idx, frame_idx in enumerate(target_frames):
            if self._stop:
                self.log("用户取消")
                self.finish_signal.emit(False, "用户取消")
                cap.release()
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                failed += 1
                continue
            frame = self._maybe_scale(frame)
            output_name = f"{self.prefix}_{frame_idx:06d}.{self.fmt}"
            cv2.imwrite(os.path.join(self.output_dir, output_name), frame, params)
            extracted += 1
            self.progress_signal.emit(idx + 1, len(target_frames))

        cap.release()
        self.log(
            f"[OpenCV/seek] 完成: 预计 {len(target_frames)} / 实得 {extracted} / 解码失败 {failed}"
        )
        self.finish_signal.emit(
            True,
            f"成功抽取 {extracted} 帧 (OpenCV/seek; 预计 {len(target_frames)}, 失败 {failed})",
        )

    def _run_sequential(self):
        """TS / 不可靠元数据：顺序 read，按时间间隔取帧（可靠）。"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finish_signal.emit(False, f"无法打开视频: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 1e-3:
            fps = 25.0
        self.log(f"[OpenCV/seq] 顺序解码 (TS 友好), FPS≈{fps:.1f}")
        self.log(
            f"[OpenCV/seq] 时间范围: {self.start_time:.3f}s ~ {self.end_time:.3f}s, "
            f"间隔 {self.interval:.3f}s"
        )

        os.makedirs(self.output_dir, exist_ok=True)
        duration = max(0.0, self.end_time - self.start_time)
        out_fps = 1.0 / self.interval if self.interval > 0 else 1.0
        expected = max(1, int(duration * out_fps) + 1)
        self.log(f"[OpenCV/seq] 预计输出: ~{expected} 帧")

        params = self._imwrite_params()
        extracted = 0
        failed = 0
        frame_i = 0
        next_t = self.start_time
        end_t = self.end_time

        while True:
            if self._stop:
                self.log("用户取消")
                self.finish_signal.emit(False, "用户取消")
                cap.release()
                return

            ret, frame = cap.read()
            if not ret:
                break

            t = frame_i / fps
            # 也尝试 POS_MSEC（部分后端更准）
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if msec and msec > 0:
                t = msec / 1000.0

            frame_i += 1
            if t < self.start_time - 1e-3:
                continue
            if t > end_t + 1e-3:
                break

            if t + 1e-6 < next_t:
                continue

            if frame is None:
                failed += 1
                next_t += self.interval
                continue

            frame = self._maybe_scale(frame)
            # 用毫秒时间戳命名，避免依赖不可靠帧号
            stamp = int(round(t * 1000))
            output_name = f"{self.prefix}_{stamp:08d}.{self.fmt}"
            ok = cv2.imwrite(os.path.join(self.output_dir, output_name), frame, params)
            if not ok:
                failed += 1
            else:
                extracted += 1
            next_t += self.interval
            self.progress_signal.emit(min(extracted, expected), expected)

        cap.release()
        self.log(
            f"[OpenCV/seq] 完成: 预计 ~{expected} / 实得 {extracted} / 写失败 {failed}"
        )
        self.finish_signal.emit(
            True,
            f"成功抽取 {extracted} 帧 (OpenCV/seq; 预计 ~{expected}, 失败 {failed})",
        )


class FFmpegExtractorWorker(QThread):
    """FFmpeg 后端: 硬件解码 + 多线程编码; TS 用容错参数 + -ss 在 -i 后。"""
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finish_signal = Signal(bool, str)

    def __init__(self, video_path, output_dir, prefix, start_time, end_time,
                 interval, fmt, fps, jpeg_quality=95, scale_width=0):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.prefix = prefix
        self.start_time = start_time
        self.end_time = end_time
        self.interval = interval
        self.fmt = fmt
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.scale_width = scale_width
        self._ts_mode = _is_transport_stream(video_path)
        self._stop = False
        self._process = None
        self._tmp_mp4 = None

    def stop(self):
        self._stop = True
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass

    def log(self, msg):
        self.log_signal.emit(msg)

    def _cleanup_tmp(self):
        if self._tmp_mp4 and os.path.isfile(self._tmp_mp4):
            try:
                os.remove(self._tmp_mp4)
            except OSError:
                pass
            self._tmp_mp4 = None

    def _remux_ts_to_mp4(self):
        """TS 无损重封装到临时 mp4，失败返回 None。"""
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="armorlabel_ts_")
        os.close(fd)
        os.remove(tmp)  # ffmpeg 自己创建
        cmd = [
            "ffmpeg", "-y",
            "-fflags", "+genpts+igndts+discardcorrupt",
            "-err_detect", "ignore_err",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-i", self.video_path,
            "-c", "copy",
            "-movflags", "+faststart",
            tmp,
        ]
        self.log("[FFmpeg] TS → 临时 mp4 重封装中…")
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if p.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
                self._tmp_mp4 = tmp
                self.log(f"[FFmpeg] 重封装成功: {tmp}")
                return tmp
            self.log(f"[FFmpeg] 重封装失败 (ret={p.returncode}), 将直接读 TS")
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None
        except Exception as e:
            self.log(f"[FFmpeg] 重封装异常: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None

    def _run_ffmpeg(self, cmd):
        """运行 ffmpeg 命令, 返回 (ret_code, stderr_text, file_count)."""
        import threading
        import glob

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, bufsize=1)

        stderr_lines = []

        def _drain_stderr():
            try:
                for line in self._process.stderr:
                    stderr_lines.append(line)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        duration = self.end_time - self.start_time
        out_fps = 1.0 / self.interval if self.interval > 0 else 1.0
        total_est = max(1, int(duration * out_fps) + 1)
        current_us = 0

        for line in self._process.stdout:
            if self._stop:
                break
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    current_us = int(line.split("=")[1])
                except ValueError:
                    pass
            elif line.startswith("progress=end"):
                break
            progress_sec = current_us / 1_000_000.0
            progress_frame = min(int(progress_sec * out_fps) + 1, total_est)
            self.progress_signal.emit(progress_frame, total_est)

        ret_code = self._process.wait()
        stderr_thread.join(timeout=5)
        stderr_text = "".join(stderr_lines)

        file_count = len(glob.glob(os.path.join(
            self.output_dir, f"{self.prefix}_*.{self.fmt}")))
        return ret_code, stderr_text, file_count, total_est

    def _build_cmd(self, input_path, hwaccel, accurate_seek: bool):
        duration = self.end_time - self.start_time
        out_fps = 1.0 / self.interval if self.interval > 0 else 1.0
        vf_parts = [f"fps={out_fps:.6f}"]
        if self.scale_width > 0:
            vf_parts.append(f"scale={int(self.scale_width)}:-2")
        vf = ",".join(vf_parts)
        output_pattern = os.path.join(self.output_dir, f"{self.prefix}_%06d.{self.fmt}")

        encode_args = []
        if self.fmt == "jpg":
            encode_args = ["-q:v", str(_jpeg_quality_to_ffmpeg_q(self.jpeg_quality))]
        else:
            encode_args = ["-compression_level", "6"]

        cmd = ["ffmpeg", "-y"]
        if accurate_seek or self._ts_mode:
            cmd += [
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-err_detect", "ignore_err",
                "-analyzeduration", "100M",
                "-probesize", "100M",
            ]
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]

        if accurate_seek:
            # -ss 在 -i 之后：精确但更慢，TS 必需
            cmd += [
                "-i", input_path,
                "-ss", f"{self.start_time:.3f}",
                "-t", f"{duration:.3f}",
            ]
        else:
            cmd += [
                "-ss", f"{self.start_time:.3f}",
                "-i", input_path,
                "-t", f"{duration:.3f}",
            ]

        cmd += [
            "-vf", vf,
            "-an", "-sn",
            "-vsync", "vfr",
            "-progress", "pipe:1",
            "-nostats",
        ] + encode_args + [output_pattern]
        return cmd

    def run(self):
        import glob
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            duration = self.end_time - self.start_time
            out_fps = 1.0 / self.interval if self.interval > 0 else 1.0
            expected = max(1, int(duration * out_fps) + 1)

            input_path = self.video_path
            accurate = self._ts_mode

            if self._ts_mode:
                remuxed = self._remux_ts_to_mp4()
                if remuxed:
                    input_path = remuxed
                    # 重封装后可用快速 -ss（仍带容错）
                    accurate = False
                    self.log("[FFmpeg] 将从临时 mp4 抽帧")
                else:
                    self.log("[FFmpeg] 直接读 TS，使用 -ss 在 -i 后（精确/较慢）")

            strategies = []
            if accurate:
                strategies = [(None, True), ("auto", True)]
            else:
                strategies = [("auto", False), (None, False), (None, True)]

            last_err = ""
            for hwaccel, acc in strategies:
                if self._stop:
                    break
                # 清理上次残留
                for f in glob.glob(os.path.join(self.output_dir, f"{self.prefix}_*.{self.fmt}")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

                cmd = self._build_cmd(input_path, hwaccel, acc)
                label = f"FFmpeg{'/' + hwaccel if hwaccel else '/soft'}{'/acc' if acc else '/fast'}"
                self.log(
                    f"[{label}] 开始: start={self.start_time:.3f}s dur={duration:.3f}s "
                    f"fps={out_fps:.3f} 预计~{expected}"
                )
                if self.scale_width > 0:
                    self.log(f"[{label}] 缩放至 {self.scale_width}px 宽")

                ret_code, stderr_text, file_count, total_est = self._run_ffmpeg(cmd)

                if self._stop:
                    self.log(f"[{label}] 用户取消")
                    self.finish_signal.emit(False, "用户取消")
                    return

                if ret_code == 0 and file_count > 0:
                    miss = max(0, expected - file_count)
                    self.log(
                        f"[{label}] 完成: 预计 ~{expected} / 实得 {file_count} / 相对缺口 ~{miss}"
                    )
                    self.finish_signal.emit(
                        True,
                        f"成功抽取 {file_count} 帧 ({label}; 预计 ~{expected})",
                    )
                    return

                last_err = stderr_text[-800:] if stderr_text else ""
                self.log(
                    f"[{label}] 失败 (ret={ret_code}, files={file_count}): {last_err[-200:]}"
                )

            self.finish_signal.emit(
                False, f"FFmpeg 抽取失败, 无输出文件\n{last_err}"
            )
        except FileNotFoundError:
            self.finish_signal.emit(False, "未找到 ffmpeg, 请检查 PATH")
        except Exception as e:
            self.finish_signal.emit(False, str(e))
        finally:
            self._process = None
            self._cleanup_tmp()


class VideoFrameExtractorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from ui.icons import Icon, Icons, C
        from ui.dialog_utils import build_dialog_header
        self.setWindowTitle("视频抽帧工具")
        self.setWindowIcon(Icon.icon(Icons.VIDEO, size=24, color=C.AMBER))
        self.setMinimumSize(900, 700)
        self.resize(960, 760)
        self.setStyleSheet(EDITORIAL_QSS)

        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.duration = 0.0
        self._unreliable_index = False
        self.is_playing = False
        self.play_timer = QTimer()

        self._setup_ui()
        self._connect_signals()
        self._load_config()

    def _setup_ui(self):
        from ui.icons import Icons, C
        from ui.dialog_utils import build_dialog_header

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 标题条
        outer.addWidget(build_dialog_header(
            self, Icons.VIDEO,
            "视频抽帧工具", "VIDEO FRAME EXTRACTOR  /  按间隔抽帧"))

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 8, 16, 8)
        main_layout.setSpacing(10)
        outer.addLayout(main_layout, 1)

        font_bold = QFont("Microsoft YaHei", 10, QFont.Bold)
        font_normal = QFont("Microsoft YaHei", 9)

        top_group = QWidget()
        top_layout = QGridLayout(top_group)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        lbl_video = QLabel("视频文件:")
        lbl_video.setFont(font_bold)
        self.edt_video = QLineEdit()
        self.edt_video.setFont(font_normal)
        self.edt_video.setPlaceholderText("选择视频文件路径...")
        self.btn_video = QPushButton("选择文件")
        self.btn_video.setFont(font_normal)
        self.btn_video.setProperty("role", "tag")
        self.btn_video.setCursor(Qt.PointingHandCursor)
        top_layout.addWidget(lbl_video, 0, 0, 1, 1)
        top_layout.addWidget(self.edt_video, 0, 1, 1, 4)
        top_layout.addWidget(self.btn_video, 0, 5, 1, 1)

        lbl_out = QLabel("输出目录:")
        lbl_out.setFont(font_bold)
        self.edt_output = QLineEdit()
        self.edt_output.setFont(font_normal)
        self.edt_output.setPlaceholderText("选择输出目录...")
        self.btn_output = QPushButton("选择目录")
        self.btn_output.setFont(font_normal)
        self.btn_output.setProperty("role", "tag")
        self.btn_output.setCursor(Qt.PointingHandCursor)
        top_layout.addWidget(lbl_out, 1, 0, 1, 1)
        top_layout.addWidget(self.edt_output, 1, 1, 1, 4)
        top_layout.addWidget(self.btn_output, 1, 5, 1, 1)

        lbl_prefix = QLabel("文件前缀:")
        lbl_prefix.setFont(font_bold)
        self.edt_prefix = QLineEdit()
        self.edt_prefix.setFont(font_normal)
        self.edt_prefix.setPlaceholderText("视频文件名")
        lbl_fmt = QLabel("输出格式:")
        lbl_fmt.setFont(font_bold)
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["jpg", "png"])
        self.cmb_format.setFont(font_normal)
        self.cmb_format.setCurrentText("jpg")

        lbl_quality = QLabel("JPEG质量:")
        lbl_quality.setObjectName("lbl_quality")
        lbl_quality.setFont(font_bold)
        self.spin_quality = QSpinBox()
        self.spin_quality.setFont(font_normal)
        self.spin_quality.setRange(1, 100)
        self.spin_quality.setValue(92)
        self.spin_quality.setSuffix(" %")

        lbl_scale = QLabel("缩放宽度:")
        lbl_scale.setFont(font_bold)
        self.spin_scale = QSpinBox()
        self.spin_scale.setFont(font_normal)
        self.spin_scale.setRange(0, 32768)
        self.spin_scale.setValue(0)
        self.spin_scale.setSuffix(" px")
        self.spin_scale.setSpecialValueText("原尺寸")
        self.spin_scale.setToolTip("0 = 原始分辨率; 设为如 1920 则按比例缩小到该宽度\n"
                                    "6K 视频缩到 1920 可快 5-10x 且输出文件小很多")

        top_layout.addWidget(lbl_prefix, 2, 0, 1, 1)
        top_layout.addWidget(self.edt_prefix, 2, 1, 1, 2)
        top_layout.addWidget(lbl_fmt, 2, 3, 1, 1)
        top_layout.addWidget(self.cmb_format, 2, 4, 1, 1)
        top_layout.addWidget(lbl_quality, 2, 5, 1, 1)
        top_layout.addWidget(self.spin_quality, 2, 6, 1, 1)

        top_layout.addWidget(lbl_scale, 3, 0, 1, 1)
        top_layout.addWidget(self.spin_scale, 3, 1, 1, 2)

        # 后端检测标签
        self.lbl_backend = QLabel()
        self.lbl_backend.setFont(QFont("Microsoft YaHei", 8))
        ff_ok = _ffmpeg_available()
        if ff_ok:
            self.lbl_backend.setText("后端: FFmpeg 硬件加速 (已检测到 ffmpeg)")
            self.lbl_backend.setProperty("role", "accent")
        else:
            self.lbl_backend.setText("后端: OpenCV (未检测到 ffmpeg, 建议安装 ffmpeg 加速)")
            self.lbl_backend.setProperty("role", "warning")
        top_layout.addWidget(self.lbl_backend, 3, 3, 1, 5)

        main_layout.addWidget(top_group)

        preview_group = QWidget()
        preview_layout = QHBoxLayout(preview_group)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(16)

        self.lbl_start_preview = self._make_preview_label("起始帧")
        self.lbl_end_preview = self._make_preview_label("结束帧")
        preview_layout.addWidget(self.lbl_start_preview)
        preview_layout.addWidget(self.lbl_end_preview)

        main_layout.addWidget(preview_group, 1)

        self.lbl_start_info = QLabel("起始帧: 0  时间: 00:00.000")
        self.lbl_start_info.setFont(font_normal)
        self.lbl_start_info.setAlignment(Qt.AlignCenter)
        self.lbl_end_info = QLabel("结束帧: 0  时间: 00:00.000")
        self.lbl_end_info.setAlignment(Qt.AlignCenter)
        self.lbl_end_info.setFont(font_normal)

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.lbl_start_info)
        info_layout.addStretch(1)
        info_layout.addWidget(self.lbl_end_info)

        main_layout.addLayout(info_layout)

        slider_container = QWidget()
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 4, 0, 4)
        slider_layout.setSpacing(4)

        self.range_slider = RangeSlider()
        slider_layout.addWidget(self.range_slider)

        slider_hint_layout = QHBoxLayout()
        slider_hint_layout.addStretch(1)
        self.lbl_timeline = QLabel("时间轴 (拖动滑块选择抽帧范围)")
        self.lbl_timeline.setFont(QFont("Microsoft YaHei", 8))
        self.lbl_timeline.setAlignment(Qt.AlignCenter)
        slider_hint_layout.addWidget(self.lbl_timeline)
        slider_hint_layout.addStretch(1)
        slider_layout.addLayout(slider_hint_layout)

        main_layout.addWidget(slider_container)

        time_range_widget = QWidget()
        time_range_layout = QGridLayout(time_range_widget)
        time_range_layout.setContentsMargins(0, 0, 0, 0)
        time_range_layout.setSpacing(8)

        lbl_start = QLabel("起始时间(秒):")
        lbl_start.setFont(font_bold)
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setFont(font_normal)
        self.spin_start.setRange(0, 9999)
        self.spin_start.setDecimals(3)
        self.spin_start.setValue(0)
        self.spin_start.setSuffix(" s")
        self.spin_start.setMinimumWidth(120)

        lbl_end = QLabel("结束时间(秒):")
        lbl_end.setFont(font_bold)
        self.spin_end = QDoubleSpinBox()
        self.spin_end.setFont(font_normal)
        self.spin_end.setRange(0, 9999)
        self.spin_end.setDecimals(3)
        self.spin_end.setValue(10)
        self.spin_end.setSuffix(" s")
        self.spin_end.setMinimumWidth(120)

        lbl_interval = QLabel("抽帧间隔:")
        lbl_interval.setFont(font_bold)
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setFont(font_normal)
        self.spin_interval.setRange(0.001, 60)
        self.spin_interval.setDecimals(3)
        self.spin_interval.setValue(0.5)
        self.spin_interval.setSuffix(" s")
        self.spin_interval.setMinimumWidth(120)

        lbl_est = QLabel("预计输出:")
        lbl_est.setFont(font_bold)
        self.lbl_estimation = QLabel("0 帧")
        self.lbl_estimation.setFont(font_normal)
        self.lbl_estimation.setProperty("role", "accent")

        time_range_layout.addWidget(lbl_start, 0, 0)
        time_range_layout.addWidget(self.spin_start, 0, 1)
        time_range_layout.addWidget(lbl_end, 0, 2)
        time_range_layout.addWidget(self.spin_end, 0, 3)
        time_range_layout.addWidget(lbl_interval, 0, 4)
        time_range_layout.addWidget(self.spin_interval, 0, 5)
        time_range_layout.addWidget(lbl_est, 0, 6)
        time_range_layout.addWidget(self.lbl_estimation, 0, 7)
        time_range_layout.setColumnStretch(6, 1)
        time_range_layout.setColumnStretch(7, 1)

        main_layout.addWidget(time_range_widget)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(font_normal)
        self.progress_bar.setMinimumHeight(20)
        self.lbl_progress = QLabel("就绪")
        self.lbl_progress.setFont(font_normal)
        self.lbl_progress.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.lbl_progress)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_start = QPushButton("  开始抽帧")
        self.btn_start.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.btn_start.setProperty("role", "primary")
        self.btn_start.setCursor(Qt.PointingHandCursor)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFont(QFont("Microsoft YaHei", 10))
        self.btn_cancel.setProperty("role", "ghost")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setEnabled(False)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)

        main_layout.addLayout(btn_layout)

        self.lbl_status = QLabel("请选择视频文件")
        self.lbl_status.setFont(QFont("Microsoft YaHei", 8))
        self.lbl_status.setProperty("role", "muted")
        self.lbl_status.setAlignment(Qt.AlignRight)
        main_layout.addWidget(self.lbl_status)

    def _make_preview_label(self, title):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(title)
        lbl.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        lbl.setAlignment(Qt.AlignCenter)

        frame = QLabel()
        frame.setMinimumSize(320, 200)
        frame.setMaximumSize(480, 320)
        frame.setProperty("role", "frame_preview")
        frame.setScaledContents(False)
        frame.setAlignment(Qt.AlignCenter)

        no_video = QLabel("未加载视频")
        no_video.setFont(QFont("Microsoft YaHei", 9))
        no_video.setProperty("role", "caption")
        no_video.setAlignment(Qt.AlignCenter)
        no_video.setFixedSize(320, 200)
        no_video.setParent(frame)

        layout.addWidget(lbl)
        layout.addWidget(frame)

        container.lbl = lbl
        container.frame = frame
        container.no_video = no_video

        return container

    def _connect_signals(self):
        self.btn_video.clicked.connect(self._on_select_video)
        self.btn_output.clicked.connect(self._on_select_output)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel.clicked.connect(self._on_cancel)

        self.spin_start.valueChanged.connect(self._on_spin_start_changed)
        self.spin_end.valueChanged.connect(self._on_spin_end_changed)
        self.spin_interval.valueChanged.connect(self._update_estimation)

        self.cmb_format.currentTextChanged.connect(self._on_format_changed)
        self.range_slider.range_changed.connect(self._on_range_changed)

        self.play_timer.timeout.connect(self._on_play_tick)
        self._on_format_changed(self.cmb_format.currentText())

    def _load_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig().get_tool_config("video_frame_extractor")
        self.edt_video.setText(cfg.get("video_path", ""))
        self.edt_output.setText(cfg.get("output_dir", ""))
        self.edt_prefix.setText(cfg.get("prefix", ""))
        self.cmb_format.setCurrentText(cfg.get("format", "jpg"))
        self.spin_interval.setValue(float(cfg.get("interval", 0.5)))
        self.spin_quality.setValue(int(cfg.get("jpeg_quality", 92)))
        self.spin_scale.setValue(int(cfg.get("scale_width", 0)))
        if self.edt_video.text():
            self._on_select_video()

    def _save_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig()
        cfg.set("video_frame_extractor", "video_path", self.edt_video.text())
        cfg.set("video_frame_extractor", "output_dir", self.edt_output.text())
        cfg.set("video_frame_extractor", "prefix", self.edt_prefix.text())
        cfg.set("video_frame_extractor", "format", self.cmb_format.currentText())
        cfg.set("video_frame_extractor", "interval", self.spin_interval.value())
        cfg.set("video_frame_extractor", "jpeg_quality", self.spin_quality.value())
        cfg.set("video_frame_extractor", "scale_width", self.spin_scale.value())
        cfg.save()

    def _get_default_dir(self, category="origin_data"):
        base = _ROOT
        if category == "origin_data":
            d = os.path.join(base, "origin_data")
            return d if os.path.exists(d) else base
        elif category == "yolo":
            d = os.path.join(base, "runs")
            return d if os.path.exists(d) else base
        elif category == "mlp":
            d = os.path.join(base, "stored_weights", "model", "armor_tradition")
            return d if os.path.exists(d) else base
        return base

    def _on_select_video(self):
        default_dir = self._get_default_dir("origin_data")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", default_dir,
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v *.ts *.3gp *.mts *.m2ts);;所有文件 (*)"
        )
        if not path:
            return

        self.edt_video.setText(path)
        prefix = os.path.splitext(os.path.basename(path))[0]
        self.edt_prefix.setText(prefix)

        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if self.fps <= 1e-3:
            self.fps = 25.0

        self._unreliable_index = (
            _is_transport_stream(path) or self.total_frames <= 0
        )
        self.duration = 0.0
        if self.total_frames > 0 and self.fps > 0 and not _is_transport_stream(path):
            self.duration = self.total_frames / self.fps

        if self.duration <= 0 or self._unreliable_index:
            probed = _probe_duration_ffprobe(path)
            if probed > 0:
                self.duration = probed
                if self.total_frames <= 0:
                    self.total_frames = max(1, int(probed * self.fps))
                self.lbl_status.setText(
                    f"探测时长(ffprobe): {self._format_time(self.duration)} …"
                )
                QApplication.processEvents()
            else:
                self.lbl_status.setText("ffprobe 失败，正在扫描估时长（可能较慢）…")
                QApplication.processEvents()
                scanned = _probe_duration_opencv_scan(path, self.fps)
                if scanned > 0:
                    self.duration = scanned
                    if self.total_frames <= 0:
                        self.total_frames = max(1, int(scanned * self.fps))
                # 重新打开（扫描会耗尽 cap）
                self.cap.release()
                self.cap = cv2.VideoCapture(path)

        if self.duration <= 0:
            QMessageBox.warning(self, "错误", "无法获取视频时长，请检查文件是否损坏")
            return

        hint = ""
        if self._unreliable_index:
            hint = " | TS/顺序解码模式"

        self.lbl_start_preview.no_video.hide()
        self.lbl_end_preview.no_video.hide()

        self.spin_start.blockSignals(True)
        self.spin_end.blockSignals(True)

        self.spin_start.setMaximum(self.duration)
        self.spin_end.setMaximum(self.duration)
        self.spin_start.setValue(0)
        self.spin_end.setValue(self.duration)

        self.spin_start.blockSignals(False)
        self.spin_end.blockSignals(False)

        self.range_slider.set_range(0, 1.0)

        self._update_estimation()
        self._update_preview_start()
        self._update_preview_end()

        self.lbl_status.setText(
            f"视频: {self.total_frames} 帧 | {self.fps:.1f} FPS | "
            f"时长: {self._format_time(self.duration)}{hint}"
        )
        self.btn_start.setEnabled(True)

    def _on_select_output(self):
        default_dir = self._get_default_dir("origin_data")
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", default_dir)
        if path:
            self.edt_output.setText(path)

    def _on_range_changed(self, start_ratio, end_ratio):
        if self.cap is None or self.duration <= 0:
            return
        start_time = start_ratio * self.duration
        end_time = end_ratio * self.duration
        self.spin_start.blockSignals(True)
        self.spin_end.blockSignals(True)
        self.spin_start.setValue(start_time)
        self.spin_end.setValue(end_time)
        self.spin_start.blockSignals(False)
        self.spin_end.blockSignals(False)
        self._update_preview_start()
        self._update_preview_end()
        self._update_estimation()

    def _on_spin_start_changed(self, value):
        if self.cap is None or self.duration <= 0:
            return
        end_val = self.spin_end.value()
        if value > end_val:
            self.spin_start.blockSignals(True)
            self.spin_start.setValue(end_val)
            self.spin_start.blockSignals(False)
            return
        ratio = value / self.duration
        start_ratio, end_ratio = self.range_slider.get_range()
        self.range_slider.set_range(ratio, end_ratio)
        self._update_preview_start()
        self._update_estimation()

    def _on_spin_end_changed(self, value):
        if self.cap is None or self.duration <= 0:
            return
        start_val = self.spin_start.value()
        if value < start_val:
            self.spin_end.blockSignals(True)
            self.spin_end.setValue(start_val)
            self.spin_end.blockSignals(False)
            return
        ratio = value / self.duration
        start_ratio, end_ratio = self.range_slider.get_range()
        self.range_slider.set_range(start_ratio, ratio)
        self._update_preview_end()
        self._update_estimation()

    def _update_estimation(self):
        if self.cap is None:
            return
        start_t = self.spin_start.value()
        end_t = self.spin_end.value()
        interval = self.spin_interval.value()
        if interval <= 0 or end_t <= start_t:
            self.lbl_estimation.setText("0 帧")
            return
        fps_est = 1.0 / interval if interval > 0 else 0
        count = int((end_t - start_t) * fps_est) + 1
        self.lbl_estimation.setText(f"{count} 帧")

    def _update_preview_start(self):
        if self.cap is None:
            return
        t = self.spin_start.value()
        frame_idx = int(t * self.fps)
        self._show_frame_at(self.lbl_start_preview, frame_idx, t)

    def _update_preview_end(self):
        if self.cap is None:
            return
        t = self.spin_end.value()
        frame_idx = int(t * self.fps)
        self._show_frame_at(self.lbl_end_preview, frame_idx, t)

    def _show_frame_at(self, preview_widget, frame_idx, time_sec):
        if self.cap is None:
            return
        frame = self._get_frame_at_time(time_sec)
        if frame is None:
            return
        if self.fps > 0:
            frame_idx = int(time_sec * self.fps)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        max_w = preview_widget.frame.width()
        max_h = preview_widget.frame.height()
        if max_w <= 0 or max_h <= 0:
            max_w, max_h = 320, 200

        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = QImage(frame_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        preview_widget.frame.setPixmap(pixmap)
        preview_widget.frame.setFixedSize(new_w, new_h)
        preview_widget.lbl.setText(f"t={self._format_time(time_sec)}")

        if preview_widget == self.lbl_start_preview:
            self.lbl_start_info.setText(
                f"起始: ~帧{frame_idx}  时间: {self._format_time(time_sec)}"
            )
        else:
            self.lbl_end_info.setText(
                f"结束: ~帧{frame_idx}  时间: {self._format_time(time_sec)}"
            )

    def _format_time(self, seconds):
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:06.3f}"

    def _get_frame_at_time(self, time_sec):
        if self.cap is None:
            return None
        time_sec = max(0.0, min(float(time_sec), max(0.0, self.duration - 1e-3)))
        # TS / 不可靠索引：按毫秒 seek，失败再回退帧号
        if getattr(self, "_unreliable_index", False):
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
            ret, frame = self.cap.read()
            if ret:
                return frame
        frame_idx = int(time_sec * self.fps) if self.fps > 0 else 0
        if self.total_frames > 0:
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return frame
        # 最后手段：毫秒 seek
        self.cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _on_format_changed(self, fmt_text):
        """JPEG 质量仅在 jpg 格式时有意义."""
        is_jpg = (fmt_text == "jpg")
        self.spin_quality.setEnabled(is_jpg)
        # 找到 quality label 并同步禁用
        lbl_q = self.findChild(QLabel, "lbl_quality")
        if lbl_q:
            lbl_q.setEnabled(is_jpg)

    def _on_start(self):
        video_path = self.edt_video.text().strip()
        output_dir = self.edt_output.text().strip()

        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "错误", "请选择有效的视频文件")
            return
        if not output_dir:
            QMessageBox.warning(self, "错误", "请选择输出目录")
            return

        prefix = self.edt_prefix.text().strip() or "frame"
        fmt = self.cmb_format.currentText()
        start_t = self.spin_start.value()
        end_t = self.spin_end.value()
        interval = self.spin_interval.value()
        jpeg_quality = self.spin_quality.value()
        scale_width = self.spin_scale.value()

        if interval <= 0:
            QMessageBox.warning(self, "错误", "抽帧间隔必须大于0")
            return
        if end_t <= start_t:
            QMessageBox.warning(self, "错误", "结束时间必须大于起始时间")
            return

        frame_start = int(start_t * self.fps)
        frame_end = int(end_t * self.fps)

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_progress.setText("正在抽帧...")
        scale_hint = f" 缩放→{scale_width}px" if scale_width > 0 else ""
        self.lbl_status.setText(
            f"处理中: {frame_start}~{frame_end} 帧{scale_hint}")

        # 按后端可用性分派: FFmpeg 优先 (硬件加速), OpenCV fallback
        sequential = getattr(self, "_unreliable_index", False) or _is_transport_stream(video_path)
        if _ffmpeg_available():
            self.worker = FFmpegExtractorWorker(
                video_path, output_dir, prefix, start_t, end_t,
                interval, fmt, self.fps, jpeg_quality, scale_width)
        else:
            self.worker = FrameExtractorWorker(
                video_path, output_dir, prefix, start_t, end_t,
                interval, fmt, frame_start, frame_end, jpeg_quality, scale_width,
                sequential=sequential)
        self.worker.log_signal.connect(self._on_log)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finish_signal.connect(self._on_finished)
        self.worker.start()

    def _on_cancel(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_progress.setText("已取消")
        self.lbl_status.setText("就绪")

    def _on_log(self, msg):
        self.lbl_progress.setText(msg)

    def _on_progress(self, current, total):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self.lbl_progress.setText(f"抽帧中: {current}/{total} ({pct}%)")

    def _on_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self.lbl_progress.setText(msg)
            self.lbl_status.setText("抽帧完成")
            QMessageBox.information(self, "完成", msg)
        else:
            self.lbl_progress.setText(f"失败: {msg}")
            self.lbl_status.setText("失败")
            QMessageBox.warning(self, "错误", msg)

    def closeEvent(self, event):
        self._save_config()
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if self.cap:
            self.cap.release()
        event.accept()

    def _on_play_tick(self):
        if self.cap is None:
            self.is_playing = False
            self.play_timer.stop()
            return
        current_ratio = self.range_slider.get_range()[0]
        current_ratio += 0.01
        if current_ratio >= self.range_slider.get_range()[1]:
            current_ratio = self.range_slider.get_range()[0]
        self.range_slider.set_range(current_ratio, self.range_slider.get_range()[1])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = VideoFrameExtractorWindow()
    window.show()
    sys.exit(app.exec())
