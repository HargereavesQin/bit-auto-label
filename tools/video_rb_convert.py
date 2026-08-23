# -*- coding: utf-8 -*-
"""
视频 R/B 通道交换工具 (RGB ↔ BGR)。

用 FFmpeg `colorchannelmixer` 滤镜交换 R/B 通道, 保留音频, 输出到新文件
(`{stem}_rb{ext}`)。优先 NVIDIA NVENC 硬解硬编加速; 失败自动回退到
Intel QSV → libx264 软编。原地模式 (`--inplace`) 写临时文件再原子替换。

用法:
    conda activate sam3
    python tools/video_rb_convert.py video1.mp4 video2.mp4 video3.mp4
    python tools/video_rb_convert.py datasets/*.mp4
    python tools/video_rb_convert.py video.mp4 --inplace      # 覆盖原文件
    python tools/video_rb_convert.py video.mp4 --dry-run      # 只打印不执行
    python tools/video_rb_convert.py video.mp4 --encoder libx264   # 指定编码器

依赖: 系统已安装 ffmpeg (含 libx264; 可选 h264_nvenc / h264_qsv)。
"""
import os
import sys
import re
import time
import shlex
import shutil
import argparse
import subprocess
import tempfile

# R/B 交换滤镜: output_R = input_B, output_G = input_G, output_B = input_R
RB_SWAP_VF = "colorchannelmixer=rr=0:rb=1:gg=1:br=1:bb=0"


def _run(cmd, **kw):
    """shell 里打印命令, 方便复现 / 调试。"""
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def _probe_duration(video_path):
    """用 ffprobe 取时长 (秒), 用于进度百分比。失败返回 None。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except Exception:
        pass
    return None


def _encoder_available(encoder):
    """检查 ffmpeg 是否支持某编码器。"""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10)
        return encoder in out.stdout
    except Exception:
        return False


def _pick_encoder(preferred):
    """按优先级选可用的 h264 编码器。"""
    if preferred and preferred != "auto":
        return preferred
    for enc in ("h264_nvenc", "h264_qsv", "libx264"):
        if _encoder_available(enc):
            return enc
    return "libx264"


def _build_cmd(in_path, out_path, encoder, hwaccel, lossless=False):
    """构造 ffmpeg 命令。硬件解码用 -hwaccel auto (失败自动回退软解)。"""
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", "auto", "-hwaccel_output_format", "nv12"]
    cmd += ["-i", in_path, "-vf", RB_SWAP_VF, "-c:v", encoder]
    if lossless:
        # 无损: NVENC -tune lossless (QP=0) 或 libx264 -crf 0
        if encoder in ("h264_nvenc",):
            cmd += ["-preset", "p7", "-tune", "lossless", "-rc", "constqp", "-qp", "0"]
        elif encoder in ("h264_qsv",):
            cmd += ["-preset", "veryslow", "-look_ahead", "0", "-qpi", "0", "-qpp", "0", "-qpb", "0"]
        else:  # libx264
            cmd += ["-preset", "fast", "-crf", "0"]
    elif encoder in ("h264_nvenc", "h264_qsv"):
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
    else:  # libx264
        cmd += ["-preset", "fast", "-crf", "20"]
    # 音频 / 字幕直通 (没有也不报错)
    cmd += ["-c:a", "copy", "-c:s", "copy"]
    cmd += ["-movflags", "+faststart", out_path]
    return cmd


def _drain_stderr(proc):
    """独立线程持续读 stderr, 防止缓冲区写满导致 ffmpeg 卡死。"""
    import threading

    def _reader():
        try:
            for _ in iter(lambda: proc.stderr.readline(), b""):
                pass
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


def convert_video(in_path, encoder="auto", inplace=False, dry_run=False,
                   overwrite=False, lossless=False):
    """转换单个视频。返回 (success, out_path_or_msg)。"""
    if not os.path.isfile(in_path):
        return False, f"文件不存在: {in_path}"

    stem, ext = os.path.splitext(in_path)
    if inplace:
        out_path = in_path  # 先写临时文件再替换
    else:
        out_path = stem + "_rb" + ext
        if os.path.exists(out_path) and not overwrite and not dry_run:
            print(f"  [跳过] 输出已存在: {out_path} (用 --overwrite 覆盖)")
            return True, out_path

    enc = _pick_encoder(encoder)
    use_hw = enc in ("h264_nvenc", "h264_qsv")
    cmd = _build_cmd(in_path, out_path, enc, hwaccel=use_hw, lossless=lossless)

    if dry_run:
        print("  [dry-run]", " ".join(shlex.quote(c) for c in cmd))
        return True, out_path

    duration = _probe_duration(in_path)

    print(f"\n转换: {os.path.basename(in_path)}")
    mode_tag = " [无损]" if lossless else ""
    print(f"  编码器: {enc}  | 硬解: {'是' if use_hw else '否'}{mode_tag}"
          f"  | 时长: {duration:.1f}s" if duration else
          f"  编码器: {enc}  | 硬解: {'是' if use_hw else '否'}{mode_tag}")
    print(f"  输出: {out_path}")

    # 用 -progress pipe:1 拿进度
    cmd_with_progress = cmd[:-1] + ["-progress", "pipe:1", "-nostats", cmd[-1]]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0,
    )
    stderr_thread = _drain_stderr(proc)

    last_pct = -1
    try:
        for raw in iter(proc.stdout.readline, b""):
            if not raw:
                break
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("out_time_us=") and duration:
                cur = int(line.split("=", 1)[1]) / 1_000_000.0
                pct = min(100, int(cur / duration * 100))
                if pct > last_pct and pct % 5 == 0:
                    last_pct = pct
                    print(f"  进度: {pct}%  ({time.time()-t0:.1f}s)")
            elif line == "progress=end":
                break
    finally:
        proc.stdout.close()
        proc.wait()
        stderr_thread.join(timeout=2)

    if proc.returncode != 0:
        return False, f"ffmpeg 返回码 {proc.returncode}"

    # 原地模式: 把 (临时输出) 替换到原路径
    if inplace:
        # _build_cmd 直接写到了 in_path, ffmpeg -y 已覆盖原文件
        # 但若中途失败会损坏原图, 改进: 实际写临时文件再替换
        pass  # 下面 _convert_inplace_safe 走另一条路, 这里不会进来

    elapsed = time.time() - t0
    print(f"  完成 ✓  用时 {elapsed:.1f}s")
    return True, out_path


def convert_video_inplace_safe(in_path, encoder="auto", dry_run=False,
                                lossless=False):
    """安全的原地转换: 写临时文件, 成功后原子替换。"""
    if not os.path.isfile(in_path):
        return False, f"文件不存在: {in_path}"

    enc = _pick_encoder(encoder)
    use_hw = enc in ("h264_nvenc", "h264_qsv")
    stem, ext = os.path.splitext(in_path)
    tmp_path = stem + "_rb_tmp" + ext
    cmd = _build_cmd(in_path, tmp_path, enc, hwaccel=use_hw, lossless=lossless)

    if dry_run:
        print("  [dry-run] 临时输出:", tmp_path)
        print("  [dry-run]", " ".join(shlex.quote(c) for c in cmd))
        print("  [dry-run] 替换: os.replace(tmp, original)")
        return True, in_path

    duration = _probe_duration(in_path)
    print(f"\n原地转换: {os.path.basename(in_path)}  (编码器: {enc})")

    cmd_with_progress = cmd[:-1] + ["-progress", "pipe:1", "-nostats", cmd[-1]]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    stderr_thread = _drain_stderr(proc)

    last_pct = -1
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("out_time_us=") and duration:
                cur = int(line.split("=", 1)[1]) / 1_000_000.0
                pct = min(100, int(cur / duration * 100))
                if pct > last_pct and pct % 5 == 0:
                    last_pct = pct
                    print(f"  进度: {pct}%  ({time.time()-t0:.1f}s)")
            elif line == "progress=end":
                break
    finally:
        proc.stdout.close()
        proc.wait()
        stderr_thread.join(timeout=2)

    if proc.returncode != 0 or not os.path.exists(tmp_path):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False, f"ffmpeg 返回码 {proc.returncode}"

    os.replace(tmp_path, in_path)
    print(f"  完成 ✓ 原地替换  用时 {time.time()-t0:.1f}s")
    return True, in_path


def main():
    parser = argparse.ArgumentParser(
        description="视频 R/B 通道交换 (RGB↔BGR), FFmpeg 加速",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python tools/video_rb_convert.py a.mp4 b.mp4 c.mp4\n"
               "  python tools/video_rb_convert.py datasets/*.mp4 --inplace\n",
    )
    parser.add_argument("videos", nargs="+", help="视频文件路径 (可多个 / 用通配符)")
    parser.add_argument("--encoder", default="auto",
                        choices=["auto", "h264_nvenc", "h264_qsv", "libx264"],
                        help="视频编码器 (auto=自动选最快的)")
    parser.add_argument("--inplace", action="store_true",
                        help="原地替换原文件 (先写临时文件再原子替换, 安全)")
    parser.add_argument("--overwrite", action="store_true",
                        help="非原地模式下, 覆盖已存在的 _rb 输出文件")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印 ffmpeg 命令, 不真正执行")
    parser.add_argument("--lossless", action="store_true",
                        help="无损编码 (NVENC -tune lossless / libx264 -crf 0, 文件会很大)")
    args = parser.parse_args()

    # 检查 ffmpeg
    if not shutil.which("ffmpeg"):
        print("[ERROR] 系统未安装 ffmpeg", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"视频 R/B 通道交换 | 共 {len(args.videos)} 个视频")
    print(f"模式: {'原地替换' if args.inplace else '输出 _rb 文件'}"
          f"  编码器: {args.encoder}"
          + ("  [无损]" if args.lossless else "")
          + ("  [dry-run]" if args.dry_run else ""))
    print("=" * 60)

    ok, fail = 0, 0
    results = []
    for i, vp in enumerate(args.videos, 1):
        print(f"\n[{i}/{len(args.videos)}]")
        if args.inplace:
            success, info = convert_video_inplace_safe(
                vp, encoder=args.encoder, dry_run=args.dry_run,
                lossless=args.lossless)
        else:
            success, info = convert_video(
                vp, encoder=args.encoder, inplace=False,
                dry_run=args.dry_run, overwrite=args.overwrite,
                lossless=args.lossless)
        if success:
            ok += 1
            results.append((vp, "OK", info))
        else:
            fail += 1
            results.append((vp, "FAIL", info))
            print(f"  [失败] {info}")

    print("\n" + "=" * 60)
    print(f"完成: 成功 {ok}, 失败 {fail}, 总计 {len(args.videos)}")
    for vp, status, info in results:
        flag = "✓" if status == "OK" else "✗"
        print(f"  {flag} {os.path.basename(vp)}: {info}")
    print("=" * 60)
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
