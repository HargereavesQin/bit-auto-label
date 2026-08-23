"""命令行批处理入口。

示例::

    python -m buff_sam3_auto.cli \\
        --images path/to/images \\
        --sam path/to/sam3.pt \\
        --out path/to/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .paths import default_onnx_path
from .strategy import BuffAutoStrategy

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Buff SAM3 自动标注：YOLO ONNX bbox + 双轮廓极轴五点"
    )
    parser.add_argument("--images", required=True, help="图片目录或单张图片")
    parser.add_argument("--sam", required=True, help="SAM3 权重路径 (.pt)")
    parser.add_argument(
        "--onnx",
        default=str(default_onnx_path()),
        help="YOLO bbox ONNX（默认使用包内 weights/yolo11n_buff.onnx）",
    )
    parser.add_argument("--out", required=True, help="YOLO txt 输出目录")
    parser.add_argument("--bpe", default="", help="可选 BPE 词表路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args(argv)

    src = Path(args.images).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        images = [src]
    else:
        images = sorted(
            p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
    if not images:
        raise SystemExit(f"未找到图片: {src}")

    strategy = BuffAutoStrategy(
        onnx_path=args.onnx,
        sam_checkpoint=args.sam,
        bpe_path=args.bpe or None,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )
    print(f"detector provider: {strategy.detector.provider}")
    print(f"onnx: {args.onnx}")

    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"[{i}/{len(images)}] skip unreadable: {path.name}")
            continue
        annotations, _ = strategy.annotate(image)
        lines = BuffAutoStrategy.to_yolo_lines(annotations)
        label_path = out_dir / f"{path.stem}.txt"
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        print(f"[{i}/{len(images)}] {path.name}: {len(lines)} objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
