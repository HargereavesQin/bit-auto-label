"""按原目录结构收存单张图片与标签，失败时恢复已移动文件。"""
import json
import os
from pathlib import Path
from uuid import uuid4


def trash_image(image_path, annotation_paths, dataset_root):
    root = Path(dataset_root).resolve()
    image = Path(image_path).resolve()
    if not image.is_file():
        raise FileNotFoundError(str(image))
    # 图片最后移动，标签失败时当前图片仍在原处。
    files = []
    for value in annotation_paths:
        if value:
            path = Path(value).resolve()
            if path.is_file() and path != image and path not in files:
                files.append(path)
    files.append(image)
    relative_paths = [path.relative_to(root) for path in files]
    bucket = root / ".deleted_images" / uuid4().hex
    bucket.mkdir(parents=True)
    manifest = {"dataset_root": str(root), "image": str(image.relative_to(root)),
                "files": [str(path) for path in relative_paths]}
    (bucket / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    moved = []
    try:
        for source, relative in zip(files, relative_paths):
            destination = bucket / "files" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, destination)
            moved.append((source, destination))
    except OSError as error:
        failures = []
        for source, destination in reversed(moved):
            try:
                if source.exists():
                    raise FileExistsError(str(source))
                os.rename(destination, source)
            except OSError as restore_error:
                failures.append(str(restore_error))
        if failures:
            raise OSError(f"移动失败：{error}；部分文件保存在 {bucket}，恢复失败：{'; '.join(failures)}") from error
        raise
    return bucket
