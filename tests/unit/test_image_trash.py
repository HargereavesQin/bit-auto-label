"""图片和不同目录中的标签必须一并收存，失败时恢复原文件。"""
import json

import pytest

from core.image_trash import trash_image


@pytest.mark.parametrize("fail_image_move", [False, True])
def test_yolo_image_and_labels_move_together(tmp_path, monkeypatch, fail_image_move):
    import core.image_trash as module

    image = tmp_path / "images/train/frame.jpg"
    label = tmp_path / "labels/train/frame.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image content")
    label.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    sidecar = image.with_suffix(".json")
    sidecar.write_text('{"shapes": []}', encoding="utf-8")
    originals = {path: path.read_bytes() for path in (image, label, sidecar)}
    if fail_image_move:
        rename = module.os.rename
        def deny_image(source, destination):
            if source == image:
                raise PermissionError("image is locked")
            return rename(source, destination)
        monkeypatch.setattr(module.os, "rename", deny_image)
        with pytest.raises(PermissionError):
            trash_image(image, [label, sidecar], tmp_path)
        assert all(path.read_bytes() == data for path, data in originals.items())
    else:
        bucket = trash_image(image, [label, sidecar, label], tmp_path)
        manifest = json.loads((bucket / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["files"]) == 3
        for path, data in originals.items():
            assert not path.exists()
            assert (bucket / "files" / path.relative_to(tmp_path)).read_bytes() == data
