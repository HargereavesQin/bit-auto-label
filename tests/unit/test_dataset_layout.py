# -*- coding: utf-8 -*-
"""C1/C4 纯逻辑单元测试：sidecar 路径推导与目录布局解析。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.dataset_layout import (  # noqa: E402
    OpenLayoutResult,
    resolve_open_layout,
    sidecar_json_path,
    sidecar_txt_path,
    yolo_label_path_for_image,
)


class TestSidecarPaths:
    def test_json_sidecar_replaces_extension(self):
        assert sidecar_json_path("/data/imgs/a.b.c.png").endswith("a.b.c.json")

    def test_txt_sidecar_replaces_extension(self):
        assert sidecar_txt_path("/data/imgs/photo.jpeg").endswith("photo.txt")

    def test_no_extension_appends(self):
        assert sidecar_json_path("/data/FILE").endswith("FILE.json")


class TestResolveOpenLayout:
    def test_flat_layout_with_sidecars(self, tmp_path, synthetic_image):
        for i in range(3):
            img = synthetic_image(seed=i)
            sidecar = img.with_suffix(".json")
            sidecar.write_text("{}", encoding="utf-8")
        result = resolve_open_layout(str(tmp_path))
        assert isinstance(result, OpenLayoutResult)
        assert result.image_paths, "平铺目录应识别出图片"
        assert len(result.image_paths) == 3

    def test_yolo_tree_layout(self, tmp_path, synthetic_image):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        images.mkdir()
        labels.mkdir()
        img = synthetic_image()
        (images / img.name).write_bytes(img.read_bytes())
        (labels / (img.stem + ".txt")).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = resolve_open_layout(str(images))
        assert result.image_paths, "标准 YOLO images/ 目录应被识别"

    def test_yolo_label_path_for_image_keeps_subdir(self, tmp_path):
        img = str(tmp_path / "images" / "sub" / "a.png")
        lbl = yolo_label_path_for_image(img, str(tmp_path / "images"), str(tmp_path / "labels"))
        assert Path(lbl).relative_to(tmp_path / "labels").as_posix() == "sub/a.txt"


@pytest.mark.parametrize("path", [
    "D:/x/../y/img.png",
    "/home/u/imgs/pic.jpg",
])
def test_sidecar_no_traversal(path):
    """sidecar 推导只做后缀替换，不得引入目录跳转。"""
    out = Path(sidecar_json_path(path))
    assert out.parent == Path(path).parent
