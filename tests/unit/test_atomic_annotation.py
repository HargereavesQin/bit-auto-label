"""Interrupted annotation writes must leave the previous labels intact."""
import json

import pytest

from core.atomic_write import atomic_text_writer
from core.exporter import Exporter


@pytest.mark.parametrize("failure", ["serialization", "replace"])
def test_failed_write_preserves_original_and_removes_temporary_file(tmp_path, monkeypatch, failure):
    import core.atomic_write as module

    target = tmp_path / "image.json"
    target.write_text('{"shapes": []}', encoding="utf-8")
    original = target.read_bytes()
    if failure == "replace":
        def deny_replace(*args):
            raise PermissionError("read only")
        monkeypatch.setattr(module.os, "replace", deny_replace)
    with pytest.raises((ValueError, PermissionError)):
        with atomic_text_writer(target) as stream:
            stream.write('{"shapes": [')
            if failure == "serialization":
                raise ValueError("serialization interrupted")
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("file_format", ["json", "yolo", "xml"])
def test_atomic_export_replaces_existing_file(tmp_path, file_format):
    target = tmp_path / ("image." + file_format)
    target.write_text("old content", encoding="utf-8")
    if file_format == "json":
        Exporter.save_json(target, "image.jpg", 640, 480, [])
        assert json.loads(target.read_text(encoding="utf-8"))["shapes"] == []
    elif file_format == "yolo":
        Exporter.save_yolo(target, 640, 480, [], [])
        assert target.read_text(encoding="utf-8") == ""
    else:
        Exporter.save_xml(target, "image.jpg", 640, 480, [])
        assert "<filename>image.jpg</filename>" in target.read_text(encoding="utf-8")
    assert list(tmp_path.iterdir()) == [target]
