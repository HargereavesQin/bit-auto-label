"""Exercise SAM pose creation through the real view and main-window slots."""
import gc
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QPointF, QSettings, Qt, QTimer

from core.canvas import CanvasMode
from core.pose_shapes import PoseShape


@pytest.fixture
def pose_window(qtbot, monkeypatch, tmp_path, synthetic_image):
    import main
    from core.sam_client import SAMClient
    from core.pose_config import PoseConfig

    original_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    monkeypatch.setattr(SAMClient, "load_model_async", lambda *args: None)
    monkeypatch.setattr(main.MainWindow, "_init_lora_sam", lambda self: None)
    monkeypatch.setattr(PoseConfig, "save", lambda *args: True)
    window = main.MainWindow()
    qtbot.addWidget(window)
    image = synthetic_image(640, 480)
    window.current_image_path = str(image)
    window.current_dir = str(tmp_path)
    window.scene.load_image(str(image))
    window.class_list = ["armor_screen"]
    window.pose_config.categories = {
        "armor_screen": {"kpt_count": 4, "keypoints": ["TL", "TR", "BR", "BL"]},
    }
    window._set_mode(CanvasMode.POSE)
    window.pose_panel.set_kpt_only(False)
    window.pose_panel.rm_auto_btn.setChecked(False)
    window.samSwitch.setChecked(True)
    window.scene.set_sam_enabled(True)
    window.show()
    window.view.fitInView(window.scene.sceneRect(), Qt.KeepAspectRatio)

    def infer(x, y, is_click):
        QTimer.singleShot(0, lambda: window.scene.handle_sam_result(
            [[100, 100], [400, 100], [400, 350], [100, 350]],
            [100, 100, 300, 250], [], 0.9, is_click,
        ))

    monkeypatch.setattr(window.sam_client, "request_inference", infer)
    yield window
    window.close()
    QSettings.setDefaultFormat(original_format)


def poses(window):
    return [item for item in window.scene.items() if isinstance(item, PoseShape)]


def click_view(qtbot, window, point, button=Qt.LeftButton):
    pos = window.view.mapFromScene(QPointF(*point))
    qtbot.mouseMove(window.view.viewport(), pos)
    qtbot.mouseClick(window.view.viewport(), button, pos=pos)
    qtbot.wait(20)
    gc.collect()


def test_confirm_sam_then_click_inside_and_edges(qtbot, pose_window):
    window = pose_window
    click_view(qtbot, window, (250, 200))
    assert len(poses(window)) == 1
    assert len(poses(window)[0].keypoint_items) == 0
    for index, point in enumerate([(110, 110), (400, 100), (400, 350), (100, 350)]):
        click_view(qtbot, window, point)
        assert len(poses(window)) == 1
        assert len(poses(window)[0].keypoint_items) == index + 1
        assert poses(window)[0].rect().getRect() == (100, 100, 300, 250)
    with open(window.current_image_path.rsplit('.', 1)[0] + '.json', encoding='utf-8') as stream:
        saved = json.load(stream)
    assert len(saved['shapes']) == 1
    assert len(saved['shapes'][0]['keypoints']) == 4


def test_right_click_keypoint_changes_visibility_without_deleting_pose(qtbot, pose_window):
    window = pose_window
    click_view(qtbot, window, (250, 200))
    click_view(qtbot, window, (110, 110))
    click_view(qtbot, window, (110, 110), Qt.RightButton)
    assert len(poses(window)) == 1
    assert poses(window)[0].keypoint_items[0].visibility == 0


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("points", [
    [(200, 200), (204, 200), (204, 204), (200, 204)],
    [(200, 200), (212, 198), (220, 220), (200, 220)],
], ids=["overlapping-markers", "number-label-overlap"])
def test_nearby_corners_fill_next_slots_without_moving_previous_points(qtbot, pose_window, scale, points):
    window = pose_window
    click_view(qtbot, window, (250, 200))
    window.view.resetTransform()
    window.view.scale(scale, scale)
    window.view.centerOn(210, 210)
    expected = []
    for index, point in enumerate(points):
        # Mouse coordinates are integer viewport pixels; compare with the
        # actual mapped click location, including at fractional zoom.
        pixel = window.view.mapFromScene(QPointF(*point))
        position = window.view.mapToScene(pixel)
        expected.append([position.x(), position.y(), 2])
        click_view(qtbot, window, point)
        assert len(poses(window)) == 1
        shape = poses(window)[0]
        assert len(shape.keypoint_items) == index + 1
        for actual, wanted in zip(shape.get_keypoints_data(), expected):
            assert actual == pytest.approx(wanted)
        assert shape.rect().getRect() == (100, 100, 300, 250)


def test_completed_pose_still_allows_dragging_keypoints(qtbot, pose_window):
    window = pose_window
    click_view(qtbot, window, (250, 200))
    for point in [(110, 110), (390, 110), (390, 340), (110, 340)]:
        click_view(qtbot, window, point)
    start = window.view.mapFromScene(QPointF(110, 110))
    end = window.view.mapFromScene(QPointF(130, 130))
    qtbot.mousePress(window.view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(window.view.viewport(), end)
    qtbot.mouseRelease(window.view.viewport(), Qt.LeftButton, pos=end)
    shape = poses(window)[0]
    assert len(shape.keypoint_items) == 4
    assert shape.get_keypoints_data()[0][:2] == pytest.approx([130, 130], abs=2)
    assert shape.rect().getRect() == (100, 100, 300, 250)


def test_autosave_reports_path_and_each_added_point(qtbot, pose_window):
    window = pose_window
    click_view(qtbot, window, (250, 200))
    target = Path(window.current_image_path).with_suffix(".json")
    assert window.saveStatusLabel.toolTip() == str(target)
    for count, point in enumerate([(110, 110), (115, 110)], 1):
        click_view(qtbot, window, point)
        saved = json.loads(target.read_text(encoding="utf-8"))
        visible = [p for p in saved["shapes"][0]["keypoints"] if p[2] == 2]
        assert len(visible) == count
        assert "已保存" in window.saveStatusLabel.text()


def test_failed_save_blocks_switch_and_close_then_ctrl_s_recovers(
        qtbot, pose_window, monkeypatch, synthetic_image):
    import main
    from PySide6.QtGui import QCloseEvent
    from core.exporter import Exporter

    window = pose_window
    click_view(qtbot, window, (250, 200))
    target = Path(window.current_image_path).with_suffix(".json")
    original = target.read_bytes()
    saved_counts = dict(window._image_label_counts)
    original_path = window.current_image_path
    window.listFiles.blockSignals(True)
    window.listFiles.addItems([original_path, str(synthetic_image())])
    window.listFiles.setCurrentRow(0)
    window.listFiles.blockSignals(False)
    notices = []
    monkeypatch.setattr(main, "DialogOver", lambda *args: notices.append(args))
    save_json = Exporter.save_json

    def fail(*args, **kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr(Exporter, "save_json", fail)
    click_view(qtbot, window, (110, 110))
    assert window._annotation_save_failed
    assert target.read_bytes() == original
    assert window._image_label_counts == saved_counts
    assert "未保存" in window.saveStatusLabel.text()
    window.listFiles.setCurrentRow(1)
    assert window.listFiles.currentRow() == 0
    assert window.current_image_path == original_path
    assert len(poses(window)[0].keypoint_items) == 1
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    qtbot.keyClick(window, Qt.Key_S, Qt.ControlModifier)
    assert not any(args[-1] == "success" for args in notices)

    monkeypatch.setattr(Exporter, "save_json", save_json)
    qtbot.keyClick(window, Qt.Key_S, Qt.ControlModifier)
    assert not window._annotation_save_failed
    assert any(args[-1] == "success" for args in notices)
    assert target.read_bytes() != original


def test_close_flushes_changes_without_creating_untouched_negative(pose_window):
    from PySide6.QtCore import QRectF

    window = pose_window
    target = Path(window.current_image_path).with_suffix(".json")
    window._saved_annotation_state = window._annotation_state()
    assert window._flush_annotation_changes()
    assert not target.exists()
    # Simulate a scene modification whose change notification was missed.
    shape = PoseShape(QRectF(100, 100, 300, 250), "armor_screen", ["TL", "TR", "BR", "BL"])
    window.scene.addItem(shape)
    assert window.close()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["shapes"][0]["label"] == "armor_screen"


def test_switch_directory_failure_preserves_current_dataset(qtbot, pose_window, monkeypatch, tmp_path):
    import main
    from core.exporter import Exporter

    window = pose_window
    click_view(qtbot, window, (250, 200))
    old_dir, old_path = window.current_dir, window.current_image_path
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(main.QFileDialog, "getExistingDirectory", lambda *args: str(other))
    monkeypatch.setattr(main, "DialogOver", lambda *args: None)
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("disk full")
        patch.setattr(Exporter, "save_json", fail)
        click_view(qtbot, window, (110, 110))
        window.open_dir()
        assert (window.current_dir, window.current_image_path) == (old_dir, old_path)
        assert window._annotation_save_failed
    assert window.auto_save_annotation()


def test_moved_image_and_json_keep_pose_labels(qtbot, pose_window, tmp_path):
    from core.dataset_layout import resolve_open_layout
    from core.exporter import Exporter

    window = pose_window
    click_view(qtbot, window, (250, 200))
    click_view(qtbot, window, (110, 110))
    expected = Exporter.extract_shapes(window.scene)
    source = Path(window.current_image_path)
    destination = tmp_path / "moved"
    destination.mkdir()
    # Simulate reopening the same image from its new dataset location.
    window.current_image_path = None
    moved = source.rename(destination / source.name)
    source.with_suffix(".json").rename(moved.with_suffix(".json"))
    layout = resolve_open_layout(str(destination))
    assert layout.annotation_map[str(moved)] == str(moved.with_suffix(".json"))
    window.current_image_path = str(moved)
    window.current_dir = str(destination)
    window.scene.load_image(str(moved))
    window.load_annotations(str(moved))
    assert Exporter.extract_shapes(window.scene) == expected


def test_hidden_panels_can_be_restored_from_view_menu(qtbot, pose_window):
    window = pose_window
    # 使用与 Qt 右键菜单相同的 action 隐藏，顶栏菜单仍可恢复。
    window.toolBar.toggleViewAction().trigger()
    assert window.toolBar.isHidden()
    assert not window.actionShowToolbar.isChecked()
    window.actionShowToolbar.trigger()
    assert window.toolBar.isVisible()
    assert window.actionShowToolbar.isChecked()
    window.toolBar.hide()
    window.dockRight.hide()
    window.actionShowPanels.trigger()
    assert window.toolBar.isVisible()
    assert window.dockRight.isVisible()


def test_export_menu_produces_yolo_pose_with_original_keypoint_order(
        qtbot, pose_window, monkeypatch, tmp_path):
    import yaml
    from tools.main_dataset_tool import DatasetToolWindow, MODES

    # 测试窗口不能写入用户日常使用的数据集工具配置。
    monkeypatch.setattr(DatasetToolWindow, "_save_config", lambda self: None)
    monkeypatch.setattr(DatasetToolWindow, "trigger_message", lambda *args: None)
    window = pose_window
    click_view(qtbot, window, (250, 200))
    for point in [(110, 110), (390, 110), (390, 340), (110, 340)]:
        click_view(qtbot, window, point)
    source = Path(window.current_image_path)
    original = source.with_suffix(".json").read_bytes()
    window.exportActions["TO_YOLO_POSE"].trigger()
    exporter = window.dataset_window
    qtbot.addWidget(exporter)
    assert window.exportButton.menu() is window.menuExport
    assert MODES[exporter.mode_combo.currentIndex()][0] == "TO_YOLO_POSE"
    assert Path(exporter.img_input.text()) == source.parent
    assert Path(exporter.ann_input.text()) == source.parent
    assert not exporter.force_rm14_chk.isChecked()
    assert not exporter.out_input.text()
    destination = tmp_path / "export"
    exporter.out_input.setText(str(destination))
    exporter.ratios = (1.0, 0.0, 0.0)
    exporter.start_btn.click()
    qtbot.waitUntil(lambda: exporter.start_btn.isEnabled(), timeout=10000)
    assert exporter.worker.wait(1000)
    assert exporter.status_lbl.text() == "完成"
    row = (destination / "labels/train" / source.with_suffix(".txt").name).read_text().split()
    assert len(row) == 17
    assert row[0] == "0"
    for i, point in enumerate(poses(window)[0].get_keypoints_data()):
        assert list(map(float, row[5 + 3*i:8 + 3*i])) == pytest.approx(
            [point[0] / 640, point[1] / 480, point[2]], abs=1e-6)
    assert (destination / "classes.txt").read_text().splitlines() == ["armor_screen"]
    config = json.loads((destination / "pose_classes.json").read_text(encoding="utf-8"))
    assert config["categories"]["armor_screen"]["keypoints"] == ["TL", "TR", "BR", "BL"]
    dataset_yaml = next(destination.glob("*.yaml"))
    assert yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))["kpt_shape"] == [4, 3]
    assert source.with_suffix(".json").read_bytes() == original


def test_pose_prompt_visible_creates_pose_boxes_and_accepts_points(qtbot, pose_window, monkeypatch):
    window = pose_window
    window._set_mode(CanvasMode.RECT)
    window._set_mode(CanvasMode.POSE)
    window._on_pose_auto_mode_changed("armor")
    assert window.samTextGroup.isVisible()
    assert window.samPromptInput.isVisible() and window.samPromptInput.isEnabled()
    assert window.dockLayout.indexOf(window.pose_panel) < window.dockLayout.indexOf(window.samTextGroup)
    # 模拟模型输出，验证实际提示词信号、位姿实例和鼠标补点链路。
    monkeypatch.setattr(window.sam_client, "model", object())
    requested = []
    monkeypatch.setattr(window.sam_client, "request_text_inference", requested.append)
    window.samPromptInput.setText("number")
    qtbot.keyClick(window.samPromptInput, Qt.Key_Return)
    assert requested == ["number"]
    results = [{"rect": [100, 100, 300, 250]}, {"rect": [420, 100, 180, 250]}]
    window.sam_client.text_result_ready.emit(results, "number")
    assert len(poses(window)) == 2
    assert "number" not in window.class_list
    assert all(shape.label == "armor_screen" and shape.kpt_count == 4 for shape in poses(window))
    first = window.scene._pose_keypoint_target()
    assert first.rect().getRect() == (100, 100, 300, 250)
    for point in [(110, 110), (390, 110), (390, 340), (110, 340)]:
        click_view(qtbot, window, point)
    assert first.is_kpt_complete()
    assert first.rect().getRect() == (100, 100, 300, 250)
    second = next(shape for shape in poses(window) if shape is not first)
    window.on_pose_instance_selected(second)
    click_view(qtbot, window, (430, 110))
    assert len(second.keypoint_items) == 1
    saved = json.loads(Path(window.current_image_path).with_suffix(".json").read_text(encoding="utf-8"))
    assert all(shape["shape_type"] == "pose" for shape in saved["shapes"])
    assert all(shape["kpt_names"] == ["TL", "TR", "BR", "BL"] for shape in saved["shapes"])
    # 图片特征已切换时，迟到的提示词结果不能增加实例。
    window.samPromptBtn.click()
    window._sam_load_token += 1
    window.sam_client.text_result_ready.emit(results, "number")
    assert len(poses(window)) == 2


def test_delete_current_image_cancel_confirm_and_last_image(
        qtbot, pose_window, monkeypatch, synthetic_image, tmp_path):
    import main

    window = pose_window
    first = Path(window.current_image_path)
    second = synthetic_image()
    window._reload_after_fs_change(preferred_image=str(first))
    click_view(qtbot, window, (250, 200))
    annotation = first.with_suffix(".json")
    saved = annotation.read_bytes()
    monkeypatch.setattr(main.QMessageBox, "question", lambda *args: main.QMessageBox.No)
    window.actionDeleteImage.trigger()
    assert first.exists() and annotation.read_bytes() == saved
    assert window.current_image_path == str(first)
    monkeypatch.setattr(main.QMessageBox, "question", lambda *args: main.QMessageBox.Yes)
    window.actionDeleteImage.trigger()
    assert not first.exists() and not annotation.exists()
    assert window.current_image_path == str(second)
    assert window.listFiles.count() == 1
    buckets = list((tmp_path / ".deleted_images").iterdir())
    assert len(buckets) == 1
    assert (buckets[0] / "files" / annotation.name).read_bytes() == saved
    window.actionDeleteImage.trigger()
    assert not second.exists()
    assert window.current_image_path is None
    assert window.scene.img_item is None
    assert window.listFiles.count() == 0
    assert not window.actionDeleteImage.isEnabled()
    assert window._flush_annotation_changes()
    assert not annotation.exists()
