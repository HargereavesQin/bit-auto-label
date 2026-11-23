"""Exercise SAM pose creation through the real view and main-window slots."""
import gc
import json

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
