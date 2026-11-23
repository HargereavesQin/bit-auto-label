"""Partially annotated poses must remain editable after save/load or undo."""
import json
import gc
import weakref
import pytest
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsSceneMouseEvent

from core.canvas import Canvas, CanvasMode
from core.pose_shapes import PoseShape
from core.shape_registry import get
from core.exporter import Exporter


NAMES = ["TL", "TR", "BR", "BL"]
POINTS = [(30, 30), (100, 30), (100, 100), (30, 100)]


@pytest.mark.parametrize("kind", ["pose", "hierarchy_pose"])
@pytest.mark.parametrize("placed", [0, 1, 3, 4])
def test_partial_pose_can_be_completed_after_round_trip(qapp, kind, placed):
    shape = PoseShape(QRectF(20, 20, 100, 100), "armor_screen", NAMES)
    for point in POINTS[:placed]:
        shape.add_keypoint(QPointF(*point))
    spec = get(kind)
    restored = spec.from_dict(spec.to_dict(shape))
    assert restored.kpt_count == 4
    assert restored.get_kpt_names() == NAMES
    assert restored.is_kpt_complete() == (placed == 4)
    for point in POINTS[placed:]:
        assert restored.add_keypoint(QPointF(*point))
    assert restored.is_kpt_complete()
    assert restored.get_keypoints_data() == [[x, y, 2] for x, y in POINTS]


def test_missing_slots_do_not_reorder_existing_keypoints(qapp):
    restored = get("pose").from_dict({
        "type": "pose", "label": "armor_screen",
        "points": [[20, 20], [120, 120]], "kpt_names": NAMES,
        "keypoints": [[30, 30, 2], [0, 0, 0], [100, 100, 2], [0, 0, 0]],
    })
    assert restored.kpt_count == 4
    assert restored.keypoint_items[2].kpt_name == "BR"
    assert restored.get_keypoints_data()[2] == [100, 100, 2]
    assert not restored.is_kpt_complete()
    assert restored.add_keypoint(QPointF(30, 100))
    assert restored.keypoint_items[3].kpt_name == "BL"


def test_sam_box_can_receive_manual_points_after_restore(qapp):
    canvas = Canvas()
    canvas.setSceneRect(0, 0, 200, 200)
    canvas.mode = CanvasMode.POSE
    canvas.set_sam_enabled(True)
    created = []

    def accept_shape(shape):
        shape.label = "armor_screen"
        shape.kpt_names = list(NAMES)
        shape.kpt_count = 4
        canvas.addItem(shape)
        created.append(shape)

    canvas.shape_drawn.connect(accept_shape)
    canvas.handle_sam_result([[20, 20], [120, 20], [120, 120]],
                            [20, 20, 100, 100], [], 0.9, True)
    shape = created[0]
    spec = get("pose")
    restored = spec.from_dict(spec.to_dict(shape))
    canvas.removeItem(shape)
    canvas.addItem(restored)
    for point in POINTS:
        assert canvas._add_pose_keypoint(restored, QPointF(*point))
    assert restored.is_kpt_complete()


def click(canvas, point):
    event = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
    event.setButton(Qt.LeftButton)
    event.setButtons(Qt.LeftButton)
    event.setScenePos(QPointF(*point))
    canvas.mousePressEvent(event)


def make_sam_canvas():
    canvas = Canvas()
    canvas.setSceneRect(0, 0, 200, 200)
    canvas.img_item = QGraphicsPixmapItem(QPixmap(200, 200))
    canvas.addItem(canvas.img_item)
    canvas.mode = CanvasMode.POSE
    canvas.set_sam_enabled(True)

    class SamStub:
        requests = []

        def request_inference(self, x, y, is_click):
            self.requests.append((x, y, is_click))

    canvas.sam_client = SamStub()

    def accept_shape(shape):
        shape.label = "armor_screen"
        shape.kpt_names = list(NAMES)
        shape.kpt_count = 4
        canvas.addItem(shape)

    canvas.shape_drawn.connect(accept_shape)
    canvas.handle_sam_result([[20, 20], [120, 20], [120, 120]],
                            [20, 20, 100, 100], [], 0.9, True)
    return canvas


@pytest.mark.parametrize("kpt_only", [False, True])
def test_sam_box_survives_clicks_moves_and_json_reload(qapp, tmp_path, kpt_only):
    canvas = make_sam_canvas()
    shape = next(item for item in canvas.items() if isinstance(item, PoseShape))
    original_rect = QRectF(shape.rect())
    canvas.pose_kpt_only = kpt_only
    shape.set_bbox_visible(not kpt_only)
    # The first corner is slightly outside the SAM box. It must still belong
    # to this instance rather than trigger another SAM segmentation.
    points = [(18, 18), (110, 25), (110, 110), (25, 110)]
    for index, point in enumerate(points):
        click(canvas, point)
        saved = Exporter.extract_shapes(canvas)
        assert len(saved) == 1
        assert len(shape.keypoint_items) == index + 1
        assert shape.rect() == original_rect
        assert saved[0]["points"] == [[20, 20], [120, 120]]
    assert canvas.sam_client.requests == []
    assert [p.label_item.toPlainText() for p in shape.keypoint_items] == ["1", "2", "3", "4"]
    shape.keypoint_items[0].setPos(15, 15)
    assert shape.rect() == original_rect
    path = tmp_path / "annotation.json"
    Exporter.save_json(str(path), "image.png", 200, 200,
                       Exporter.extract_shapes(canvas))
    restored = get("pose").from_dict(json.loads(path.read_text())["shapes"][0])
    assert restored.rect() == original_rect
    assert not restored.bbox_from_keypoints
    assert restored.get_keypoints_data()[0] == [15, 15, 2]


def test_point_only_new_instance_keeps_automatic_bbox_after_reload(qapp, tmp_path):
    canvas = make_sam_canvas()
    canvas.clear_shapes()
    canvas.pose_kpt_only = True
    for point in POINTS:
        click(canvas, point)
    saved = Exporter.extract_shapes(canvas)
    assert len(saved) == 1
    assert saved[0]["points"] == [[20, 20], [110, 110]]
    path = tmp_path / "points.json"
    Exporter.save_json(str(path), "image.png", 200, 200, saved)
    restored = get("pose").from_dict(json.loads(path.read_text())["shapes"][0])
    assert restored.bbox_from_keypoints
    canvas.clear_shapes()
    canvas.addItem(restored)
    restored.keypoint_items[1].setPos(130, 30)
    assert restored.rect().right() == 140


def test_older_saved_box_recovers_point_slots_from_category(qapp):
    from types import SimpleNamespace
    from main import MainWindow

    shape = get("pose").from_dict({
        "label": "armor_screen", "points": [[20, 20], [120, 120]],
        "keypoints": [], "kpt_names": [],
    })
    canvas = Canvas()
    window = SimpleNamespace(
        class_list=["armor_screen"], scene=canvas,
        pose_config=SimpleNamespace(get_category=lambda _: {
            "kpt_count": 4, "keypoints": NAMES,
        }),
    )
    MainWindow._add_shape_to_scene(window, shape, "armor_screen")
    assert not shape.is_kpt_complete()
    assert shape.kpt_count == 4
    assert shape.rect() == QRectF(20, 20, 100, 100)
    assert shape.add_keypoint(QPointF(30, 30))


def test_late_sam_result_does_not_create_duplicate_during_point_entry(qapp):
    canvas = make_sam_canvas()
    canvas.handle_sam_result([[50, 50], [150, 50], [150, 150]],
                            [50, 50, 100, 100], [], 0.9, True)
    assert len(Exporter.extract_shapes(canvas)) == 1
    assert canvas.sam_hover_item is None


@pytest.mark.parametrize("operation", ["remove", "switch_image", "clear"])
def test_canvas_releases_annotation_ownership_after_removal(qapp, operation):
    canvas = make_sam_canvas()
    shape = next(item for item in canvas.items() if isinstance(item, PoseShape))
    reference = weakref.ref(shape)
    if operation == "remove":
        canvas.removeItem(shape)
    elif operation == "switch_image":
        canvas.clear_shapes()
    else:
        canvas.clear()
    del shape
    gc.collect()
    assert reference() is None
    assert not any(isinstance(item, PoseShape) for item in canvas.items())
