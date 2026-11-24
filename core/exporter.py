"""
core/exporter.py — 标注 I/O.

从 QGraphicsScene / dict 写出 JSON / YOLO / VOC XML,
从 JSON 读入 dict.

新版本走 core.shape_registry: 不再写 shape_type 字符串分发.
加新 shape 只需要在 registry 里 register, 这里不动一行.
"""
import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from .shape_registry import spec_for_item, get as spec_get
from .constants import SHAPE_POSE, SHAPE_HIERARCHY
from .atomic_write import atomic_text_writer

# Pose 与 Hierarchy 共用 keypoints/kpt_names 字段，统一判断
_POSE_LIKE_TYPES = (SHAPE_POSE, SHAPE_HIERARCHY)


class Exporter:
    @staticmethod
    def extract_shapes(scene):
        """把 scene 里所有 QGraphicsItem 转成 dict 列表. 通过 registry 派发."""
        shapes_data = []
        for item in scene.items():
            if getattr(item, 'is_temp', False):
                continue
            spec = spec_for_item(item)
            if spec is not None:
                shapes_data.append(spec.to_dict(item))
        shapes_data.reverse()
        return shapes_data

    @staticmethod
    def save_json(filepath, image_path, image_width, image_height, shapes, mlp_label=None):
        data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageHeight": int(image_height),
            "imageWidth": int(image_width)
        }
        if mlp_label is not None:
            data["mlp_label"] = mlp_label
        for s in shapes:
            shape_type = s["type"]
            shape_dict = {
                "label": s["label"],
                "points": s["points"],
                "group_id": None,
                "shape_type": shape_type,
                "flags": {}
            }
            # OBB 额外字段
            if shape_type == "obb":
                shape_dict["angle"] = s.get("angle", 0)
                shape_dict["rect"] = s.get("rect", [0, 0, 0, 0])
            # Pose 额外字段
            elif shape_type in _POSE_LIKE_TYPES:
                shape_dict["keypoints"] = s.get("keypoints", [])
                shape_dict["kpt_names"] = s.get("kpt_names", [])
                shape_dict["bbox_from_keypoints"] = s.get("bbox_from_keypoints", False)

            data["shapes"].append(shape_dict)

        with atomic_text_writer(filepath) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_mlp_label(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("mlp_label")
        except Exception:
            return None

    @staticmethod
    def save_yolo(filepath, image_width, image_height, shapes, classes_list):
        """所有 shape 走 registry 的 to_yolo_line 派发. 不再有 if/elif 链."""
        lines = []
        for s in shapes:
            if s["label"] not in classes_list:
                continue
            class_id = classes_list.index(s["label"])
            spec = spec_get(s["type"])
            if spec is None or spec.to_yolo_line is None:
                continue
            line = spec.to_yolo_line(s, image_width, image_height, class_id)
            if line:
                lines.append(line)

        with atomic_text_writer(filepath) as f:
            f.write("\n".join(lines))

    @staticmethod
    def save_xml(filepath, image_path, image_width, image_height, shapes):
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = os.path.basename(image_path)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(int(image_width))
        ET.SubElement(size, "height").text = str(int(image_height))
        ET.SubElement(size, "depth").text = "3"

        for s in shapes:
            min_x = min(p[0] for p in s["points"])
            max_x = max(p[0] for p in s["points"])
            min_y = min(p[1] for p in s["points"])
            max_y = max(p[1] for p in s["points"])

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = s["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(min_x))
            ET.SubElement(bndbox, "ymin").text = str(int(min_y))
            ET.SubElement(bndbox, "xmax").text = str(int(max_x))
            ET.SubElement(bndbox, "ymax").text = str(int(max_y))

            # Pose 关键点
            if s.get("type") in _POSE_LIKE_TYPES and s.get("keypoints"):
                kpts_node = ET.SubElement(obj, "keypoints")
                kpt_names = s.get("kpt_names", [])
                for i, kpt in enumerate(s["keypoints"]):
                    kpt_node_elem = ET.SubElement(kpts_node, "keypoint")
                    ET.SubElement(kpt_node_elem, "name").text = kpt_names[i] if i < len(kpt_names) else f"kpt_{i}"
                    ET.SubElement(kpt_node_elem, "x").text = str(int(kpt[0]))
                    ET.SubElement(kpt_node_elem, "y").text = str(int(kpt[1]))
                    ET.SubElement(kpt_node_elem, "visibility").text = str(int(kpt[2]) if len(kpt) > 2 else 2)

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with atomic_text_writer(filepath) as f:
            f.write(xmlstr)
