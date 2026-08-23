# -*- coding: utf-8 -*-
"""
统计模块 - 统计标注数据集的类别分布、图片标注情况等
"""
import os
import json


class DatasetStatistics:
    """数据集统计工具"""

    @staticmethod
    def scan_directory(dir_path):
        """
        扫描目录，返回图片文件列表和对应的标注文件映射
        返回: (image_files, annotation_map)
            image_files: 所有图片文件路径列表
            annotation_map: {图片路径: 标注文件路径}，无标注的图片值为 None
        """
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        annotation_extensions = ('.json', '.txt', '.xml')

        image_files = []
        annotation_map = {}

        if not os.path.isdir(dir_path):
            return image_files, annotation_map

        for f in sorted(os.listdir(dir_path)):
            f_lower = f.lower()
            full_path = os.path.join(dir_path, f)

            if f_lower.endswith(image_extensions):
                image_files.append(full_path)
                # 查找对应的标注文件
                base_name = os.path.splitext(full_path)[0]
                found_anno = None
                for ext in annotation_extensions:
                    anno_path = base_name + ext
                    if os.path.exists(anno_path):
                        found_anno = anno_path
                        break
                annotation_map[full_path] = found_anno

        return image_files, annotation_map

    @staticmethod
    def count_annotations_from_json(json_path):
        """从 JSON 标注文件中提取类别信息"""
        labels = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                label = shape.get("label", "").strip()
                if label:
                    labels.append(label)
        except Exception:
            pass
        return labels

    @staticmethod
    def count_annotations_from_yolo(txt_path, classes_list):
        """从 YOLO 标注文件中提取类别信息"""
        labels = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id < len(classes_list):
                            labels.append(classes_list[class_id])
                        else:
                            labels.append(str(class_id))
        except Exception:
            pass
        return labels

    @staticmethod
    def count_annotations_from_xml(xml_path):
        """从 XML 标注文件中提取类别信息"""
        import xml.etree.ElementTree as ET
        labels = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                label = obj.find("name")
                if label is not None and label.text:
                    labels.append(label.text.strip())
        except Exception:
            pass
        return labels

    @staticmethod
    def get_annotation_labels(anno_path, classes_list=None):
        """根据文件扩展名自动选择解析方式"""
        if not anno_path or not os.path.exists(anno_path):
            return []
        ext = os.path.splitext(anno_path)[1].lower()
        if ext == '.json':
            return DatasetStatistics.count_annotations_from_json(anno_path)
        elif ext == '.txt':
            return DatasetStatistics.count_annotations_from_yolo(anno_path, classes_list or [])
        elif ext == '.xml':
            return DatasetStatistics.count_annotations_from_xml(anno_path)
        return []

    @staticmethod
    def _empty_geom():
        return {
            "max_area": 0.0,
            "min_area": 0.0,
            "max_aspect": 0.0,
            "min_aspect": 0.0,
            "has_rect": False,
        }

    @staticmethod
    def _geom_from_wh(widths_heights):
        """从 [(w,h), ...] 汇总面积/长宽比统计。"""
        if not widths_heights:
            return DatasetStatistics._empty_geom()
        areas = []
        aspects = []
        for w, h in widths_heights:
            w = abs(float(w))
            h = abs(float(h))
            if w <= 1e-9 or h <= 1e-9:
                continue
            areas.append(w * h)
            aspects.append(max(w / h, h / w))
        if not areas:
            return DatasetStatistics._empty_geom()
        return {
            "max_area": max(areas),
            "min_area": min(areas),
            "max_aspect": max(aspects),
            "min_aspect": min(aspects),
            "has_rect": True,
        }

    @staticmethod
    def get_detection_geom_stats(anno_path):
        """解析检测框几何：仅 rectangle / YOLO det 行（忽略 pose/poly 等）。"""
        empty = DatasetStatistics._empty_geom()
        if not anno_path or not os.path.exists(anno_path):
            return empty
        ext = os.path.splitext(anno_path)[1].lower()
        wh = []
        try:
            if ext == ".json":
                with open(anno_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    st = (shape.get("shape_type") or "").lower()
                    if st and st not in ("rectangle", "rect"):
                        continue
                    pts = shape.get("points") or []
                    if len(pts) < 2:
                        continue
                    # 两点矩形；多点时用外接轴对齐框
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                    wh.append((max(xs) - min(xs), max(ys) - min(ys)))
            elif ext == ".txt":
                with open(anno_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        # detect: cls cx cy w h ；pose 行更长，仍可用 w/h
                        if len(parts) < 5:
                            continue
                        w = float(parts[3])
                        h = float(parts[4])
                        wh.append((w, h))
            else:
                return empty
        except Exception:
            return empty
        return DatasetStatistics._geom_from_wh(wh)
    @staticmethod
    def generate_report(dir_path, classes_list=None, annotation_map=None, image_files=None):
        """
        生成完整的统计报告
        返回: dict
            {
                "total_images": 总图片数,
                "annotated_images": 已标注图片数,
                "unannotated_images": 未标注图片数,
                "total_annotations": 总标注数,
                "class_stats": {类别名: {"count": 出现次数, "images": 包含该类的图片数}},
                "unannotated_files": [无标注的图片路径列表],
                "annotated_files": [有标注的图片路径列表],
                "format_stats": {"json": json标注图片数, "yolo": yolo标注图片数, "xml": xml标注图片数},
            }

        annotation_map / image_files:
            打开 YOLO 布局时由 main 传入（images/ + labels/），避免只扫旁路 sidecar 导致全 0。
        """
        if annotation_map is not None:
            if image_files is None:
                image_files = sorted(annotation_map.keys())
        else:
            image_files, annotation_map = DatasetStatistics.scan_directory(dir_path)

        total_images = len(image_files)
        annotated_images = 0
        unannotated_images = 0
        unannotated_files = []
        annotated_files = []

        # 各类别统计: {类别名: {"count": 总出现次数, "images": set(图片路径)}}
        class_stats = {}
        total_annotations = 0

        # 格式统计
        format_stats = {"json": 0, "yolo": 0, "xml": 0}

        for img_path in image_files:
            anno_path = annotation_map.get(img_path) if annotation_map else None
            if anno_path and os.path.isfile(anno_path):
                annotated_images += 1
                annotated_files.append(img_path)
                ext = os.path.splitext(anno_path)[1].lower()
                if ext == '.json':
                    format_stats["json"] += 1
                elif ext == '.txt':
                    format_stats["yolo"] += 1
                elif ext == '.xml':
                    format_stats["xml"] += 1

                labels = DatasetStatistics.get_annotation_labels(anno_path, classes_list)
                total_annotations += len(labels)
                seen_labels_in_image = set()
                for label in labels:
                    if label not in class_stats:
                        class_stats[label] = {"count": 0, "images": set()}
                    class_stats[label]["count"] += 1
                    seen_labels_in_image.add(label)
                for label in seen_labels_in_image:
                    class_stats[label]["images"].add(img_path)
            else:
                unannotated_images += 1
                unannotated_files.append(img_path)

        # 将 set 转成 int 计数；并补齐 class_list 中 0 次的类别
        result_class_stats = {}
        for label, data in class_stats.items():
            result_class_stats[label] = {
                "count": data["count"],
                "images": len(data["images"])
            }
        if classes_list:
            for name in classes_list:
                if name and name not in result_class_stats:
                    result_class_stats[name] = {"count": 0, "images": 0}

        return {
            "total_images": total_images,
            "annotated_images": annotated_images,
            "unannotated_images": unannotated_images,
            "total_annotations": total_annotations,
            "class_stats": result_class_stats,
            "unannotated_files": unannotated_files,
            "annotated_files": annotated_files,
            "format_stats": format_stats,
        }

    @staticmethod
    def generate_current_image_report(scene, class_list):
        """
        统计当前画布上的标注情况
        返回: dict
            {
                "total": 总标注数,
                "class_stats": {类别名: 数量},
            }
        """
        from core.shapes import RectShape, PolyShape, PointShape, RotatedRectShape
        from core.pose_shapes import PoseShape

        class_stats = {}
        total = 0

        for item in scene.items():
            if isinstance(item, (RectShape, PolyShape, PointShape, RotatedRectShape, PoseShape)):
                if not getattr(item, 'is_temp', False):
                    label = getattr(item, 'label', '') or '未分类'
                    class_stats[label] = class_stats.get(label, 0) + 1
                    total += 1

        return {
            "total": total,
            "class_stats": class_stats,
        }
