# -*- coding: utf-8 -*-
"""
数据集融合核心逻辑
支持自动识别格式：YOLO已导出 / LabelMe原始 / Pascal VOC原始 / YOLO原始txt
"""
import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from PySide6.QtCore import QThread, Signal


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def detect_format(dir_path):
    """自动识别目录格式，返回格式字符串和详细信息"""
    dir_path = dir_path.strip()
    if not os.path.isdir(dir_path):
        return None, "目录不存在"

    yaml_path = os.path.join(dir_path, "data.yaml")
    images_dir = os.path.join(dir_path, "images")
    labels_dir = os.path.join(dir_path, "labels")

    has_yaml = os.path.exists(yaml_path)
    has_images = os.path.isdir(images_dir)
    has_labels = os.path.isdir(labels_dir)
    has_classes_txt = os.path.exists(os.path.join(dir_path, "classes.txt"))

    files = os.listdir(dir_path)
    has_json = any(f.lower().endswith('.json') for f in files)
    has_xml = any(f.lower().endswith('.xml') for f in files)
    has_txt = any(f.lower().endswith('.txt') for f in files)

    # 统计图片数量
    img_count = 0
    for f in files:
        if f.lower().endswith(IMAGE_EXTS):
            img_count += 1

    # 1. YOLO 已导出（含 data.yaml + images/ + labels/）
    if has_yaml and has_images and has_labels:
        format_type, sub_info = _detect_yolo_subtype(yaml_path, labels_dir)
        return {
            'type': 'yolo_exported',
            'subtype': format_type,
            'yaml_path': yaml_path,
            'images_dir': images_dir,
            'labels_dir': labels_dir,
            'img_count': img_count,
            'class_map': _read_yolo_yaml(yaml_path),
            'sub_info': sub_info,
        }, "YOLO已导出"

    # 2. YOLO 原始（含 .txt + classes.txt，无 yaml）
    if has_txt and has_classes_txt and not has_yaml:
        class_map = _read_classes_txt(os.path.join(dir_path, "classes.txt"))
        yolo_type = _detect_yolo_txt_type(dir_path)
        return {
            'type': 'yolo_raw',
            'subtype': yolo_type,
            'img_count': img_count,
            'class_map': class_map,
        }, f"YOLO原始({yolo_type})"

    # 3. LabelMe 原始（含 .json）
    if has_json:
        class_map = _detect_labelme_classes(dir_path)
        return {
            'type': 'labelme',
            'img_count': img_count,
            'class_map': class_map,
        }, "LabelMe原始"

    # 4. Pascal VOC 原始（含 .xml）
    if has_xml:
        class_map = _detect_voc_classes(dir_path)
        return {
            'type': 'voc',
            'img_count': img_count,
            'class_map': class_map,
        }, "Pascal VOC原始"

    return None, "无法识别的格式"


def _detect_yolo_subtype(yaml_path, labels_dir):
    """从 data.yaml 和 label 文件内容判断 YOLO 子格式"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否为 pose（kpt_shape）
    if 'kpt_shape' in content:
        return 'pose', 'Pose姿态'

    # 检查 labels 目录下的 .txt 文件列数
    txt_files = []
    for root_dir in [labels_dir, os.path.join(labels_dir, 'train'), os.path.join(labels_dir, 'val'), os.path.join(labels_dir, 'test')]:
        if os.path.isdir(root_dir):
            for f in os.listdir(root_dir):
                if f.endswith('.txt'):
                    txt_files.append(os.path.join(root_dir, f))
                    break
        if txt_files:
            break

    if not txt_files:
        return 'detect', '检测'

    # 读取第一个 label 文件判断列数
    with open(txt_files[0], 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    if not first_line:
        return 'detect', '检测'

    parts = first_line.split()
    col_count = len(parts) - 1  # 去掉 class_id

    if col_count == 4:
        return 'detect', '检测'
    elif col_count == 8:
        return 'obb', 'OBB旋转矩形'
    elif col_count == 6:
        return 'detect', '检测'  # cx,cy,w,h
    elif col_count == 10 or col_count > 10:
        # 可能是 segment（多边形点）或 pose
        if 'pose' in content.lower():
            return 'pose', 'Pose姿态'
        return 'seg', '分割'
    else:
        # 数一下第一个对象的点数
        nums = [p for p in parts[1:] if _is_float(p)]
        if len(nums) >= 6 and len(nums) % 2 == 0:
            return 'seg', '分割'
        return 'detect', '检测'


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _detect_yolo_txt_type(dir_path):
    """检测 YOLO 原始 txt 目录的子格式"""
    for f in os.listdir(dir_path):
        if f.endswith('.txt'):
            txt_path = os.path.join(dir_path, f)
            try:
                with open(txt_path, 'r', encoding='utf-8') as fp:
                    line = fp.readline().strip()
                if line:
                    parts = line.split()
                    col_count = len(parts) - 1
                    if col_count == 4:
                        return '检测'
                    elif col_count == 8:
                        return 'OBB'
                    elif col_count > 4:
                        return '分割'
            except Exception:
                pass
    return '检测'


def _read_yolo_yaml(yaml_path):
    """读取 data.yaml 提取 class_map"""
    class_map = {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#') and not line.startswith('path') and not line.startswith('train') and not line.startswith('val') and not line.startswith('test') and not line.startswith('nc') and not line.startswith('names') and not line.startswith('kpt') and not line.startswith('flip'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key.isdigit():
                        class_map[val] = int(key)
                    elif val.isdigit():
                        class_map[key] = int(val)
    return class_map


def _read_classes_txt(path):
    """读取 classes.txt 生成 class_map"""
    class_map = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    class_map[name] = i
    return class_map


def _detect_labelme_classes(dir_path):
    """从 LabelMe JSON 中提取所有类别"""
    class_map = {}
    next_id = 0
    for f in os.listdir(dir_path):
        if f.lower().endswith('.json'):
            try:
                with open(os.path.join(dir_path, f), 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                for shape in data.get('shapes', []):
                    label = shape.get('label', '')
                    if label and label not in class_map:
                        class_map[label] = next_id
                        next_id += 1
            except Exception:
                pass
    return class_map


def _detect_voc_classes(dir_path):
    """从 Pascal VOC XML 中提取所有类别"""
    class_map = {}
    next_id = 0
    for f in os.listdir(dir_path):
        if f.lower().endswith('.xml'):
            try:
                tree = ET.parse(os.path.join(dir_path, f))
                root = tree.getroot()
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None:
                        label = name_elem.text.strip()
                        if label and label not in class_map:
                            class_map[label] = next_id
                            next_id += 1
            except Exception:
                pass
    return class_map


def merge_class_maps(class_maps):
    """合并多个 class_map，返回统一映射表 {name: new_id}"""
    merged = {}
    next_id = 0
    for cm in class_maps:
        for name, old_id in cm.items():
            if name not in merged:
                merged[name] = next_id
                next_id += 1
    return merged


def remap_yolo_txt(txt_path, old_to_new):
    """重映射 YOLO .txt 文件中的 class ID"""
    if not os.path.exists(txt_path):
        return
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            old_id = int(parts[0])
            new_id = old_to_new.get(old_id, old_id)
            parts[0] = str(new_id)
            new_lines.append(' '.join(parts) + '\n')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def convert_labelme_to_yolo(json_path, img_path, out_txt_path, class_map, target_subtype='detect'):
    """将单个 LabelMe JSON 转换为 YOLO .txt"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_w = data.get('imageWidth', 1)
        img_h = data.get('imageHeight', 1)
        shapes = data.get('shapes', [])

        yolo_lines = []
        for shape in shapes:
            label = shape.get('label', '')
            if label not in class_map:
                continue
            class_id = class_map[label]
            stype = shape.get('shape_type', '')
            pts = shape.get('points', [])

            if stype == 'rectangle' and len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                bw = abs(x2 - x1) / img_w
                bh = abs(y2 - y1) / img_h
                cx = max(0.000001, min(0.999999, cx))
                cy = max(0.000001, min(0.999999, cy))
                bw = max(0.000001, min(0.999999, bw))
                bh = max(0.000001, min(0.999999, bh))
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            elif stype == 'polygon' and len(pts) >= 3:
                flat_pts = []
                for px, py in pts:
                    flat_pts.extend([f"{px / img_w:.6f}", f"{py / img_h:.6f}"])
                yolo_lines.append(f"{class_id} " + " ".join(flat_pts) + "\n")

            elif stype == 'point' and len(pts) == 1:
                cx = pts[0][0] / img_w
                cy = pts[0][1] / img_h
                pw, ph = 0.02, 0.02
                cx = max(pw / 2, min(1.0 - pw / 2, cx))
                cy = max(ph / 2, min(1.0 - ph / 2, cy))
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {pw:.6f} {ph:.6f}\n")

            elif stype == 'obb' and len(pts) >= 4:
                obb_pts = pts[:4]
                flat_pts = []
                for px, py in obb_pts:
                    flat_pts.extend([f"{px / img_w:.6f}", f"{py / img_h:.6f}"])
                yolo_lines.append(f"{class_id} " + " ".join(flat_pts) + "\n")

            elif stype == 'pose' and len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                bw = abs(x2 - x1) / img_w
                bh = abs(y2 - y1) / img_h
                kpts = shape.get('keypoints', [])
                parts = [f"{class_id}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
                for kpt in kpts:
                    kx = kpt[0] / img_w
                    ky = kpt[1] / img_h
                    kv = kpt[2] if len(kpt) > 2 else 2
                    parts.append(f"{kx:.6f} {ky:.6f} {int(kv)}")
                yolo_lines.append(" ".join(parts) + "\n")

        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.writelines(yolo_lines)
    except Exception as e:
        pass


def convert_voc_to_yolo(xml_path, img_path, out_txt_path, class_map):
    """将单个 Pascal VOC XML 转换为 YOLO .txt（检测格式）"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find('size')
        if size_elem is not None:
            w_elem = size_elem.find('width')
            h_elem = size_elem.find('height')
            img_w = float(w_elem.text) if w_elem is not None else 1
            img_h = float(h_elem.text) if h_elem is not None else 1
        else:
            img_w, img_h = 1, 1

        yolo_lines = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            label = name_elem.text.strip()
            if label not in class_map:
                continue
            class_id = class_map[label]

            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                cx = (xmin + xmax) / 2 / img_w
                cy = (ymin + ymax) / 2 / img_h
                bw = (xmax - xmin) / img_w
                bh = (ymax - ymin) / img_h
                cx = max(0.000001, min(0.999999, cx))
                cy = max(0.000001, min(0.999999, cy))
                bw = max(0.000001, min(0.999999, bw))
                bh = max(0.000001, min(0.999999, bh))
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.writelines(yolo_lines)
    except Exception as e:
        pass


def unique_name(base_name, ext, used_names):
    """生成不冲突的文件名"""
    if base_name + ext not in used_names:
        return base_name + ext
    i = 1
    while f"{base_name}_{i}{ext}" in used_names:
        i += 1
    return f"{base_name}_{i}{ext}"


class DatasetFusionWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    stat_signal = Signal(dict)
    finish_signal = Signal(bool, str)

    def __init__(self, sources, output_dir, output_format, split_policy, ratios):
        super().__init__()
        self.sources = sources
        self.output_dir = output_dir
        self.output_format = output_format
        self.split_policy = split_policy
        self.train_r, self.val_r, self.test_r = ratios

    def run(self):
        try:
            self.log_signal.emit("=" * 50)
            self.log_signal.emit("开始数据集融合...")

            # 1. 检测每个源的格式
            self.log_signal.emit("\n[步骤1] 检测源文件夹格式...")
            source_info = []
            class_maps = []
            for i, src in enumerate(self.sources):
                info, label = detect_format(src)
                if info is None:
                    self.finish_signal.emit(False, f"无法识别目录格式: {src}")
                    return
                info['path'] = src
                info['label'] = label
                source_info.append(info)
                class_maps.append(info.get('class_map', {}))
                self.log_signal.emit(f"  源{i + 1}: {os.path.basename(src)} → {label}, "
                                    f"图片:{info['img_count']}张, 类别:{len(info.get('class_map', {}))}个")

            # 2. 检查 YOLO 子格式是否一致
            self.log_signal.emit("\n[步骤2] 验证格式兼容性...")
            yolo_subtypes = [s['subtype'] for s in source_info if s['type'] in ('yolo_exported', 'yolo_raw')]
            if yolo_subtypes:
                first = yolo_subtypes[0]
                for j, st in enumerate(yolo_subtypes[1:], 1):
                    if st != first:
                        self.finish_signal.emit(False, f"YOLO 子格式不一致: 源{j + 1} 是 {st}，源1 是 {first}。融合只支持同类型。")
                        return
                self.log_signal.emit(f"  YOLO 子格式一致: {first}")

            # 3. 合并类别映射
            self.log_signal.emit("\n[步骤3] 合并类别列表...")
            merged_class_map = merge_class_maps(class_maps)
            self.log_signal.emit(f"  合并后类别数: {len(merged_class_map)}")
            for name, cid in sorted(merged_class_map.items(), key=lambda x: x[1]):
                self.log_signal.emit(f"    [{cid}] {name}")

            # 4. 创建输出目录结构
            self.log_signal.emit("\n[步骤4] 创建输出目录结构...")
            img_train = os.path.join(self.output_dir, 'images', 'train')
            img_val = os.path.join(self.output_dir, 'images', 'val')
            img_test = os.path.join(self.output_dir, 'images', 'test')
            lbl_train = os.path.join(self.output_dir, 'labels', 'train')
            lbl_val = os.path.join(self.output_dir, 'labels', 'val')
            lbl_test = os.path.join(self.output_dir, 'labels', 'test')
            for d in [img_train, img_val, img_test, lbl_train, lbl_val, lbl_test]:
                os.makedirs(d, exist_ok=True)
            self.log_signal.emit(f"  输出目录: {self.output_dir}")

            # 5. 收集所有图片对 (img_path, label_path, src_idx, split_from)
            self.log_signal.emit("\n[步骤5] 收集并处理数据...")
            all_pairs = []
            for src_idx, info in enumerate(source_info):
                src_path = info['path']
                src_type = info['type']
                self.log_signal.emit(f"\n  处理源{src_idx + 1}: {os.path.basename(src_path)} ({info['label']})")

                if src_type == 'yolo_exported':
                    pairs = self._collect_yolo_exported(info, src_idx)
                elif src_type == 'yolo_raw':
                    pairs = self._collect_yolo_raw(info, src_idx)
                elif src_type == 'labelme':
                    pairs = self._collect_labelme(info, src_idx)
                elif src_type == 'voc':
                    pairs = self._collect_voc(info, src_idx)
                else:
                    pairs = []
                all_pairs.extend(pairs)
                self.log_signal.emit(f"    收集到 {len(pairs)} 个图片对")

            total = len(all_pairs)
            self.log_signal.emit(f"\n  总计: {total} 个图片对")

            # 6. 划分数据集
            self.log_signal.emit("\n[步骤6] 划分数据集...")
            if self.split_policy == 'shuffle':
                random.shuffle(all_pairs)
                train_c = int(total * self.train_r)
                val_c = int(total * self.val_r)
                train_pairs = all_pairs[:train_c]
                val_pairs = all_pairs[train_c:train_c + val_c]
                test_pairs = all_pairs[train_c + val_c:]
            else:
                train_pairs = [(p, l, si, s) for p, l, si, s in all_pairs if s == 'train']
                val_pairs = [(p, l, si, s) for p, l, si, s in all_pairs if s == 'val']
                test_pairs = [(p, l, si, s) for p, l, si, s in all_pairs if s == 'test']
                self.log_signal.emit(f"  保持原划分: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

            train_c = len(train_pairs)
            val_c = len(val_pairs)
            test_c = len(test_pairs)
            self.log_signal.emit(f"  划分结果: train={train_c}, val={val_c}, test={test_c}")

            # 7. 复制和转换文件
            self.log_signal.emit("\n[步骤7] 复制和转换文件...")
            used_names = set()
            stats = {'total': total, 'train': train_c, 'val': val_c, 'test': test_c,
                     'classes': len(merged_class_map), 'class_counts': {}}

            for name in merged_class_map:
                stats['class_counts'][name] = 0

            def process_pairs(pairs, split_name):
                cnt = 0
                for img_path, label_path, src_idx, _ in pairs:
                    src_info = source_info[src_idx]
                    src_type = src_info['type']
                    orig_name = os.path.splitext(os.path.basename(img_path))[0]
                    new_base = unique_name(f"{split_name}_{orig_name}", '', used_names)
                    used_names.add(new_base)
                    new_img_name = new_base + os.path.splitext(img_path)[1]
                    new_lbl_name = new_base + '.txt'

                    # 复制图片
                    dst_img = os.path.join(self.output_dir, 'images', split_name, new_img_name)
                    shutil.copy2(img_path, dst_img)

                    # 处理标注
                    if label_path and os.path.exists(label_path):
                        if src_type in ('yolo_exported', 'yolo_raw'):
                            # 直接复制 YOLO txt（需要重映射 ID）
                            old_map = source_info[src_idx].get('class_map', {})
                            new_map = merged_class_map
                            old_to_new = {old_id: new_map[name] for name, old_id in old_map.items()}
                            dst_lbl = os.path.join(self.output_dir, 'labels', split_name, new_lbl_name)
                            shutil.copy2(label_path, dst_lbl)
                            if old_to_new:
                                remap_yolo_txt(dst_lbl, old_to_new)
                            self._count_labels(dst_lbl, merged_class_map, stats['class_counts'])
                        elif src_type == 'labelme':
                            dst_lbl = os.path.join(self.output_dir, 'labels', split_name, new_lbl_name)
                            convert_labelme_to_yolo(label_path, img_path, dst_lbl, merged_class_map)
                            self._count_labels(dst_lbl, merged_class_map, stats['class_counts'])
                        elif src_type == 'voc':
                            dst_lbl = os.path.join(self.output_dir, 'labels', split_name, new_lbl_name)
                            convert_voc_to_yolo(label_path, img_path, dst_lbl, merged_class_map)
                            self._count_labels(dst_lbl, merged_class_map, stats['class_counts'])

                    cnt += 1
                    if cnt % 50 == 0 or cnt == len(pairs):
                        self.progress_signal.emit(cnt, len(pairs))
                        self.log_signal.emit(f"    {split_name}: {cnt}/{len(pairs)}")

            process_pairs(train_pairs, 'train')
            process_pairs(val_pairs, 'val')
            process_pairs(test_pairs, 'test')

            # 8. 生成 data.yaml 和 classes.txt
            self.log_signal.emit("\n[步骤8] 生成配置文件...")
            self._generate_yaml(merged_class_map, stats)
            self._generate_classes_txt(merged_class_map)

            self.log_signal.emit("\n" + "=" * 50)
            self.log_signal.emit("融合完成！")
            self.log_signal.emit(f"  总图片: {total}")
            self.log_signal.emit(f"  train: {train_c}, val: {val_c}, test: {test_c}")
            self.log_signal.emit(f"  类别数: {len(merged_class_map)}")

            stats['merged_class_map'] = merged_class_map
            self.stat_signal.emit(stats)
            self.finish_signal.emit(True, f"融合完成！总计 {total} 张图片，{len(merged_class_map)} 个类别")

        except Exception as e:
            import traceback
            self.log_signal.emit(f"\n错误: {e}\n{traceback.format_exc()}")
            self.finish_signal.emit(False, str(e))

    def _collect_yolo_exported(self, info, src_idx):
        pairs = []
        labels_dir = info['labels_dir']
        images_dir = info['images_dir']

        # 遍历 train/val/test 子目录
        for split in ['train', 'val', 'test']:
            lbl_split = os.path.join(labels_dir, split)
            img_split = os.path.join(images_dir, split)
            if not os.path.isdir(lbl_split):
                continue
            if not os.path.isdir(img_split):
                continue

            lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_split) if f.endswith('.txt')}
            for lbl_file in lbl_files:
                lbl_path = os.path.join(lbl_split, lbl_file + '.txt')
                # 查找对应图片
                img_found = None
                for ext in IMAGE_EXTS:
                    candidate = os.path.join(img_split, lbl_file + ext)
                    if os.path.exists(candidate):
                        img_found = candidate
                        break
                if img_found:
                    pairs.append((img_found, lbl_path, src_idx, split))
        return pairs

    def _collect_yolo_raw(self, info, src_idx):
        pairs = []
        src_path = info['path']
        files = os.listdir(src_path)
        basenames = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.txt')}
        for bn in basenames:
            lbl_path = os.path.join(src_path, bn + '.txt')
            img_found = None
            for ext in IMAGE_EXTS:
                candidate = os.path.join(src_path, bn + ext)
                if os.path.exists(candidate):
                    img_found = candidate
                    break
            if img_found:
                pairs.append((img_found, lbl_path, src_idx, 'train'))
        return pairs

    def _collect_labelme(self, info, src_idx):
        pairs = []
        src_path = info['path']
        files = os.listdir(src_path)
        basenames = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.json')}
        for bn in basenames:
            json_path = os.path.join(src_path, bn + '.json')
            img_found = None
            for ext in IMAGE_EXTS:
                candidate = os.path.join(src_path, bn + ext)
                if os.path.exists(candidate):
                    img_found = candidate
                    break
            if img_found:
                pairs.append((img_found, json_path, src_idx, 'train'))
        return pairs

    def _collect_voc(self, info, src_idx):
        pairs = []
        src_path = info['path']
        files = os.listdir(src_path)
        basenames = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.xml')}
        for bn in basenames:
            xml_path = os.path.join(src_path, bn + '.xml')
            img_found = None
            for ext in IMAGE_EXTS:
                candidate = os.path.join(src_path, bn + ext)
                if os.path.exists(candidate):
                    img_found = candidate
                    break
            if img_found:
                pairs.append((img_found, xml_path, src_idx, 'train'))
        return pairs

    def _count_labels(self, lbl_path, class_map, class_counts):
        if not os.path.exists(lbl_path):
            return
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        for name, c in class_map.items():
                            if c == cid:
                                class_counts[name] = class_counts.get(name, 0) + 1
                                break
        except Exception:
            pass

    def _generate_yaml(self, class_map, stats):
        yaml_path = os.path.join(self.output_dir, 'data.yaml')
        abs_out = os.path.abspath(self.output_dir).replace("\\", "/")
        nc = len(class_map)
        names_dict = {v: k for k, v in class_map.items()}

        lines = []
        lines.append(f"# 数据路径")
        lines.append(f"path: {abs_out}")
        lines.append(f"train: images/train")
        lines.append(f"val: images/val")
        lines.append(f"test: images/test")
        lines.append(f"\n# Number of classes")
        lines.append(f"nc: {nc}")
        lines.append(f"\n# Class names")
        lines.append(f"names:")
        for i in range(nc):
            lines.append(f"  {i}: {names_dict[i]}")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

    def _generate_classes_txt(self, class_map):
        txt_path = os.path.join(self.output_dir, 'classes.txt')
        names = sorted(class_map.keys(), key=lambda k: class_map[k])
        with open(txt_path, 'w', encoding='utf-8') as f:
            for name in names:
                f.write(name + "\n")
