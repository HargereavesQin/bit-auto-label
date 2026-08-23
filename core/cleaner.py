# -*- coding: utf-8 -*-
"""
批量删除模块 - 批量删除无标注文件的图片
"""
import os
import shutil


class DatasetCleaner:
    """数据集清理工具"""

    @staticmethod
    def find_unlabeled_images(dir_path):
        """
        找出目录下所有没有对应标注文件的图片
        返回: [无标注的图片路径列表]
        """
        from core.statistics import DatasetStatistics
        image_files, annotation_map = DatasetStatistics.scan_directory(dir_path)

        unlabeled = []
        for img_path in image_files:
            if annotation_map.get(img_path) is None:
                unlabeled.append(img_path)

        return unlabeled

    @staticmethod
    def find_empty_label_files(dir_path):
        """
        找出目录下所有标注文件为空（无有效标注内容）的图片
        返回: [(图片路径, 标注文件路径, 标注格式)]
        """
        from core.statistics import DatasetStatistics
        image_files, annotation_map = DatasetStatistics.scan_directory(dir_path)

        empty_annotated = []
        for img_path in image_files:
            anno_path = annotation_map.get(img_path)
            if anno_path is None:
                continue

            labels = DatasetStatistics.get_annotation_labels(anno_path)
            if not labels:
                # 标注文件存在但内容为空
                ext = os.path.splitext(anno_path)[1].lower()
                empty_annotated.append((img_path, anno_path, ext))

        return empty_annotated

    @staticmethod
    def delete_images(image_paths, to_trash=True):
        """
        批量删除图片文件（及对应的标注文件）
        参数:
            image_paths: 要删除的图片路径列表
            to_trash: True=移动到回收站目录, False=彻底删除
        返回: (成功数, 失败列表)
        """
        success_count = 0
        failed = []

        if to_trash:
            # 在图片所在目录创建 .trash 文件夹作为回收站
            if not image_paths:
                return 0, []
            base_dir = os.path.dirname(image_paths[0])
            trash_dir = os.path.join(base_dir, ".trash")
            os.makedirs(trash_dir, exist_ok=True)

        for img_path in image_paths:
            try:
                if to_trash:
                    # 移动到回收站
                    dest = os.path.join(trash_dir, os.path.basename(img_path))
                    shutil.move(img_path, dest)
                else:
                    # 彻底删除
                    os.remove(img_path)
                success_count += 1
            except Exception as e:
                failed.append((img_path, str(e)))

        return success_count, failed

    @staticmethod
    def delete_images_with_annotations(image_paths, to_trash=True):
        """
        批量删除图片及其对应的标注文件
        参数:
            image_paths: 要删除的图片路径列表
            to_trash: True=移动到回收站目录, False=彻底删除
        返回: (成功数, 失败列表)
        """
        from core.statistics import DatasetStatistics
        success_count = 0
        failed = []

        if not image_paths:
            return 0, []

        if to_trash:
            base_dir = os.path.dirname(image_paths[0])
            trash_dir = os.path.join(base_dir, ".trash")
            os.makedirs(trash_dir, exist_ok=True)

        for img_path in image_paths:
            try:
                # 查找对应的标注文件
                base_name = os.path.splitext(img_path)[0]
                anno_extensions = ('.json', '.txt', '.xml')
                files_to_delete = [img_path]

                for ext in anno_extensions:
                    anno_path = base_name + ext
                    if os.path.exists(anno_path):
                        files_to_delete.append(anno_path)

                for file_path in files_to_delete:
                    if to_trash:
                        dest = os.path.join(trash_dir, os.path.basename(file_path))
                        # 处理重名
                        counter = 1
                        while os.path.exists(dest):
                            name, ext = os.path.splitext(os.path.basename(file_path))
                            dest = os.path.join(trash_dir, f"{name}_{counter}{ext}")
                            counter += 1
                        shutil.move(file_path, dest)
                    else:
                        os.remove(file_path)

                success_count += 1
            except Exception as e:
                failed.append((img_path, str(e)))

        return success_count, failed

    @staticmethod
    def restore_from_trash(dir_path):
        """
        从回收站恢复所有文件
        返回: (成功数, 失败列表)
        """
        trash_dir = os.path.join(dir_path, ".trash")
        if not os.path.isdir(trash_dir):
            return 0, []

        success_count = 0
        failed = []

        for f in os.listdir(trash_dir):
            src = os.path.join(trash_dir, f)
            dst = os.path.join(dir_path, f)
            try:
                # 如果目标已存在，跳过
                if os.path.exists(dst):
                    failed.append((src, f"目标文件已存在: {dst}"))
                    continue
                shutil.move(src, dst)
                success_count += 1
            except Exception as e:
                failed.append((src, str(e)))

        # 如果回收站空了，删除目录
        try:
            if os.path.isdir(trash_dir) and not os.listdir(trash_dir):
                os.rmdir(trash_dir)
        except Exception:
            pass

        return success_count, failed
