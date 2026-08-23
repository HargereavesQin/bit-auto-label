# -*- coding: utf-8 -*-
"""
位姿类别定义管理模块
"""
import os
import json


class PoseConfig:
    """位姿类别配置管理"""

    def __init__(self, config_path=None):
        self.config_path = config_path or ""
        self.categories = {}  # {类别名: {"kpt_count": N, "keypoints": [名称列表]}}

    def load(self, config_path=None):
        """从文件加载位姿类别定义"""
        if config_path:
            self.config_path = config_path
        if not self.config_path or not os.path.exists(self.config_path):
            self.categories = {}
            return False

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.categories = data.get("categories", {})
            return True
        except Exception:
            self.categories = {}
            return False

    def save(self, config_path=None):
        """保存位姿类别定义到文件"""
        if config_path:
            self.config_path = config_path
        if not self.config_path:
            return False

        try:
            data = {"categories": self.categories}
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def add_category(self, name, keypoints):
        """
        添加位姿类别
        name: 类别名
        keypoints: 关键点名称列表（有序）
        """
        self.categories[name] = {
            "kpt_count": len(keypoints),
            "keypoints": keypoints
        }

    def remove_category(self, name):
        """删除位姿类别"""
        if name in self.categories:
            del self.categories[name]

    def get_category(self, name):
        """获取位姿类别定义"""
        return self.categories.get(name)

    def get_kpt_count(self, name):
        """获取关键点数量"""
        cat = self.categories.get(name)
        return cat["kpt_count"] if cat else 0

    def get_kpt_names(self, name):
        """获取关键点名称列表"""
        cat = self.categories.get(name)
        return cat["keypoints"] if cat else []

    def get_category_names(self):
        """获取所有位姿类别名列表"""
        return list(self.categories.keys())

    def has_category(self, name):
        """检查是否存在该位姿类别"""
        return name in self.categories
