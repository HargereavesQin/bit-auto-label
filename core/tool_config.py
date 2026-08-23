# -*- coding: utf-8 -*-
"""
工具窗口配置持久化 - YAML格式
每个工具窗口的配置独立存储，下次打开自动读取上次记录
"""
import os
import yaml


class ToolConfig:
    CONFIG_DIR = os.path.expanduser("~/.config/LuoHuaLabel")
    CONFIG_PATH = os.path.join(CONFIG_DIR, "tool_configs.yaml")

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = {}
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if os.path.exists(self.CONFIG_PATH):
            try:
                with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
                    self._data = yaml.safe_load(f) or {}
                self._loaded = True
            except Exception:
                self._data = {}
                self._loaded = True
        else:
            self._data = {}
            self._loaded = True

    def save(self):
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        try:
            with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._data, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"[ToolConfig] 保存配置失败: {e}")

    def get(self, tool_name, key, default=None):
        if not self._loaded:
            self.load()
        return self._data.get(tool_name, {}).get(key, default)

    def set(self, tool_name, key, value):
        if not self._loaded:
            self.load()
        if tool_name not in self._data:
            self._data[tool_name] = {}
        self._data[tool_name][key] = value

    def get_tool_config(self, tool_name):
        if not self._loaded:
            self.load()
        return dict(self._data.get(tool_name, {}))

    def set_tool_config(self, tool_name, config_dict):
        if not self._loaded:
            self.load()
        self._data[tool_name] = config_dict

    def save_path(self, tool_name, key, value):
        if value and os.path.exists(value):
            self.set(tool_name, key, value)
        else:
            self.set(tool_name, key, "")

    def get_path(self, tool_name, key, default=""):
        val = self.get(tool_name, key, default)
        if val and os.path.exists(val):
            return val
        return default
