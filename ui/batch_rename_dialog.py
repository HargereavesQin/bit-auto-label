# -*- coding: utf-8 -*-
"""
批量重命名对话框 — RM 主题.

将当前目录下的所有图片和标注文件重命名为序号格式, 同步更新
JSON/XML 内部 imagePath / filename 引用.
"""
import os
import json
import shutil
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton,
    QGroupBox, QTextEdit, QComboBox, QLineEdit,
    QSpinBox, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal

from .icons import Icon, Icons, C
from .dialog_utils import build_dialog_header


class BatchRenameWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finished_signal = Signal(bool, str)

    def __init__(self, dir_path, file_list, start_index, digit_count):
        super().__init__()
        self.dir_path = dir_path
        self.file_list = file_list
        self.start_index = start_index
        self.digit_count = digit_count

    def run(self):
        try:
            total = len(self.file_list)
            self.log_signal.emit(f"开始重命名 {total} 个文件...")

            renamed_count = 0
            skipped_count = 0

            for i, (old_img_path, _) in enumerate(self.file_list):
                old_base, old_ext = os.path.splitext(os.path.basename(old_img_path))
                new_name = f"{self.start_index + i:0{self.digit_count}d}{old_ext.lower()}"

                if old_base == new_name.replace(old_ext.lower(), ''):
                    skipped_count += 1
                    continue

                tmp_name = f"__rename_tmp__{self.start_index + i:0{self.digit_count}d}{old_ext.lower()}"
                tmp_path = os.path.join(self.dir_path, tmp_name)

                if os.path.exists(tmp_path):
                    self.log_signal.emit(f"  警告: 临时文件已存在, 跳过: {old_img_path}")
                    skipped_count += 1
                    continue

                os.rename(old_img_path, tmp_path)

                for ext in ['.json', '.xml', '.txt']:
                    old_anno = os.path.join(self.dir_path, old_base + ext)
                    if os.path.exists(old_anno):
                        tmp_anno = os.path.join(self.dir_path, tmp_name.replace(old_ext.lower(), ext))
                        os.rename(old_anno, tmp_anno)

                new_img_path = os.path.join(self.dir_path, new_name)
                os.rename(tmp_path, new_img_path)

                for ext in ['.json', '.xml', '.txt']:
                    tmp_anno = os.path.join(self.dir_path, tmp_name.replace(old_ext.lower(), ext))
                    new_anno = os.path.join(self.dir_path, new_name.replace(old_ext.lower(), ext))
                    if os.path.exists(tmp_anno):
                        os.rename(tmp_anno, new_anno)
                        self._update_annotation_ref(new_anno, ext, new_name)

                renamed_count += 1
                if renamed_count % 20 == 0:
                    self.progress_signal.emit(renamed_count, total)
                    self.log_signal.emit(f"  已处理 {renamed_count}/{total} ...")

            self.progress_signal.emit(renamed_count, total)
            self.log_signal.emit(f"\n重命名完成! 成功: {renamed_count}, 跳过(已是目标名): {skipped_count}")
            self.finished_signal.emit(True, f"成功重命名 {renamed_count} 个文件")

        except Exception as e:
            import traceback
            self.log_signal.emit(f"错误: {e}\n{traceback.format_exc()}")
            self.finished_signal.emit(False, str(e))

    def _update_annotation_ref(self, anno_path, ext, new_name):
        try:
            if ext == '.json':
                with open(anno_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['imagePath'] = new_name
                with open(anno_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif ext == '.xml':
                tree = ET.parse(anno_path)
                root = tree.getroot()
                fn_elem = root.find('filename')
                if fn_elem is not None:
                    fn_elem.text = new_name
                folder_elem = root.find('folder')
                if folder_elem is not None:
                    folder_elem.text = 'images'
                tree.write(anno_path, encoding='utf-8', xml_declaration=True)
        except Exception:
            pass


class BatchRenameDialog(QDialog):
    def __init__(self, dir_path, file_list, parent=None):
        super().__init__(parent)
        self.dir_path = dir_path
        self.file_list = file_list
        self.setWindowTitle("批量重命名")
        self.setWindowIcon(Icon.icon(Icons.RENAME, size=24, color=C.AMBER))
        self.setMinimumSize(700, 540)
        self._init_ui()
        self._load_config()
        self._update_preview()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(build_dialog_header(
            self, Icons.RENAME,
            "批量重命名", "BATCH RENAME  /  序号格式批量改名"))

        body = QVBoxLayout()
        body.setContentsMargins(16, 8, 16, 8)
        body.setSpacing(10)
        outer.addLayout(body, 1)

        # 目录信息
        info_group = QGroupBox("当前目录信息  /  DIRECTORY")
        info_group.setProperty("accent", "muted")
        info_layout = QGridLayout(info_group)
        info_layout.setContentsMargins(10, 14, 10, 10)
        info_layout.setSpacing(8)

        info_layout.addWidget(QLabel("目录:"), 0, 0)
        self.dir_label = QLabel(self.dir_path)
        self.dir_label.setProperty("role", "caption")
        self.dir_label.setWordWrap(True)
        info_layout.addWidget(self.dir_label, 0, 1, 1, 3)

        total_images = len(self.file_list)
        json_count = sum(1 for _, bn in self.file_list if os.path.exists(os.path.join(self.dir_path, bn + '.json')))
        xml_count = sum(1 for _, bn in self.file_list if os.path.exists(os.path.join(self.dir_path, bn + '.xml')))
        txt_count = sum(1 for _, bn in self.file_list if os.path.exists(os.path.join(self.dir_path, bn + '.txt')))

        self.counts_label = QLabel(
            f"图片: <b style='color:#C9A36B;'>{total_images}</b> 张  |  "
            f"JSON: <b style='color:#8FA68F;'>{json_count}</b> 个  |  "
            f"XML: <b style='color:#8FA68F;'>{xml_count}</b> 个  |  "
            f"TXT: <b style='color:#8C8C95;'>{txt_count}</b> 个"
        )
        self.counts_label.setTextFormat(Qt.RichText)
        info_layout.addWidget(self.counts_label, 1, 0, 1, 4)
        body.addWidget(info_group)

        # 重命名参数
        param_group = QGroupBox("重命名参数  /  PARAMETERS")
        param_group.setProperty("accent", "amber")
        param_layout = QGridLayout(param_group)
        param_layout.setContentsMargins(10, 14, 10, 10)
        param_layout.setSpacing(10)

        param_layout.addWidget(QLabel("起始序号:"), 0, 0)
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 999999)
        self.start_spin.setValue(0)
        self.start_spin.setProperty("role", "mono")
        param_layout.addWidget(self.start_spin, 0, 1)

        param_layout.addWidget(QLabel("位数:"), 0, 2)
        self.digit_spin = QSpinBox()
        self.digit_spin.setRange(1, 6)
        self.digit_spin.setValue(6)
        self.digit_spin.setProperty("role", "mono")
        param_layout.addWidget(self.digit_spin, 0, 3)

        param_layout.addWidget(QLabel("示例:"), 1, 0)
        self.example_label = QLabel()
        self.example_label.setProperty("role", "mono_strong")
        param_layout.addWidget(self.example_label, 1, 1, 1, 3)
        body.addWidget(param_group)

        # 预览
        preview_group = QGroupBox("预览  /  PREVIEW (前 10 个)")
        preview_group.setProperty("accent", "blue")
        pl = QVBoxLayout(preview_group)
        pl.setContentsMargins(10, 14, 10, 10)
        self.preview_list = QListWidget()
        self.preview_list.setMinimumHeight(140)
        pl.addWidget(self.preview_list)
        body.addWidget(preview_group)

        # 日志
        log_label = QLabel("处理日志  /  LOG")
        log_label.setProperty("role", "section_header")
        body.addWidget(log_label)
        self.console = QTextEdit()
        self.console.setMaximumHeight(120)
        self.console.setReadOnly(True)
        self.console.setProperty("role", "mono")
        body.addWidget(self.console)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(16, 8, 16, 14)
        btn_row.setSpacing(8)
        btn_row.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setProperty("role", "ghost")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.confirm_btn = QPushButton("  确定重命名")
        self.confirm_btn.setCursor(Qt.PointingHandCursor)
        self.confirm_btn.setProperty("role", "primary")
        self.confirm_btn.setIcon(Icon.icon(Icons.RENAME, size=14, color=C.BG_APP))
        self.confirm_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(self.confirm_btn)
        outer.addLayout(btn_row)

        self.start_spin.valueChanged.connect(self._update_preview)
        self.digit_spin.valueChanged.connect(self._update_preview)

    def _load_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig().get_tool_config("batch_rename")
        self.start_spin.setValue(int(cfg.get("start_index", 0)))
        self.digit_spin.setValue(int(cfg.get("digit_count", 6)))

    def _save_config(self):
        from core.tool_config import ToolConfig
        cfg = ToolConfig()
        cfg.set("batch_rename", "start_index", self.start_spin.value())
        cfg.set("batch_rename", "digit_count", self.digit_spin.value())
        cfg.save()

    def closeEvent(self, event):
        self._save_config()
        event.accept()

    def _update_preview(self):
        self.preview_list.clear()
        start = self.start_spin.value()
        digits = self.digit_spin.value()
        ext_sample = os.path.splitext(self.file_list[0][0])[1] if self.file_list else '.ext'
        self.example_label.setText(
            f"{start:0{digits}d}{ext_sample}  →  {start + len(self.file_list) - 1:0{digits}d}{ext_sample}"
        )

        for i, (img_path, _) in enumerate(self.file_list[:10]):
            old_name = os.path.basename(img_path)
            ext = os.path.splitext(img_path)[1]
            new_name = f"{start + i:0{digits}d}{ext.lower()}"
            item = QListWidgetItem(f"  {old_name:<40}  →  {new_name}")
            self.preview_list.addItem(item)

        if len(self.file_list) > 10:
            self.preview_list.addItem(f"  ... 共 {len(self.file_list)} 个文件")

    def _on_confirm(self):
        total = len(self.file_list)
        start = self.start_spin.value()
        digits = self.digit_spin.value()
        last_index = start + total - 1
        max_index = 10 ** digits - 1

        if last_index > max_index:
            QMessageBox.warning(self, "序号溢出",
                f"当前设置 ({start} ~ {last_index}) 超出 {digits} 位能表示的最大值 ({max_index})!\n"
                f"请增加位数或减少文件数量。")
            return

        reply = QMessageBox.question(self, "确认重命名",
            f"确定要将 <b>{total}</b> 个文件重命名为序号格式吗?\n\n"
            f"起始: {start}, 位数: {digits}\n"
            f"示例: {start:0{digits}d}{os.path.splitext(self.file_list[0][0])[1] if self.file_list else '.ext'}\n\n"
            f"此操作不可逆!",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self.confirm_btn.setEnabled(False)
        self.console.clear()

        self.worker = BatchRenameWorker(self.dir_path, self.file_list, start, digits)
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _append_log(self, msg):
        self.console.append(msg)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _on_progress(self, current, total):
        self.confirm_btn.setText(f"处理中... {current}/{total}")

    def _on_finished(self, success, msg):
        self.confirm_btn.setText("  确定重命名")
        self.confirm_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", msg)
            self.accept()
        else:
            QMessageBox.warning(self, "错误", msg)
