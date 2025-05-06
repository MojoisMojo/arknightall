from PyQt6.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QGroupBox, QLabel
from PyQt6.QtCore import Qt, pyqtSignal # Import pyqtSignal

# 导入设置模块
from src.core import settings

class SettingsTab(QWidget):
    """设置标签页的 UI"""
    # Signal emitted when a setting changes. Passes the setting key (str) and new value (bool).
    setting_changed = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_initial_settings()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop) # 内容顶部对齐

        # --- 显示设置组 ---
        display_group = QGroupBox("显示设置")
        display_layout = QVBoxLayout()

        self.hide_image_checkbox = QCheckBox("隐藏识别结果图像")
        self.hide_image_checkbox.stateChanged.connect(self.save_hide_image_setting)
        display_layout.addWidget(self.hide_image_checkbox)

        self.simplify_card_checkbox = QCheckBox("简化怪物卡片（仅显示名称、数量、头像）")
        self.simplify_card_checkbox.stateChanged.connect(self.save_simplify_card_setting)
        display_layout.addWidget(self.simplify_card_checkbox)

        self.always_on_top_checkbox = QCheckBox("窗口置顶")
        self.always_on_top_checkbox.stateChanged.connect(self.save_always_on_top_setting)
        display_layout.addWidget(self.always_on_top_checkbox)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # --- 其他设置组 (示例) ---
        # other_group = QGroupBox("其他设置")
        # other_layout = QVBoxLayout()
        # other_layout.addWidget(QLabel("未来可以添加更多设置"))
        # other_group.setLayout(other_layout)
        # layout.addWidget(other_group)

        self.setLayout(layout)

    def load_initial_settings(self):
        """加载初始设置并更新 UI"""
        # 加载显示设置
        hide_image_value = settings.get_setting("hide_recognition_image", False)
        self.hide_image_checkbox.setChecked(hide_image_value)

        simplify_card_value = settings.get_setting("simplify_monster_card", False)
        self.simplify_card_checkbox.setChecked(simplify_card_value)

        # 加载窗口置顶设置
        always_on_top_value = settings.get_setting("always_on_top", False)
        self.always_on_top_checkbox.setChecked(always_on_top_value)

    def save_hide_image_setting(self, state):
        """保存隐藏识别结果图像复选框的状态，并发出信号"""
        is_checked = (state == Qt.CheckState.Checked.value)
        settings.update_setting("hide_recognition_image", is_checked)
        #print(f"设置 'hide_recognition_image' 已保存为: {is_checked}")
        self.setting_changed.emit("hide_recognition_image", is_checked) # Emit signal

    def save_simplify_card_setting(self, state):
        """保存简化怪物卡片复选框的状态"""
        is_checked = (state == Qt.CheckState.Checked.value)
        settings.update_setting("simplify_monster_card", is_checked)
        #print(f"设置 'simplify_monster_card' 已保存为: {is_checked}")

    def save_always_on_top_setting(self, state):
        """保存窗口置顶复选框的状态，并发出信号"""
        is_checked = (state == Qt.CheckState.Checked.value)
        settings.update_setting("always_on_top", is_checked)
        #print(f"设置 'always_on_top' 已保存为: {is_checked}")
        self.setting_changed.emit("always_on_top", is_checked) # Emit signal

# 可以添加一个简单的运行块来单独测试这个标签页
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    main_win = QMainWindow()
    settings_widget = SettingsTab()
    main_win.setCentralWidget(settings_widget)
    main_win.setWindowTitle("设置标签页测试")
    main_win.setGeometry(200, 200, 400, 300)
    main_win.show()
    sys.exit(app.exec())