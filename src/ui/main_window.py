import logging  # Import logging module
import os
import sys
from typing import Dict, Literal, List, Optional, Tuple

import cv2
import mss  # Keep mss import here as it's used directly for capture
import numpy as np
import json # Added for mapping
import re # Added for extracting ID from template name
from PyQt6.QtCore import Qt, QRect, QUrl, pyqtSignal, QTimer, pyqtSlot, QPoint
# Added QAction, QIntValidator
from PyQt6.QtGui import QCloseEvent # Added for closeEvent
from PyQt6.QtGui import QPixmap, QTextOption, QMouseEvent, QImage, QAction, QFont, QIntValidator, QPainter, QColor # Added QPainter, QColor
# import sqlite3 # No longer needed directly here
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QScrollArea, QSizePolicy, QTextBrowser, QMenu, QLineEdit, QTabWidget, QFrame
    # QDialog, QTextEdit, QSpinBox, QFormLayout, QListWidget, QListWidgetItem # Moved to manager
)

# Adjust import paths to work when run from main.py or directly
# This assumes main.py is in the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the screen selection widget
from src.core.screen_capture import ScreenSelectionWidget
from src.core.image_recognition import load_templates, recognize_monsters
from src.core.data_loader import load_monster_data
from src.models.monster import Monster # Import the Monster model
from src.ui.damage_info_window import DamageInfoWindow # Import the new window
from src.ui.mistake_book_manager import MistakeBookManager, MistakeBookEntryDialog, MistakeBookQueryDialog # Import mistake book components
from src.core.log import logger # Import the logger
from src.core import prediction # Import the prediction module
from src.ui.settings_tab import SettingsTab # Import SettingsTab
from src.core import settings # Import settings module
# No longer need ImageViewer
# from src.ui.image_viewer import ImageViewer
from src.ui.monster_card_widget import MonsterDisplayArea # Import the new display area widget

# Define paths relative to this file's potential execution context
# This might need adjustment depending on how the app is run/packaged
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TEMPLATE_DIR = os.path.join(DATA_DIR, 'image') # Keep for manual add list
MONSTER_CSV_PATH = os.path.join(DATA_DIR, 'monster.csv')
# MISTAKE_BOOK_PATH = os.path.join(DATA_DIR, 'mistake_book.json') # No longer needed
DATABASE_PATH = os.path.join(DATA_DIR, 'mistakes.db') # Path for SQLite database


# --- Custom Title Bar ---

# --- End Custom Title Bar ---


# --- Custom Clickable Label (REMOVED - Now in monster_card_widget.py) ---
# class ClickableImageLabel(QLabel): ...


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("明日方舟斗蛐蛐错题册")
        # Default geometry is now handled in _load_window_geometry fallback

        # --- Data Loading ---
        self.templates = self._load_templates_safe()
        self.monster_data: Dict[str, Monster] = self._load_monster_data_safe() # Load once, store as dict mapping template_name to Monster obj
        self.mistake_manager = MistakeBookManager(DATABASE_PATH) # Instantiate the manager
        # Load prediction resources
        self.prediction_model = prediction.load_prediction_model()
        self.id_mapping = prediction.load_id_mapping()
        # self.mistake_book_entries: List[Dict] = self.mistake_manager.load_all_mistakes() # Load entries via manager (optional preload)

        # --- State Variables ---
        self.last_roi_screenshot: np.ndarray | None = None # To store screenshot for add dialog
        # self.current_screenshot: np.ndarray | None = None # No longer store full screenshot
        # self.current_selection = QRect() # No longer need selection from image viewer
        self.recognition_roi = QRect() # Stores the user-defined Region of Interest (screen coordinates)
        self.screen_selector: ScreenSelectionWidget | None = None # Reference to the selector widget
        self.selected_monster_template_name: str | None = None # Track selected monster for manual add
        # Removed QButtonGroup as selection is handled by QTextBrowser clicks

        # --- UI Elements ---
        # Create Tab Widget
        self.tab_widget = QTabWidget()
        # Add the tab widget *below* the title bar in the container layout
        self.setCentralWidget(self.tab_widget) 

        # --- Create Main Tab ("主页") ---
        self.main_tab = QWidget()
        self.tab_widget.addTab(self.main_tab, "主页")

        # Main layout for the main tab's content (NO LONGER THE TOP LEVEL LAYOUT)
        self.layout = QVBoxLayout(self.main_tab) # Apply layout to main_tab

        # --- Buttons Layout (Now inside main tab layout) ---
        self.button_layout = QHBoxLayout()
        self.set_roi_button = QPushButton("设置识别区域")
        self.set_roi_button.clicked.connect(self.prompt_select_roi)
        self.recognize_button = QPushButton("识别") # Changed button text
        self.recognize_button.clicked.connect(self.recognize_roi) # Changed slot connection
        self.recognize_button.setEnabled(False) # Disabled until ROI is set

        # --- Prediction Button ---
        self.predict_button = QPushButton("预测")
        self.predict_button.clicked.connect(self._handle_prediction)
        self.predict_button.setEnabled(False) # Disable initially

        # --- Mistake Book Menu Button ---
        self.mistake_book_button = QPushButton("错题本")
        self.mistake_book_menu = QMenu(self)
        self.mistake_book_button.setMenu(self.mistake_book_menu)

        # Actions for the menu
        self.action_add_mistake = QAction("添加当前组合到错题本", self)
        self.action_add_mistake.triggered.connect(self._add_mistake_book_entry)
        self.action_add_mistake.setEnabled(False) # Disable initially

        self.action_query_combination = QAction("查询当前组合是否存在记录", self)
        self.action_query_combination.triggered.connect(self._query_current_combination)
        self.action_query_combination.setEnabled(False) # Disable initially

        self.action_browse_history = QAction("浏览错题本历史", self)
        self.action_browse_history.triggered.connect(self._browse_mistake_history)
        # action_browse_history is always enabled if manager is available

        self.mistake_book_menu.addAction(self.action_add_mistake)
        self.mistake_book_menu.addAction(self.action_query_combination)
        self.mistake_book_menu.addSeparator()
        self.mistake_book_menu.addAction(self.action_browse_history)

        self.button_layout.addWidget(self.set_roi_button)
        self.button_layout.addWidget(self.recognize_button)
        self.button_layout.addWidget(self.predict_button) # Add predict button next to recognize
        self.button_layout.addStretch() # Push mistake book button to the right
        self.button_layout.addWidget(self.mistake_book_button)
        self.layout.addLayout(self.button_layout) # Add button layout first

        # --- Annotated Image Display ---
        self.annotated_image_display = QLabel("识别结果图像将显示在此处")
        self.annotated_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.annotated_image_display.setMinimumHeight(120) # Reduced height
        self.annotated_image_display.setStyleSheet("border: 1px solid gray; background-color: #e0e0e0;") # Style it slightly
        # Set initial visibility based on settings
        initial_hide_image = settings.get_setting("hide_recognition_image", False)
        self.annotated_image_display.setVisible(not initial_hide_image)
        self.layout.addWidget(self.annotated_image_display) # Add it below buttons

        # --- Prediction Result Display ---
        self.prediction_result_label = QLabel("点击“预测”按钮获取结果")
        self.prediction_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Make font larger and bold
        font = QFont()
        font.setPointSize(14) # Adjust size as needed
        font.setBold(True)
        self.prediction_result_label.setFont(font)
        self.prediction_result_label.setWordWrap(True) # Allow wrapping if text is long
        self.prediction_result_label.setMinimumHeight(40) # Ensure some space for the text
        self.prediction_result_label.setStyleSheet("border: 1px dashed gray; padding: 5px; background-color: #f0f8ff;") # AliceBlue background
        self.layout.addWidget(self.prediction_result_label) # Add it below the annotated image

        # --- Manual Monster Addition Section ---

        # Button to toggle the visibility of the selection area
        self.toggle_monster_list_button = QPushButton("隐藏怪物列表 ▼")
        self.toggle_monster_list_button.setCheckable(False) # Make it a normal button
        self.toggle_monster_list_button.clicked.connect(self._toggle_monster_list)
        self.layout.addWidget(self.toggle_monster_list_button)

        # Container widget for the collapsible elements
        self.monster_selection_container = QWidget()
        self.monster_selection_container_layout = QVBoxLayout(self.monster_selection_container)
        self.monster_selection_container_layout.setContentsMargins(0, 0, 0, 0) # No extra margins
        self.monster_selection_container_layout.setSpacing(5) # Spacing between browser and controls

        # Monster Selection Area (Using QTextBrowser for auto-wrap)
        self.monster_selection_browser = QTextBrowser()
        self.monster_selection_browser.setOpenLinks(False) # Don't open external links
        self.monster_selection_browser.setOpenExternalLinks(False)
        self.monster_selection_browser.anchorClicked.connect(self._handle_monster_selection) # Connect signal
        self.monster_selection_browser.setFixedHeight(150) # Limit height
        self.monster_selection_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) # Expand horizontally
        # Ensure word wrap is enabled (usually default, but good to be explicit)
        self.monster_selection_browser.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.monster_selection_container_layout.addWidget(self.monster_selection_browser) # Add browser to container

        # Layout for Add Buttons and Selection Label
        self.add_controls_layout = QHBoxLayout()

        # Label to show current selection
        self.selected_monster_label = QLabel("当前选择: 无")
        self.selected_monster_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) # Allow label to take space
        self.add_controls_layout.addWidget(self.selected_monster_label)

        # Add Buttons (Pushed to the right)
        self.add_controls_layout.addStretch() # Push buttons to the right
        self.add_left_button = QPushButton("添加到左侧")
        self.add_right_button = QPushButton("添加到右侧")
        self.add_left_button.clicked.connect(lambda: self._add_monster_manually('left'))
        self.add_right_button.clicked.connect(lambda: self._add_monster_manually('right'))
        self.add_controls_layout.addWidget(self.add_left_button)
        self.add_controls_layout.addWidget(self.add_right_button)

        # Add controls layout to the container
        self.monster_selection_container_layout.addLayout(self.add_controls_layout)

        # Add the container widget to the main layout
        self.layout.addWidget(self.monster_selection_container)

        # Load initial state for monster list visibility from settings
        initial_monster_list_hidden = settings.get_setting("monster_list_hidden", False) # Default to shown
        self.monster_selection_container.setVisible(not initial_monster_list_hidden)
        if initial_monster_list_hidden:
            self.toggle_monster_list_button.setText("显示怪物列表 ▲")
        else:
            self.toggle_monster_list_button.setText("隐藏怪物列表 ▼")


        # --- Remove Image Viewer ---
        # self.image_viewer = ImageViewer()
        # self.image_viewer.new_selection.connect(self.handle_image_selection)
        # self.image_viewer.setMinimumHeight(300)
        # self.image_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # self.layout.addWidget(self.image_viewer, stretch=1)

        # --- Status Label ---
        self.status_label = QLabel("状态：请先点击“设置识别区域”。")
        self.layout.addWidget(self.status_label)

        # --- Tables Layout --- (Keep this part)
        self.tables_layout = QHBoxLayout()

        # --- Left Display Area (Using MonsterDisplayArea) ---
        self.left_display_widget = QWidget() # Container for label + display area
        self.left_layout = QVBoxLayout(self.left_display_widget)
        self.left_label = QLabel("左侧怪物")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Pass the full monster data dict needed by the display area
        self.left_display_area = MonsterDisplayArea('left', self.monster_data)
        self.left_layout.addWidget(self.left_label)
        self.left_layout.addWidget(self.left_display_area) # Add the new widget
        self.tables_layout.addWidget(self.left_display_widget)

        # --- Right Display Area (Using MonsterDisplayArea) ---
        self.right_display_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_display_widget)
        self.right_label = QLabel("右侧怪物")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Pass the full monster data dict
        self.right_display_area = MonsterDisplayArea('right', self.monster_data)
        self.right_layout.addWidget(self.right_label)
        self.right_layout.addWidget(self.right_display_area) # Add the new widget
        self.tables_layout.addWidget(self.right_display_widget)

        # Add the horizontal layout containing display areas to the main vertical layout
        self.layout.addLayout(self.tables_layout, stretch=1) # Give display areas stretch factor

        # Populate the monster selection list after all UI is set up
        self._populate_monster_selection_list()
        # Connect signals from the new display areas
        self.left_display_area.monsters_changed.connect(self._update_mistake_actions_state)
        self.right_display_area.monsters_changed.connect(self._update_mistake_actions_state)
        self.left_display_area.monster_clicked.connect(self._show_damage_info)
        self.right_display_area.monster_clicked.connect(self._show_damage_info)

        self._update_mistake_actions_state() # Set initial state

        # --- Create Settings Tab ("设置") ---
        self.settings_tab = SettingsTab()
        self.tab_widget.addTab(self.settings_tab, "设置")
        # Connect the signal from the settings tab to the handler in MainWindow
        self.settings_tab.setting_changed.connect(self._handle_setting_change)
        self._load_window_geometry()
        # Apply initial always on top setting after loading geometry
        self._update_always_on_top_state(settings.get_setting("always_on_top", False))

    # --- Removed _create_results_table ---

    def _update_mistake_actions_state(self):
        """Updates the enabled state of mistake book actions based on current monsters."""
        # Get monsters from the new display areas
        left_monsters = self.left_display_area.get_monsters()
        right_monsters = self.right_display_area.get_monsters()
        has_monsters = bool(left_monsters or right_monsters)

        # Enable mistake actions if there are monsters
        self.action_add_mistake.setEnabled(has_monsters)
        self.action_query_combination.setEnabled(has_monsters)

        # Enable predict button if there are monsters AND model/mapping are loaded
        can_predict = has_monsters and self.prediction_model is not None and self.id_mapping is not None
        self.predict_button.setEnabled(can_predict)

        # logger.debug(f"Actions updated: Add/Query={has_monsters}, Predict={can_predict}") # Optional Debugging

    def _load_templates_safe(self):
        """Loads templates and handles potential errors."""
        templates = load_templates(TEMPLATE_DIR)
        if not templates:
            QMessageBox.warning(self, "模板错误",
                                f"无法从以下路径加载怪物模板：\n{TEMPLATE_DIR}\n\n请确保目录存在且包含 PNG 图片。")
            return {} # Return empty dict if loading fails
        return templates

    def _load_monster_data_safe(self) -> Dict[str, Monster]:
        """
        Loads monster CSV data using data_loader, converts rows to Monster objects,
        and returns a dictionary mapping template_name to Monster object.
        Handles potential errors during loading or conversion.
        """
        raw_monster_list = load_monster_data() # Gets list of dicts with Chinese keys
        if not raw_monster_list:
             QMessageBox.warning(self, "数据错误",
                                f"无法从以下路径加载怪物数据或文件为空：\n{MONSTER_CSV_PATH}\n\n请确保文件存在且为有效的 CSV 文件，并包含必需列（如 ID, 名称）。")
             return {}

        monster_data_dict: Dict[str, Monster] = {}
        skipped_count = 0
        for row_dict in raw_monster_list:
            try:
                monster_obj = Monster.from_dict(row_dict)
                # Use template_name (e.g., 'obj_36') as the key
                if monster_obj.template_name and monster_obj.template_name != "obj_unknown":
                     monster_data_dict[monster_obj.template_name] = monster_obj
                else:
                     logger.warning(f"跳过无法生成有效 template_name 的怪物数据：{row_dict}")
                     skipped_count += 1
            except Exception as e:
                logger.error(f"转换 CSV 行到 Monster 对象时出错：{row_dict} - {e}")
                skipped_count += 1

        if skipped_count > 0:
             QMessageBox.warning(self, "数据转换警告",
                                f"加载数据时跳过了 {skipped_count} 行无效或不完整的怪物数据。请检查控制台输出和 CSV 文件。")

        if not monster_data_dict:
            QMessageBox.critical(self, "数据错误",f"未能成功加载任何有效的怪物数据。请检查 CSV 文件格式和内容。\n路径: {MONSTER_CSV_PATH}")
            return {}
        logger.info(f"成功加载并转换了 {len(monster_data_dict)} 条怪物数据。")
        return monster_data_dict


   # --- Manual Add/Select Methods ---

    def _populate_monster_selection_list(self, selected_template_name: str | None = None):
        """
        Populates the QTextBrowser with clickable monster icons using HTML.
        Highlights the icon corresponding to selected_template_name.
        """
        if not self.monster_data:
            self.monster_selection_browser.hide() # Hide if no data
            return
        else:
            self.monster_selection_browser.show()

        self.monster_selection_browser.clear() # Clear previous content
        # Use inline-block display for images within a div to allow wrapping
        # Removed style='line-height: 0;' as it might interfere with spacing
        html_content = "<div>"

        # Sort monsters by template name for consistent order
        sorted_template_names = sorted(self.monster_data.keys())

        for template_name in sorted_template_names:
            monster_info = self.monster_data[template_name]
            # Need absolute path or relative path accessible by Qt's resource system for images in QTextBrowser
            # Using absolute path is simpler here. Convert potential backslashes for HTML.
            icon_path_abs = os.path.abspath(os.path.join(TEMPLATE_DIR, f"{template_name}.png")).replace("\\", "/")
            icon_url = QUrl.fromLocalFile(icon_path_abs).toString() # Convert path to file URL
            tooltip = f"{monster_info.name} ({template_name})"

            # Determine style for the anchor tag based on selection
            if template_name == selected_template_name:
                # Selected style: blue border and light blue background
                anchor_style = "background-color: lightblue; border: 2px solid blue; display: inline-block; margin: 2px; padding: 1px;"
            else:
                # Default style: transparent border (for consistent spacing) and background
                anchor_style = "background-color: transparent; border: 2px solid transparent; display: inline-block; margin: 2px; padding: 1px;"

            # Image style (no specific border needed if anchor provides it)
            img_style = "display: block;"

            # Create an anchor tag wrapping the image
            # Apply the highlight style (border and background) to the anchor tag
            html_snippet = (
                f"<a href='{template_name}' style='{anchor_style}'>" # Use combined anchor style
                f"<img src='{icon_url}' title='{tooltip}' width='48' height='48' "
                f"style='{img_style}'>" # Apply basic image style
                f"</a>"
            )
            html_content += html_snippet

        html_content += "</div>"
        self.monster_selection_browser.setHtml(html_content)

    def _handle_monster_selection(self, url: QUrl):
        """Handles clicks on monster icons (anchors) in the QTextBrowser."""
        template_name = url.toString() # The href is the template_name
        self.selected_monster_template_name = template_name

        # Regenerate the list HTML to highlight the new selection
        self._populate_monster_selection_list(selected_template_name=template_name)

        # Safely get monster name for the label, provide default if not found
        monster_name = "未知"
        if template_name in self.monster_data:
             monster_name = self.monster_data[template_name].name
        self.selected_monster_label.setText(f"当前选择: {monster_name} ({template_name})")
        logger.debug(f"已选择: {template_name}") # Debugging output

    def _add_monster_manually(self, side: Literal['left', 'right']):
        """Adds the selected monster card to the specified side."""
        if not self.selected_monster_template_name:
            QMessageBox.information(self, "提示", "请先从上方列表中选择一个怪物。")
            return

        monster_info = self.monster_data.get(self.selected_monster_template_name)
        if not monster_info:
            QMessageBox.warning(self, "错误", f"找不到所选怪物 '{self.selected_monster_template_name}' 的数据。")
            return

        target_display_area = self.left_display_area if side == 'left' else self.right_display_area
        target_display_area.add_monster_manually(monster_info)
        # No need to call _update_mistake_actions_state here, as the signal connection handles it
        logger.debug(f"已手动添加 {monster_info.name} 到 {side} 侧。")


    # --- New Workflow Methods ---

    def prompt_select_roi(self):
        """Initiates the screen region selection process for the ROI."""
        self.status_label.setText("状态：请在屏幕上拖拽选择要持续识别的区域...")
        # Hide main window while selecting (optional, can cause issues on some systems)
        # self.hide()
        QApplication.processEvents()

        primary_screen = QApplication.primaryScreen()
        if not primary_screen:
            QMessageBox.critical(self, "屏幕错误", "无法访问主屏幕。")
            self.status_label.setText("状态：访问屏幕时出错。")
            # self.show() # Reshow if hidden
            return

        # Create and show the selection widget
        # Keep a reference to prevent garbage collection before signal is emitted
        self.screen_selector = ScreenSelectionWidget(primary_screen)
        self.screen_selector.area_selected.connect(self.handle_roi_selection)
        self.screen_selector.show()

    def handle_roi_selection(self, selected_rect: QRect):
        """Slot to receive the selected ROI from ScreenSelectionWidget."""
        # Reshow main window if it was hidden
        # self.show()

        if selected_rect.isValid() and selected_rect.width() > 0 and selected_rect.height() > 0:
            self.recognition_roi = selected_rect
            self.recognize_button.setEnabled(True) # Enable recognition button
            self.status_label.setText(f"状态：识别区域已设置: {selected_rect.x()},{selected_rect.y()} {selected_rect.width()}x{selected_rect.height()}。点击“识别”。")
            logger.info(f"ROI 设置成功: {self.recognition_roi}")
        else:
            # Selection was cancelled or invalid
            if not self.recognition_roi.isValid(): # Only reset status if no valid ROI was set before
                 self.recognize_button.setEnabled(False)
                 self.status_label.setText("状态：未设置有效识别区域。请点击“设置识别区域”。")
            else:
                 # Keep previous valid ROI and status
                 self.status_label.setText(f"状态：识别区域保持为: {self.recognition_roi.x()},{self.recognition_roi.y()} {self.recognition_roi.width()}x{self.recognition_roi.height()}。")

        # Clean up the selector widget reference
        self.screen_selector = None


    def recognize_roi(self):
        """Captures the defined ROI and performs recognition."""
        if not self.recognition_roi.isValid() or self.recognition_roi.width() <= 0 or self.recognition_roi.height() <= 0:
            QMessageBox.warning(self, "错误", "请先点击“设置识别区域”并选择一个有效区域。")
            return

        self.status_label.setText(f"状态：正在捕获区域 {self.recognition_roi.x()},{self.recognition_roi.y()} 并识别...")
        QApplication.processEvents()

        # --- Capture the ROI using mss ---
        roi_screenshot = None
        monitor = {
            "top": self.recognition_roi.y(),
            "left": self.recognition_roi.x(),
            "width": self.recognition_roi.width(),
            "height": self.recognition_roi.height(),
        }
        try:
            with mss.mss() as sct:
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                # Convert BGRA to BGR
                roi_screenshot = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Save the captured ROI for debugging
                try:
                    save_path = "captured_roi_for_recognition.png"
                    cv2.imwrite(save_path, roi_screenshot)
                    logger.debug(f"已将捕获的 ROI 区域保存至 {save_path}")
                except Exception as save_e:
                    logger.warning(f"保存捕获的 ROI 图像失败: {save_e}")

            # Store the screenshot temporarily for the add mistake dialog
            self.last_roi_screenshot = roi_screenshot.copy() # Store a copy

        except Exception as e:
            QMessageBox.critical(self, "捕获错误", f"捕获指定区域时出错: {e}")
            self.status_label.setText("状态：捕获识别区域失败。")
            return

        if roi_screenshot is None or roi_screenshot.size == 0:
             QMessageBox.warning(self, "捕获错误", "捕获的识别区域图像无效或为空。")
             self.status_label.setText("状态：捕获的识别区域无效。")
             return

        # --- Perform Recognition on ROI Screenshot ---
        # No need to convert to gray here if recognize_monsters handles it
        # gray_roi_screenshot = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)

        # Check if templates are loaded
        if not self.templates:
             QMessageBox.critical(self, "错误", "模板未加载，无法进行识别。")
             self.status_label.setText("状态：识别失败（模板未加载）。")
             return

        # Call the ORB-based recognize_monsters
        # Assuming it still returns counts, but we'll ignore them for display per type
        # It returns a tuple: (image_with_boxes, results_dict)
        # We only need the results_dict here. Ignore the image.
        # Call recognize_monsters and get both the annotated image and the results dict
        annotated_image, recognition_results = recognize_monsters( # Unpack the tuple
            roi_screenshot,       # Pass the captured ROI
            self.templates        # Pass the loaded template data
        )
        # The actual type of recognition_results is Dict[Literal['left', 'right'], Dict[str, Optional[int]]]

        # --- Display Annotated Image (Conditional) ---
        hide_image = settings.get_setting("hide_recognition_image", False)
        if hide_image:
            self.annotated_image_display.setVisible(False) # Directly hide the widget
            logger.debug("识别结果图像根据设置已隐藏")
        else: # If not hiding the image
            self.annotated_image_display.setVisible(True) # Ensure the widget is visible
            if annotated_image is not None and annotated_image.shape[0] > 0 and annotated_image.shape[1] > 0:
                # Display the annotated image
                try:
                    height, width, channel = annotated_image.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
                    pixmap = QPixmap.fromImage(q_image)
                    # Scale pixmap to fit the label while keeping aspect ratio
                    self.annotated_image_display.setPixmap(pixmap.scaled(
                        self.annotated_image_display.size(), # Scale to the label's current size
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    self.annotated_image_display.setText("") # Clear any previous text
                except Exception as img_disp_e:
                    logger.error(f"在UI中显示标注图像时出错: {img_disp_e}")
                    self.annotated_image_display.setText("无法显示识别结果图像") # Show error text
                    self.annotated_image_display.setPixmap(QPixmap()) # Clear pixmap on error
            else:
                # No valid image returned, but widget should be visible
                self.annotated_image_display.setText("识别函数未返回有效图像")
                self.annotated_image_display.setPixmap(QPixmap()) # Clear any previous pixmap

        # --- Display Results (Using MonsterDisplayArea) ---
        # Populate displays using the new widgets
        self.left_display_area.populate_display(
            list(recognition_results.get('left', {}).keys()),
            recognition_results
        )
        self.right_display_area.populate_display(
            list(recognition_results.get('right', {}).keys()),
            recognition_results
        )

        # No need to call _update_mistake_actions_state here, signals handle it

        # Get current monster types *after* populating for the check
        left_monsters_after = self.left_display_area.get_monsters()
        right_monsters_after = self.right_display_area.get_monsters()
        left_types_after = [m.template_name for m, c in left_monsters_after]
        right_types_after = [m.template_name for m, c in right_monsters_after]

        # Check for mistake book matches (for automatic notification)
        self._check_for_mistake_book_matches(left_types_after, right_types_after)

        # Update status to indicate clicking the image
        # Status message will be updated by _check_for_mistake_book_matches if needed
        # self.status_label.setText(f"状态：识别完成。左侧: {len(left_monster_types)} 种, 右侧: {len(right_monster_types)} 种。点击怪物图片查看双向伤害详情。")


    # --- Methods Removed (Now handled by MonsterDisplayArea and MonsterCardWidget) ---
    # _clear_layout(...) is effectively handled by MonsterDisplayArea.clear_display()
    # _create_monster_card(...) is now MonsterCardWidget.__init__()
    # _update_card_count(...) is now MonsterCardWidget._update_internal_count()
    # _get_monsters_from_layout(...) is now MonsterDisplayArea.get_monsters()
    # _populate_display(...) is now MonsterDisplayArea.populate_display()
    # --- End Removed Methods ---

    def _show_damage_info(self, clicked_monster: Monster, clicked_side: Literal['left', 'right']):
        """
        Shows the DamageInfoWindow for the clicked monster, calculating bidirectional damage.
        Triggered by clicking the monster's image label.
        """
        logger.info(f"显示 {clicked_monster.name} ({clicked_side}侧) 的双向伤害信息...")

        # Identify the source and target lists based on which side was clicked
        if clicked_side == 'left':
            left_monsters_for_dialog = [clicked_monster]
            # Get monsters from the right display area
            right_monsters_with_counts = self.right_display_area.get_monsters()
            right_monsters_for_dialog = [monster for monster, count in right_monsters_with_counts]
            left_side_name = "左侧选中"
            right_side_name = "右侧全体"
            if not right_monsters_for_dialog:
                 QMessageBox.information(self, "提示", "右侧没有怪物可供计算伤害。")
                 return
        else: # Clicked on the right side
            # Get monsters from the left display area
            left_monsters_with_counts = self.left_display_area.get_monsters()
            left_monsters_for_dialog = [monster for monster, count in left_monsters_with_counts]
            right_monsters_for_dialog = [clicked_monster]
            left_side_name = "左侧全体"
            right_side_name = "右侧选中"
            if not left_monsters_for_dialog:
                 QMessageBox.information(self, "提示", "左侧没有怪物可供计算伤害。")
                 return


        # Create and show the dialog, passing the lists of Monster objects
        dialog = DamageInfoWindow(left_monsters_for_dialog, right_monsters_for_dialog, left_side_name, right_side_name, self)
        dialog.exec() # Use exec() for modal behavior

    # --- Mistake Book Menu Actions ---

    def _add_mistake_book_entry(self):
        """Action: Opens the dialog to add the current combination to the mistake book."""
        # Get current monsters and their counts from the display areas
        left_monsters_with_counts = self.left_display_area.get_monsters()
        right_monsters_with_counts = self.right_display_area.get_monsters()

        if not left_monsters_with_counts and not right_monsters_with_counts:
            QMessageBox.information(self, "提示", "左右两侧均无怪物，无法添加错题记录。请先进行识别或手动添加怪物。")
            return

        # Pass the last screenshot if available (will be None if monsters were added manually)
        screenshot_to_show = self.last_roi_screenshot

        # --- Prepare data for MistakeBookEntryDialog (including counts) ---
        # The dialog needs to be updated to accept this structure
        # For now, we'll pass the lists of tuples.
        # TODO: Update MistakeBookEntryDialog to handle counts.

        # --- Use MistakeBookEntryDialog ---
        # Pass existing_entry=None explicitly and self as parent
        # Pass the lists containing tuples (Monster, count)
        dialog = MistakeBookEntryDialog(left_monsters_with_counts, right_monsters_with_counts, self.monster_data, screenshot_to_show, existing_entry=None, parent=self)
        if dialog.exec():
            # Assuming get_entry_data() is updated or handles the input correctly
            new_entry_data = dialog.get_entry_data() # Dialog returns the data dict
            if new_entry_data:
                # Insert using the manager
                new_id = self.mistake_manager.insert_mistake(new_entry_data)
                if new_id is not None:
                     # No need to reload self.mistake_book_entries unless actively used elsewhere
                     QMessageBox.information(self, "成功", "错题记录已添加。")
                # else: Error message shown by manager's insert_mistake

        # Clear the temporary screenshot regardless of whether save was successful
        self.last_roi_screenshot = None # Clear screenshot after dialog attempt

    def _query_current_combination(self):
        """Action: Checks if the currently displayed monster combination exists in the mistake book."""
        left_monsters_with_counts = self.left_display_area.get_monsters()
        right_monsters_with_counts = self.right_display_area.get_monsters()

        if not left_monsters_with_counts and not right_monsters_with_counts:
            QMessageBox.information(self, "查询当前组合", "左右两侧均无怪物可供查询。")
            return

        # Extract only the template_name
        left_types = [monster.template_name for monster, count in left_monsters_with_counts]
        right_types = [monster.template_name for monster, count in right_monsters_with_counts]

        matching_ids = self.mistake_manager.find_matching_mistakes(left_types, right_types)

        if matching_ids:
            match_count = len(matching_ids)
            reply = QMessageBox.question(self, "查询结果",
                                         f"找到 {match_count} 条与当前组合匹配的错题记录。\n\n"
                                         "是否立即跳转到历史记录查看？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.Yes) # Default to Yes

            if reply == QMessageBox.StandardButton.Yes:
                # 调用浏览历史的方法，并传递需要高亮的 ID
                # 注意: _browse_mistake_history 需要修改以接受 highlight_ids 参数
                self._browse_mistake_history(highlight_ids=matching_ids)
        else:
            QMessageBox.information(self, "查询当前组合", "未找到与当前左右怪物组合匹配的错题记录。")


    def _browse_mistake_history(self, highlight_ids: Optional[List[int]] = None):
        """
        Action: Opens the dialog to browse all mistake book entries.
        Optionally highlights specific entries based on provided IDs.
        """
        # Load fresh data from DB via manager when querying
        current_entries = self.mistake_manager.load_all_mistakes()
        if not current_entries:
            QMessageBox.information(self, "浏览错题本", "错题本为空。")
            return

        # Use MistakeBookQueryDialog to show all entries
        # Pass highlight_ids to the dialog constructor
        # Note: MistakeBookQueryDialog needs to be updated to accept and use this parameter
        dialog = MistakeBookQueryDialog(
            current_entries,
            self.monster_data,
            self.mistake_manager,
            highlight_ids=highlight_ids, # Pass the IDs here
            parent=self
        )
        dialog.exec()


    def _check_for_mistake_book_matches(self, current_left_types: List[str], current_right_types: List[str]):
        """Checks for mistake book matches using MistakeBookManager (for automatic notification)."""
        base_status = f"状态：识别完成。左: {len(current_left_types)} 种, 右: {len(current_right_types)} 种。点击图片查看详情。"

        # Use the manager to find matches
        matching_ids = self.mistake_manager.find_matching_mistakes(current_left_types, current_right_types)

        if matching_ids:
            match_count = len(matching_ids)
            # Update the status label text.
            self.status_label.setText(
                f"{base_status} 发现 {match_count} 条匹配的错题记录！"
            )
            # Ask the user if they want to view the matches
            reply = QMessageBox.question(self, "错题提示",
                                         f"发现 {match_count} 条与当前识别结果匹配的错题记录。\n\n是否立即查看？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            logger.debug(f"发现匹配的错题记录 ID: {matching_ids}")
            if reply == QMessageBox.StandardButton.Yes:
                # 传递 matching_ids 以便高亮显示
                self._browse_mistake_history(highlight_ids=matching_ids) # Open the history browser if user clicks Yes
        else:
            # Default status if no matches found
             self.status_label.setText(base_status)

        # Note: This method is now only for the automatic notification after recognition.
        # The manual check is handled by _query_current_combination.

    # --- Prediction Handling ---
    def _handle_prediction(self):
        """Handles the click of the 'Predict' button."""
        logger.info("Handling prediction request...")
        self.prediction_result_label.setText("正在预测...")
        self.prediction_result_label.setStyleSheet("border: 1px dashed gray; padding: 5px; background-color: #f0f8ff; color: black;") # Reset color
        QApplication.processEvents() # Update UI

        if not self.prediction_model or not self.id_mapping:
            QMessageBox.critical(self, "错误", "预测模型或ID映射未加载，无法预测。")
            self.prediction_result_label.setText("预测失败：资源未加载")
            return

        # 1. Get monster lists from display areas
        left_monsters_with_counts = self.left_display_area.get_monsters()
        right_monsters_with_counts = self.right_display_area.get_monsters()

        if not left_monsters_with_counts and not right_monsters_with_counts:
            QMessageBox.information(self, "提示", "左右两侧均无怪物，无法进行预测。")
            self.prediction_result_label.setText("无怪物可预测")
            return

        # 2. Extract UI IDs (numeric part from template_name)
        left_ui_ids = []
        right_ui_ids = []
        id_extract_pattern = re.compile(r'obj_(\d+)') # Regex to get number from 'obj_XX'

        for monster, count in left_monsters_with_counts:
            match = id_extract_pattern.match(monster.template_name)
            if match:
                # Add the ID 'count' times if count is known, otherwise once
                num_to_add = count if count is not None and count > 0 else 1
                left_ui_ids.extend([match.group(1)] * num_to_add)
            else:
                logger.warning(f"无法从左侧怪物模板名称提取ID: {monster.template_name}")

        for monster, count in right_monsters_with_counts:
            match = id_extract_pattern.match(monster.template_name)
            if match:
                num_to_add = count if count is not None and count > 0 else 1
                right_ui_ids.extend([match.group(1)] * num_to_add)
            else:
                logger.warning(f"无法从右侧怪物模板名称提取ID: {monster.template_name}")

        logger.debug(f"Extracted Left UI IDs for prediction: {left_ui_ids}")
        logger.debug(f"Extracted Right UI IDs for prediction: {right_ui_ids}")

        # 3. Call prediction function
        # Pass the extracted lists of UI ID strings
        prediction_result = prediction.predict_outcome(left_ui_ids, right_ui_ids)

        # 4. Display result
        if prediction_result is not None:
            right_win_prob = prediction_result
            left_win_prob = 1.0 - right_win_prob
            result_text = f"预测结果: 左方胜率 {left_win_prob:.2%} | 右方胜率 {right_win_prob:.2%}"
            self.prediction_result_label.setText(result_text)

            # Set color based on higher probability
            if left_win_prob > right_win_prob:
                # Use a shade of red/orange for left win
                self.prediction_result_label.setStyleSheet("border: 1px solid gray; padding: 5px; background-color: #ffe4e1; color: #b22222;") # MistyRose background, Firebrick text
            elif right_win_prob > left_win_prob:
                # Use a shade of blue for right win
                self.prediction_result_label.setStyleSheet("border: 1px solid gray; padding: 5px; background-color: #e0ffff; color: #1e90ff;") # LightCyan background, DodgerBlue text
            else:
                # Neutral color for a tie or exact 50%
                self.prediction_result_label.setStyleSheet("border: 1px dashed gray; padding: 5px; background-color: #f5f5f5; color: black;") # WhiteSmoke background
            logger.info(f"Prediction displayed: {result_text}")
        else:
            self.prediction_result_label.setText("预测失败，请查看日志了解详情。")
            self.prediction_result_label.setStyleSheet("border: 1px solid red; padding: 5px; background-color: #fff0f0; color: red;") # Error style
            logger.error("Prediction function returned None.")

    def _toggle_monster_list(self):
        """Toggles the visibility of the monster selection container."""
        is_visible = self.monster_selection_container.isVisible()
        new_visibility = not is_visible
        self.monster_selection_container.setVisible(new_visibility)

        # Update button text
        if new_visibility: # Now visible
            self.toggle_monster_list_button.setText("隐藏怪物列表 ▼")
        else: # Now hidden
            self.toggle_monster_list_button.setText("显示怪物列表 ▲")

        # Save the new state (True if hidden, False if visible)
        settings.update_setting("monster_list_hidden", not new_visibility)
        logger.debug(f"怪物列表隐藏状态已保存: {not new_visibility}")


    @pyqtSlot(str, bool) # Decorate as a slot
    def _handle_setting_change(self, setting_key: str, new_value: bool):
        """Handles signals emitted when a setting is changed in the SettingsTab."""
        if setting_key == "hide_recognition_image":
            # Update visibility immediately based on the new setting value
            # If new_value is True (meaning hide), set visible to False.
            # If new_value is False (meaning show), set visible to True.
            self.annotated_image_display.setVisible(not new_value)
            logger.debug(f"识别结果图像可见性已根据设置更新为: {not new_value}")
        elif setting_key == "simplify_monster_card":
            # Re-render cards if the simplification setting changed
            # This requires re-populating the display layouts
            logger.debug("简化卡片设置已更改，正在重新渲染卡片...")
            # We need the last recognition results to re-populate correctly.
            # This is complex as results aren't stored persistently.
            # A simpler approach for now: Clear and ask user to re-recognize.
            # Or, ideally, store the last results and call _populate_display again.
            # For now, just log it. A full re-render requires more state management.
            QMessageBox.information(self, "设置更改", "卡片简化设置已更改。请重新识别以应用更改。")
        elif setting_key == "always_on_top":
             # Update the window's always-on-top state immediately
             self._update_always_on_top_state(new_value)

    def _update_always_on_top_state(self, enabled: bool):
        """Updates the window's always-on-top state."""
        current_flags = self.windowFlags()
        if enabled:
            # Add the stay-on-top hint
            self.setWindowFlags(current_flags | Qt.WindowType.WindowStaysOnTopHint)
            logger.debug("窗口置顶已启用")
        else:
            # Remove the stay-on-top hint
            self.setWindowFlags(current_flags & ~Qt.WindowType.WindowStaysOnTopHint)
            logger.debug("窗口置顶已禁用")
        # Re-show the window to apply flag changes
        # This might cause a flicker, but is often necessary
        self.show()

    # Override setWindowTitle to update custom title bar
    def setWindowTitle(self, title: str) -> None:
        super().setWindowTitle(title)
        # Update the custom title bar if it exists
        if hasattr(self, 'title_bar'):
            self.title_bar.update_title()

    # --- Rest of the MainWindow methods ---
    # ... (methods like _update_mistake_actions_state, _load_templates_safe, etc.)

    def _load_window_geometry(self):
        """Loads window geometry from settings and applies it."""
        geometry_data = settings.get_setting("window_geometry")
        if isinstance(geometry_data, list) and len(geometry_data) == 4:
            try:
                x, y, width, height = map(int, geometry_data)
                 # Basic validation to prevent unreasonable sizes/positions
                if width > 0 and height > 0:
                    self.setGeometry(x, y, width, height)
                    logger.debug(f"已加载窗口几何信息: {x},{y} {width}x{height}")
                    return # Successfully loaded and applied
                else:
                    logger.warning(f"加载的窗口几何信息无效 (宽度或高度 <= 0): {geometry_data}")
            except (ValueError, TypeError) as e:
                logger.warning(f"加载的窗口几何信息格式错误: {geometry_data} - {e}")
        else:
            logger.debug("未找到有效的窗口几何信息设置，使用默认值。")

        # Fallback to default if loading fails or no setting exists
        self.setGeometry(100, 100, 1000, 700) # Default geometry

    def closeEvent(self, event: QCloseEvent):
        """Saves window geometry when the window is closed."""
        geometry = self.geometry()
        geometry_data = [geometry.x(), geometry.y(), geometry.width(), geometry.height()]
        settings.update_setting("window_geometry", geometry_data)
        logger.debug(f"已保存窗口几何信息: {geometry_data}")
        event.accept() # Ensure the event is accepted and the window closes


# --- Mistake Book Dialogs --- (REMOVED - Now in mistake_book_manager.py)


if __name__ == '__main__':
    # This allows running the window directly for basic UI testing.
    # It now relies on the actual data files (monster.csv, template images)
    # being present in the correct locations relative to the project root.
    # The internal error handling (_load_*_safe) should show warnings/errors if files are missing.
    app = QApplication(sys.argv)

    # No longer create dummy files here, rely on actual data or error messages.
    # Configure basic logging for the __main__ block
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("启动主窗口...")
    logger.info(f"预期数据文件路径: {MONSTER_CSV_PATH}")
    logger.info(f"预期模板目录路径: {TEMPLATE_DIR}")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
