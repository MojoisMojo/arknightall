import logging  # Import logging module
import os
import sys
from typing import Dict, Literal, List, Optional, Tuple

import cv2
import mss  # Keep mss import here as it's used directly for capture
import numpy as np
import json # Added for mapping
import re # Added for extracting ID from template name
from PyQt6.QtCore import Qt, QRect, QUrl, pyqtSignal, QTimer  # Added QSize, QUrl for anchor clicks, pyqtSignal, QTimer
# Added QAction, QIntValidator
from PyQt6.QtGui import QPixmap, QTextOption, QMouseEvent, QImage, QAction, QFont, QIntValidator # Added QFont, QIntValidator
# import sqlite3 # No longer needed directly here
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QScrollArea, QSizePolicy, QTextBrowser, QMenu, QLineEdit # Added QMenu, QLineEdit
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

# No longer need ImageViewer
# from src.ui.image_viewer import ImageViewer

# Define paths relative to this file's potential execution context
# This might need adjustment depending on how the app is run/packaged
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TEMPLATE_DIR = os.path.join(DATA_DIR, 'image')
MONSTER_CSV_PATH = os.path.join(DATA_DIR, 'monster.csv')
# MISTAKE_BOOK_PATH = os.path.join(DATA_DIR, 'mistake_book.json') # No longer needed
DATABASE_PATH = os.path.join(DATA_DIR, 'mistakes.db') # Path for SQLite database


# --- Custom Clickable Label ---
class ClickableImageLabel(QLabel):
    """A QLabel that emits a clicked signal when clicked, passing monster_info and side."""
    # Signal emitting the Monster object and the side ('left' or 'right')
    clicked = pyqtSignal(Monster, str)

    def __init__(self, monster_info: Monster, side: Literal['left', 'right'], parent=None):
        super().__init__(parent)
        self.monster_info = monster_info
        self.side = side
        self.setCursor(Qt.CursorShape.PointingHandCursor) # Indicate clickable

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Emit the signal with the stored monster info and side
            self.clicked.emit(self.monster_info, self.side)
        # Call the base class implementation to handle the event further if needed
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("明日方舟斗蛐蛐错题册")
        # Increase default window size
        self.setGeometry(100, 100, 1000, 700) # x, y, width, height

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
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Main layout: Buttons, ImageViewer, Status, Tables
        self.layout = QVBoxLayout(self.central_widget)

        # --- Buttons Layout ---
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

        # Monster Selection Area (Using QTextBrowser for auto-wrap)
        self.monster_selection_browser = QTextBrowser()
        self.monster_selection_browser.setOpenLinks(False) # Don't open external links
        self.monster_selection_browser.setOpenExternalLinks(False)
        self.monster_selection_browser.anchorClicked.connect(self._handle_monster_selection) # Connect signal
        self.monster_selection_browser.setFixedHeight(150) # Limit height
        self.monster_selection_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) # Expand horizontally
        # Ensure word wrap is enabled (usually default, but good to be explicit)
        self.monster_selection_browser.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.layout.addWidget(self.monster_selection_browser) # Add browser directly

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

        self.layout.addLayout(self.add_controls_layout) # Add controls layout

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

        # --- Left Display Area (Scrollable) ---
        self.left_scroll_area = QScrollArea()
        self.left_scroll_area.setWidgetResizable(True)
        self.left_container_widget = QWidget() # Container for the layout inside scroll area
        self.left_display_layout = QVBoxLayout(self.left_container_widget) # Layout for monster cards
        self.left_display_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align cards to the top
        self.left_scroll_area.setWidget(self.left_container_widget)

        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_label = QLabel("左侧怪物")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(self.left_label)
        self.left_layout.addWidget(self.left_scroll_area) # Add scroll area instead of table
        self.tables_layout.addWidget(self.left_widget)

        # --- Right Display Area (Scrollable) ---
        self.right_scroll_area = QScrollArea()
        self.right_scroll_area.setWidgetResizable(True)
        self.right_container_widget = QWidget()
        self.right_display_layout = QVBoxLayout(self.right_container_widget)
        self.right_display_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.right_scroll_area.setWidget(self.right_container_widget)

        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_label = QLabel("右侧怪物")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(self.right_label)
        self.right_layout.addWidget(self.right_scroll_area) # Add scroll area instead of table
        self.tables_layout.addWidget(self.right_widget)

        # Add the horizontal layout containing display areas to the main vertical layout
        self.layout.addLayout(self.tables_layout, stretch=1) # Give display areas stretch factor

        # Populate the monster selection list after all UI is set up
        self._populate_monster_selection_list()
        self._update_mistake_actions_state() # Set initial state

    # --- Removed _create_results_table ---

    def _update_mistake_actions_state(self):
        """Updates the enabled state of mistake book actions based on current monsters."""
        # Check layouts and update actions.
        left_monsters = self._get_monsters_from_layout(self.left_display_layout)
        right_monsters = self._get_monsters_from_layout(self.right_display_layout)
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

        target_layout = self.left_display_layout if side == 'left' else self.right_display_layout
        # Pass the side when creating the card
        monster_card = self._create_monster_card(monster_info, side)
        target_layout.addWidget(monster_card)
        self._update_mistake_actions_state() # Update state after manual add
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

        # --- Display Annotated Image ---
        if annotated_image is not None and annotated_image.shape[0] > 0 and annotated_image.shape[1] > 0:
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
            except Exception as img_disp_e:
                logger.error(f"在UI中显示标注图像时出错: {img_disp_e}")
                self.annotated_image_display.setText("无法显示识别结果图像")
        else:
            self.annotated_image_display.setText("识别函数未返回有效图像")


        # --- Display Results (Monster Cards) ---
        self._clear_layout(self.left_display_layout) # Clear previous cards
        self._clear_layout(self.right_display_layout)

        # Get unique monster template names for each side
        left_monster_types = list(recognition_results.get('left', {}).keys())
        right_monster_types = list(recognition_results.get('right', {}).keys())

        if not left_monster_types and not right_monster_types:
            self.status_label.setText("状态：在指定区域未识别到任何怪物。")
            # Keep the annotated image displayed even if no monsters identified
            # No return here, proceed to enable actions etc.

        # Populate displays, passing the side identifier and the full results dict for counts
        self._populate_display(self.left_display_layout, left_monster_types, 'left', recognition_results)
        self._populate_display(self.right_display_layout, right_monster_types, 'right', recognition_results)

        # Update mistake book actions state based on current monsters
        self._update_mistake_actions_state()

        # Check for mistake book matches (for automatic notification)
        self._check_for_mistake_book_matches(left_monster_types, right_monster_types)

        # Update status to indicate clicking the image
        # Status message will be updated by _check_for_mistake_book_matches if needed
        # self.status_label.setText(f"状态：识别完成。左侧: {len(left_monster_types)} 种, 右侧: {len(right_monster_types)} 种。点击怪物图片查看双向伤害详情。")


    def _clear_layout(self, layout):
        """Removes all widgets from a layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _create_monster_card(self, monster_info: Monster, side: Literal['left', 'right'], count: Optional[int] = None) -> QWidget:
        """
        Creates a widget (card) displaying a monster's image, stats,
        a remove button, and a details button. Stores the monster object and recognized count.

        Args:
            monster_info: The Monster object for this card.
            side: Which side ('left' or 'right') the card is on.
        """
        card = QWidget()
        # Store the monster data and recognized count directly on the widget object
        card.monster_data = monster_info
        card.side = side # Store the side
        card.recognized_count = count # Store the count from OCR

        # Add a style sheet for subtle bordering and hover effect
        card.setStyleSheet("""
            QWidget {
                border: 1px solid lightgray;
                border-radius: 3px;
                margin-bottom: 2px;
                background-color: white; /* Default background */
            }
            QWidget:hover {
                background-color: #f0f0f0; /* Light gray on hover */
            }
        """)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(5, 5, 5, 5) # Add some padding
        card_layout.setSpacing(5)

        # --- Layout for Image and Count Input ---
        image_and_count_layout = QVBoxLayout()
        image_and_count_layout.setContentsMargins(0, 0, 0, 0)
        image_and_count_layout.setSpacing(2) # Small spacing between image and input

        # --- Clickable Image Label ---
        # Use the custom ClickableImageLabel, passing monster_info and side
        image_label = ClickableImageLabel(monster_info, side)
        image_label.setFixedSize(64, 64) # Adjust size as needed
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("border: 1px solid gray; background-color: #f8f8f8;") # Optional border and slight background
        image_label.setToolTip("点击查看双向伤害详情") # Add tooltip

        image_path = os.path.join(TEMPLATE_DIR, f"{monster_info.template_name}.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            image_label.setText("无图")
            logger.warning(f"无法加载图片 {image_path}")
        else:
            # Scale pixmap while preserving aspect ratio
            image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Connect the clicked signal from the label to the handler function
        image_label.clicked.connect(self._show_damage_info)

        image_and_count_layout.addWidget(image_label) # Add image to the vertical layout

        # --- Count Input ---
        count_input = QLineEdit()
        count_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_input.setFixedWidth(image_label.width()) # Match image width
        count_input.setToolTip("怪物数量 (可编辑)")
        # Set initial value, default to 1 if count is None or invalid
        initial_count_str = "1"
        if count is not None:
            try:
                if int(count) > 0:
                    initial_count_str = str(count)
            except (ValueError, TypeError):
                pass # Keep default "1" if count is not a valid positive integer
        count_input.setText(initial_count_str)
        # Add validator for integers between 1 and 99 (adjust range if needed)
        count_input.setValidator(QIntValidator(1, 99, self))
        image_and_count_layout.addWidget(count_input) # Add input below image

        # Store reference to the input widget on the card for later access
        card.count_input = count_input
        # Connect signal to update the stored count when editing is finished
        count_input.editingFinished.connect(lambda: self._update_card_count(card))

        # Add the vertical layout (image + count) to the main horizontal layout
        card_layout.addLayout(image_and_count_layout)

        # --- Stats Labels ---
        stats_widget = QWidget()
        # Remove border from inner stats widget
        stats_widget.setStyleSheet("border: none; background-color: transparent;")
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0,0,0,0) # No internal margins
        stats_layout.setSpacing(1) # Reduce spacing between labels further

        # Helper to format optional values
        def format_value(value, default='-'):
            return str(value) if value is not None and str(value).strip() != '' else default

        # Add labels for each attribute
        # Use a slightly smaller font size for stats
        font_size = 12 # Reduced font size for stats
        name_font_size = 14 # Keep name slightly larger
        # Removed count_str from the label display
        name_label = QLabel(f"<font style='font-size: {name_font_size}px;'><b>{monster_info.name}</b> ({monster_info.template_name})</font>")
        stats_layout.addWidget(name_label)

        # Determine attack color based on damage type
        attack_color = "red" if monster_info.is_magic_damage else "blue" # Red for magic, Blue for physical
        attack_text = f"攻击: {format_value(monster_info.attack)}"
        attack_label = QLabel(f"<font color='{attack_color}' style='font-size: {font_size}px;'>{attack_text}</font>")
        stats_layout.addWidget(attack_label)

        # Add labels for other attributes
        stats_layout.addWidget(QLabel(f"<font color='green' style='font-size: {font_size}px;'>生命: {format_value(monster_info.health)}</font>"))
        stats_layout.addWidget(QLabel(f"<font color='saddlebrown' style='font-size: {font_size}px;'>防御: {format_value(monster_info.defense)}</font>"))
        stats_layout.addWidget(QLabel(f"<font color='darkviolet' style='font-size: {font_size}px;'>法抗: {format_value(monster_info.resistance)}</font>"))
        stats_layout.addWidget(QLabel(f"<font color='orange' style='font-size: {font_size}px;'>攻速: {format_value(monster_info.attack_interval)}</font>"))

        # --- Add Attack Range ---
        attack_range_value = monster_info.attack_range # Assuming attribute name is 'attack_range'
        attack_range_text = "近战" # Default to Melee
        try:
            # Check if it's a valid number > 0
            if attack_range_value is not None and str(attack_range_value).strip() and float(attack_range_value) > 0:
                attack_range_text = format_value(attack_range_value)
        except (ValueError, TypeError):
            # If conversion fails or it's not a number, keep "近战"
            pass
        stats_layout.addWidget(QLabel(f"<font color='darkcyan' style='font-size: {font_size}px;'>攻击范围: {attack_range_text}</font>"))
        # --- End Attack Range ---

        # stats_layout.addWidget(QLabel(f"<font color='teal' style='font-size: {font_size}px;'>移速: {format_value(monster_info.move_speed)}</font>")) # Maybe hide move speed

        ability_text = format_value(monster_info.special_ability, '无')
        ability_label = QLabel(f"<font style='font-size: {font_size}px;'>特殊: {ability_text}</font>")
        ability_label.setWordWrap(True) # Allow wrapping for long abilities
        stats_layout.addWidget(ability_label)

        card_layout.addWidget(stats_widget, stretch=1) # Allow stats to take remaining space

        # --- Buttons Layout (Vertical) ---
        button_vlayout = QVBoxLayout()
        button_vlayout.setSpacing(3)
        button_vlayout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align buttons to top

        # --- Remove Button (Keep this) ---
        remove_button = QPushButton("X")
        remove_button.setToolTip("移除此卡片")
        remove_button.setFixedSize(20, 20) # Small button
        remove_button.setStyleSheet("QPushButton { border: none; background-color: #FFDDDD; border-radius: 10px; font-weight: bold; color: red; } QPushButton:hover { background-color: #FFBBBB; }")
        # Connect clicked signal to delete the parent card widget
        # Need to find the card in the layout and remove it properly
        remove_button.clicked.connect(card.deleteLater)
        # Use QTimer to update state *after* the widget is likely removed from layout
        # This ensures the layout count is updated before checking enablement state
        remove_button.clicked.connect(lambda: QTimer.singleShot(10, self._update_mistake_actions_state)) # Added small delay
        button_vlayout.addWidget(remove_button)

        card_layout.addLayout(button_vlayout) # Add vertical button layout to the main horizontal layout


        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) # Expand horizontally, fixed height

        return card

    def _update_card_count(self, card: QWidget):
        """Updates the recognized_count on the card widget based on QLineEdit input."""
        if hasattr(card, 'count_input') and hasattr(card, 'monster_data'):
            text = card.count_input.text()
            old_count = card.recognized_count # Store old value for logging/reset
            try:
                new_count = int(text)
                if new_count > 0:
                    card.recognized_count = new_count
                    logger.debug(f"怪物 {card.monster_data.name} ({card.side}) 数量更新为: {new_count}")
                    # Update states that depend on monster counts
                    self._update_mistake_actions_state()
                    # If prediction button state depends directly on counts > 0, update it too
                    # self._update_prediction_button_state() # Example
                else:
                    # Handle invalid input (e.g., 0 or negative) - reset to 1
                    reset_count = 1
                    card.count_input.setText(str(reset_count))
                    card.recognized_count = reset_count
                    logger.warning(f"输入无效 '{text}' (<=0) for {card.monster_data.name}. 数量重置为 {reset_count}.")
                    self._update_mistake_actions_state()
            except ValueError:
                # Handle non-integer input - reset to previous valid count or 1
                reset_count = old_count if old_count is not None and old_count > 0 else 1
                card.count_input.setText(str(reset_count))
                # Keep card.recognized_count as the old_count unless it was None/invalid
                if card.recognized_count is None or card.recognized_count <= 0:
                    card.recognized_count = 1
                logger.warning(f"输入无效 '{text}' (非整数) for {card.monster_data.name}. 数量恢复为 {card.recognized_count}.")
                # No state update needed if count didn't actually change to a valid value

    def _get_monsters_from_layout(self, layout: QVBoxLayout) -> List[Tuple[Monster, Optional[int]]]:
        """
        Retrieves Monster objects and their recognized counts from card widgets
        within a given layout.
        """
        monsters_with_counts = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget():
                card_widget = item.widget()
                # Check if the widget has the 'monster_data' attribute we set
                if hasattr(card_widget, 'monster_data') and isinstance(card_widget.monster_data, Monster):
                    # Get the count stored on the card, default to None if not present
                    count = getattr(card_widget, 'recognized_count', None)
                    monsters_with_counts.append((card_widget.monster_data, count))
        return monsters_with_counts

    def _show_damage_info(self, clicked_monster: Monster, clicked_side: Literal['left', 'right']):
        """
        Shows the DamageInfoWindow for the clicked monster, calculating bidirectional damage.
        Triggered by clicking the monster's image label.
        """
        logger.info(f"显示 {clicked_monster.name} ({clicked_side}侧) 的双向伤害信息...")

        # Identify the source and target lists based on which side was clicked
        if clicked_side == 'left':
            left_monsters_for_dialog = [clicked_monster] # Correct: List[Monster]
            # Extract only Monster objects from the tuples returned by _get_monsters_from_layout
            right_monsters_with_counts = self._get_monsters_from_layout(self.right_display_layout)
            right_monsters_for_dialog = [monster for monster, count in right_monsters_with_counts] # Correct: List[Monster]
            left_side_name = "左侧选中"
            right_side_name = "右侧全体"
            if not right_monsters_for_dialog:
                 QMessageBox.information(self, "提示", "右侧没有怪物可供计算伤害。")
                 return
        else: # Clicked on the right side
            # Extract only Monster objects from the tuples returned by _get_monsters_from_layout
            left_monsters_with_counts = self._get_monsters_from_layout(self.left_display_layout)
            left_monsters_for_dialog = [monster for monster, count in left_monsters_with_counts] # Correct: List[Monster]
            right_monsters_for_dialog = [clicked_monster] # Correct: List[Monster]
            left_side_name = "左侧全体"
            right_side_name = "右侧选中"
            if not left_monsters_for_dialog:
                 QMessageBox.information(self, "提示", "左侧没有怪物可供计算伤害。")
                 return


        # Create and show the dialog, passing the lists of Monster objects
        dialog = DamageInfoWindow(left_monsters_for_dialog, right_monsters_for_dialog, left_side_name, right_side_name, self)
        dialog.exec() # Use exec() for modal behavior

    def _populate_display(self,
                          display_layout: QVBoxLayout,
                          monster_template_names: List[str],
                          side: Literal['left', 'right'],
                          recognition_results: Dict[Literal['left', 'right'], Dict[str, Optional[int]]]):
        """
        Populates the given layout with monster cards for the specified side,
        including the recognized count.
        """
        if not monster_template_names:
            # Optionally add a placeholder if no monsters are found for this side
            # display_layout.addWidget(QLabel("未发现"))
            return

        for template_name in monster_template_names:
            monster_info: Monster | None = self.monster_data.get(template_name)

            if not monster_info:
                logger.warning(f"在数据中找不到模板名称 '{template_name}' 对应的怪物信息。")
                # Optionally add a placeholder card for missing data
                missing_label = QLabel(f"{template_name} (数据缺失)")
                display_layout.addWidget(missing_label)
                continue

            # Get the count for this specific monster from the results
            count = recognition_results.get(side, {}).get(template_name)

            # Create and add the monster card, passing the side and count
            monster_card = self._create_monster_card(monster_info, side, count)
            display_layout.addWidget(monster_card)


    # --- Mistake Book Menu Actions ---

    def _add_mistake_book_entry(self):
        """Action: Opens the dialog to add the current combination to the mistake book."""
        # Get current monsters and their counts from display layouts
        left_monsters_with_counts = self._get_monsters_from_layout(self.left_display_layout)
        right_monsters_with_counts = self._get_monsters_from_layout(self.right_display_layout)

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
        left_monsters = self._get_monsters_from_layout(self.left_display_layout)
        right_monsters = self._get_monsters_from_layout(self.right_display_layout)

        if not left_monsters and not right_monsters:
            QMessageBox.information(self, "查询当前组合", "左右两侧均无怪物可供查询。")
            return

        # Extract only the Monster object (first element) from the tuple
        left_types = [monster.template_name for monster, count in left_monsters]
        right_types = [monster.template_name for monster, count in right_monsters]

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
                self._browse_mistake_history() # Open the history browser if user clicks Yes
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

        # 1. Get monster lists from layouts
        left_monsters_with_counts = self._get_monsters_from_layout(self.left_display_layout)
        right_monsters_with_counts = self._get_monsters_from_layout(self.right_display_layout)

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