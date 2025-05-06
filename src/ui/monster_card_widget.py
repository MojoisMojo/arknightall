# src/ui/monster_card_widget.py

import os
from typing import Dict, Literal, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QUrl, QPoint # Added QPoint, QUrl, QTimer
from PyQt6.QtGui import QPixmap, QMouseEvent, QImage, QFont, QIntValidator, QPainter, QColor # Added QPainter, QColor, QFont, QIntValidator
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QScrollArea, QSizePolicy
)

# Assuming these imports are accessible from this new location
# Adjust relative paths if necessary based on your project structure
from src.models.monster import Monster
from src.core import settings # Import settings module
from src.core.log import logger # Import the logger

# Define paths relative to this file's potential execution context
# This might need adjustment depending on how the app is run/packaged
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TEMPLATE_DIR = os.path.join(DATA_DIR, 'image')


# --- Custom Clickable Label ---
class ClickableImageLabel(QLabel):
    """A QLabel that emits a clicked signal when clicked, passing monster_info and side."""
    # Signal emitting the Monster object and the side ('left' or 'right')
    # The side information might be redundant if the parent (MonsterCardWidget) already knows its side.
    # Let's emit only the Monster object for simplicity here. The receiver can get the side from the sender's parent if needed.
    # UPDATE: Keeping side for now as the original connection used it.
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


# --- Monster Card Widget ---
class MonsterCardWidget(QWidget):
    """Displays a single monster's information."""
    # Signal emitted when the remove button is clicked
    remove_requested = pyqtSignal(QWidget) # Emit self to be removed
    # Signal emitted when the image label is clicked, passing the Monster object and side
    details_requested = pyqtSignal(Monster, str)
    # Signal emitted when the count changes
    count_changed = pyqtSignal()

    def __init__(self, monster_info: Monster, side: Literal['left', 'right'], initial_count: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.monster_data = monster_info
        self.side = side
        # Ensure recognized_count is initialized properly, defaulting to 1 if initial_count is None or invalid
        self.recognized_count = 1
        if initial_count is not None:
             try:
                 if int(initial_count) > 0:
                     self.recognized_count = int(initial_count)
             except (ValueError, TypeError):
                 pass # Keep default 1

        simplify_card = settings.get_setting("simplify_monster_card", False)

        # Add a style sheet for subtle bordering and hover effect
        self.setStyleSheet("""
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

        # --- Main Card Layout (Horizontal or Vertical based on setting) ---
        if simplify_card:
            card_layout = QHBoxLayout(self)
            card_layout.setContentsMargins(3, 3, 3, 3)
            card_layout.setSpacing(4)
        else:
            card_layout = QHBoxLayout(self)
            card_layout.setContentsMargins(5, 5, 5, 5)
            card_layout.setSpacing(5)

        # --- Image Label (Common) ---
        self.image_label = ClickableImageLabel(monster_info, side)
        image_size = 48 if simplify_card else 64
        self.image_label.setFixedSize(image_size, image_size)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f8f8f8;")
        self.image_label.setToolTip("点击查看双向伤害详情")

        image_path = os.path.join(TEMPLATE_DIR, f"{monster_info.template_name}.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText("无图")
            logger.warning(f"无法加载图片 {image_path}")
        else:
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Connect the label's clicked signal to the card's details_requested signal
        self.image_label.clicked.connect(self.details_requested)

        # --- Count Input (Common) ---
        self.count_input = QLineEdit()
        self.count_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.count_input.setFixedWidth(self.image_label.width()) # Let's remove fixed width for now
        self.count_input.setToolTip("怪物数量 (可编辑)")
        # Use the initialized self.recognized_count
        self.count_input.setText(str(self.recognized_count))
        self.count_input.setValidator(QIntValidator(1, 99, self))
        self.count_input.editingFinished.connect(self._update_internal_count)

        # --- Layout Specific Elements ---
        if simplify_card:
            card_layout.addWidget(self.image_label) # Image on left

            right_column_layout = QVBoxLayout()
            right_column_layout.setContentsMargins(0, 0, 0, 0)
            right_column_layout.setSpacing(2)

            name_label = QLabel(f"<b>{monster_info.name}</b>")
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            name_label.setWordWrap(True)
            right_column_layout.addWidget(name_label, stretch=1)

            count_remove_hbox = QHBoxLayout()
            count_remove_hbox.setContentsMargins(0, 0, 0, 0)
            count_remove_hbox.setSpacing(5)

            self.count_input.setMaximumWidth(40) # Keep count input small

            remove_button_simplified = QPushButton("X")
            remove_button_simplified.setToolTip("移除此卡片")
            remove_button_simplified.setFixedSize(18, 18)
            remove_button_simplified.setStyleSheet("QPushButton { border: none; background-color: #FFDDDD; border-radius: 9px; font-weight: bold; color: red; } QPushButton:hover { background-color: #FFBBBB; }")
            remove_button_simplified.clicked.connect(lambda: self.remove_requested.emit(self)) # Emit signal

            count_remove_hbox.addWidget(self.count_input)
            count_remove_hbox.addWidget(remove_button_simplified)
            count_remove_hbox.addStretch()

            right_column_layout.addLayout(count_remove_hbox)
            card_layout.addLayout(right_column_layout, stretch=1)

        else: # Full Layout
            image_and_count_layout = QVBoxLayout()
            image_and_count_layout.setContentsMargins(0, 0, 0, 0)
            image_and_count_layout.setSpacing(2)
            image_and_count_layout.addWidget(self.image_label)
            # Set count input width to match image width for alignment
            self.count_input.setFixedWidth(image_size)
            image_and_count_layout.addWidget(self.count_input)
            card_layout.addLayout(image_and_count_layout)

            stats_widget = QWidget()
            stats_widget.setStyleSheet("border: none; background-color: transparent;")
            stats_layout = QVBoxLayout(stats_widget)
            stats_layout.setContentsMargins(0,0,0,0)
            stats_layout.setSpacing(1)

            def format_value(value, default='-'):
                return str(value) if value is not None and str(value).strip() != '' else default

            font_size = 12
            name_font_size = 14
            name_label = QLabel(f"<font style='font-size: {name_font_size}px;'><b>{monster_info.name}</b> ({monster_info.template_name})</font>")
            stats_layout.addWidget(name_label)

            attack_color = "red" if monster_info.is_magic_damage else "blue"
            attack_text = f"攻击: {format_value(monster_info.attack)}"
            attack_label = QLabel(f"<font color='{attack_color}' style='font-size: {font_size}px;'>{attack_text}</font>")
            stats_layout.addWidget(attack_label)

            stats_layout.addWidget(QLabel(f"<font color='green' style='font-size: {font_size}px;'>生命: {format_value(monster_info.health)}</font>"))
            stats_layout.addWidget(QLabel(f"<font color='saddlebrown' style='font-size: {font_size}px;'>防御: {format_value(monster_info.defense)}</font>"))
            stats_layout.addWidget(QLabel(f"<font color='darkviolet' style='font-size: {font_size}px;'>法抗: {format_value(monster_info.resistance)}</font>"))
            stats_layout.addWidget(QLabel(f"<font color='orange' style='font-size: {font_size}px;'>攻速: {format_value(monster_info.attack_interval)}</font>"))

            attack_range_value = monster_info.attack_range
            attack_range_text = "近战"
            try:
                if attack_range_value is not None and str(attack_range_value).strip() and float(attack_range_value) > 0:
                    attack_range_text = format_value(attack_range_value)
            except (ValueError, TypeError):
                pass
            stats_layout.addWidget(QLabel(f"<font color='darkcyan' style='font-size: {font_size}px;'>攻击范围: {attack_range_text}</font>"))

            ability_text = format_value(monster_info.special_ability, '无')
            ability_label = QLabel(f"<font style='font-size: {font_size}px;'>特殊: {ability_text}</font>")
            ability_label.setWordWrap(True)
            stats_layout.addWidget(ability_label)
            card_layout.addWidget(stats_widget, stretch=1)

            # --- Remove Button (Full Layout) ---
            button_vlayout = QVBoxLayout()
            button_vlayout.setSpacing(3)
            button_vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)

            remove_button = QPushButton("X")
            remove_button.setToolTip("移除此卡片")
            remove_button.setFixedSize(20, 20)
            remove_button.setStyleSheet("QPushButton { border: none; background-color: #FFDDDD; border-radius: 10px; font-weight: bold; color: red; } QPushButton:hover { background-color: #FFBBBB; }")
            remove_button.clicked.connect(lambda: self.remove_requested.emit(self)) # Emit signal
            button_vlayout.addWidget(remove_button)
            card_layout.addLayout(button_vlayout)

        # --- Size Policy ---
        if simplify_card:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.setMaximumHeight(image_size + 10)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _update_internal_count(self):
        """Updates the recognized_count based on QLineEdit input and emits signal."""
        text = self.count_input.text()
        old_count = self.recognized_count
        new_count_valid = False
        try:
            new_count = int(text)
            if new_count > 0:
                if self.recognized_count != new_count:
                    self.recognized_count = new_count
                    logger.debug(f"怪物 {self.monster_data.name} ({self.side}) 数量更新为: {new_count}")
                    new_count_valid = True
            else:
                # Input is <= 0, reset to 1
                reset_count = 1
                self.count_input.setText(str(reset_count))
                if self.recognized_count != reset_count:
                    self.recognized_count = reset_count
                    logger.warning(f"输入无效 '{text}' (<=0) for {self.monster_data.name}. 数量重置为 {reset_count}.")
                    new_count_valid = True # Count changed to 1
        except ValueError:
            # Input is not an integer, reset to previous valid count or 1
            reset_count = old_count if old_count is not None and old_count > 0 else 1
            self.count_input.setText(str(reset_count))
            # Only update internal count if it was previously invalid (None or <=0)
            if self.recognized_count is None or self.recognized_count <= 0:
                 self.recognized_count = 1 # Ensure it's set to 1
                 logger.warning(f"输入无效 '{text}' (非整数) for {self.monster_data.name}. 数量恢复为 {self.recognized_count}.")
                 new_count_valid = True # Count might have changed from None/invalid to 1
            elif self.recognized_count != reset_count:
                 # This case should ideally not happen if old_count was valid > 0
                 # but added for robustness. Reset internal count if text was invalid.
                 self.recognized_count = reset_count
                 logger.warning(f"输入无效 '{text}' (非整数) for {self.monster_data.name}. 数量恢复为 {self.recognized_count}.")
                 new_count_valid = True # Count might have changed

        # Emit signal only if the count actually changed to a valid value
        if new_count_valid:
            self.count_changed.emit()

    def get_monster_info(self) -> Tuple[Monster, Optional[int]]:
        """Returns the monster object and its current count."""
        # Return the validated internal count
        return self.monster_data, self.recognized_count


# --- Monster Display Area Widget ---
class MonsterDisplayArea(QWidget):
    """Manages a scrollable area displaying multiple MonsterCardWidgets for one side."""
    # Signal emitted when any monster card's image label is clicked
    monster_clicked = pyqtSignal(Monster, str)
    # Signal emitted when the list of monsters changes (add/remove/count change)
    monsters_changed = pyqtSignal()

    def __init__(self, side: Literal['left', 'right'], monster_data_dict: Dict[str, Monster], parent=None):
        super().__init__(parent)
        self.side = side
        self.monster_data_dict = monster_data_dict # Keep reference to all monster data

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.container_widget = QWidget() # Container for the layout inside scroll area
        self.display_layout = QVBoxLayout(self.container_widget) # Layout for monster cards
        self.display_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align cards to the top
        self.scroll_area.setWidget(self.container_widget)

        # Main layout for this widget contains only the scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(self.scroll_area)

    def clear_display(self):
        """Removes all monster cards from the display."""
        needs_emit = self.display_layout.count() > 0 # Check if there's anything to clear
        while self.display_layout.count():
            child = self.display_layout.takeAt(0)
            if child.widget():
                # Disconnect signals before deleting? Maybe not necessary with deleteLater
                child.widget().deleteLater()
        # Emit changed signal only if something was actually cleared
        if needs_emit:
            QTimer.singleShot(0, self.monsters_changed.emit) # Use timer to ensure deletion happens first

    def populate_display(self,
                         monster_template_names: List[str],
                         recognition_results: Dict[Literal['left', 'right'], Dict[str, Optional[int]]]):
        """Populates the display with monster cards based on recognition results."""
        self.clear_display() # Clear existing cards first

        if not monster_template_names:
            # Emit changed signal even if empty (clear_display might have emitted)
            # Ensure it's emitted if the list was already empty
            if self.display_layout.count() == 0: # Check again after clear
                 QTimer.singleShot(0, self.monsters_changed.emit)
            return

        for template_name in monster_template_names:
            monster_info: Monster | None = self.monster_data_dict.get(template_name)

            if not monster_info:
                logger.warning(f"在数据中找不到模板名称 '{template_name}' 对应的怪物信息。")
                missing_label = QLabel(f"{template_name} (数据缺失)")
                self.display_layout.addWidget(missing_label)
                continue

            count = recognition_results.get(self.side, {}).get(template_name)
            self._add_card(monster_info, count, emit_change=False) # Add card without emitting signal yet

        # Emit changed signal once after all cards are added
        QTimer.singleShot(0, self.monsters_changed.emit)

    def add_monster_manually(self, monster_info: Monster):
        """Adds a single monster card manually (count defaults to 1)."""
        self._add_card(monster_info, initial_count=1, emit_change=True) # Emits signal

    def _add_card(self, monster_info: Monster, initial_count: Optional[int], emit_change: bool = True):
        """Helper to create and add a card, connecting its signals."""
        monster_card = MonsterCardWidget(monster_info, self.side, initial_count)
        # Connect the card's signals to the display area's handlers/signals
        monster_card.remove_requested.connect(self._handle_remove_request)
        monster_card.details_requested.connect(self.monster_clicked) # Re-emit the signal
        monster_card.count_changed.connect(self.monsters_changed) # Re-emit signal when count changes
        self.display_layout.addWidget(monster_card)
        if emit_change:
            # Emit changed signal after adding a single card
            QTimer.singleShot(0, self.monsters_changed.emit)


    def _handle_remove_request(self, card_widget: QWidget):
        """Handles the remove_requested signal from a MonsterCardWidget."""
        card_widget.deleteLater()
        # Emit changed signal after removing a card
        # Use a timer to ensure the widget is removed from layout before state is checked elsewhere
        QTimer.singleShot(10, self.monsters_changed.emit)

    def get_monsters(self) -> List[Tuple[Monster, Optional[int]]]:
        """Retrieves Monster objects and their counts from the current cards."""
        monsters_with_counts = []
        for i in range(self.display_layout.count()):
            item = self.display_layout.itemAt(i)
            widget = item.widget()
            # Check if it's a MonsterCardWidget instance
            if isinstance(widget, MonsterCardWidget):
                monsters_with_counts.append(widget.get_monster_info())
        return monsters_with_counts