import json
import os

# 定义设置文件的路径
SETTINGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
SETTINGS_FILE = os.path.join(SETTINGS_DIR, 'settings.json')

# 确保目录存在
os.makedirs(SETTINGS_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    "hide_recognition_image": False,  # 新增：隐藏识别结果图像
    "simplify_monster_card": False, # 新增：简化怪物卡片显示
    "monster_list_hidden": False, # 新增：怪物列表是否隐藏
    "window_geometry": None,      # 新增：存储窗口位置和大小 [x, y, width, height]
    "always_on_top": False,         # 新增：窗口是否置顶
    # 在这里添加其他默认设置
}

def load_settings():
    """加载设置文件，如果文件不存在则返回默认设置"""
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy() # 返回默认设置的副本
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings_data = json.load(f)
            # 合并加载的设置和默认设置，以防缺少某些键
            current_settings_state = DEFAULT_SETTINGS.copy()
            current_settings_state.update(settings_data)
            return current_settings_state
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading settings file: {e}. Returning default settings.")
        return DEFAULT_SETTINGS.copy() # 出错时返回默认设置的副本

def save_settings(settings_to_save):
    """将设置保存到文件"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_to_save, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving settings file: {e}")

# 在模块加载时加载一次设置
# 将加载的设置存储在一个可修改的变量中
_current_settings = load_settings()

def get_setting(key, default=None):
    """获取单个设置项的值"""
    # 从 _current_settings 获取值，如果键不存在，则尝试从 DEFAULT_SETTINGS 获取
    return _current_settings.get(key, DEFAULT_SETTINGS.get(key, default))

def update_setting(key, value):
    """更新单个设置项并保存"""
    _current_settings[key] = value
    save_settings(_current_settings)