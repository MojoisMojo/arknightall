# Arknight Monster Recognition Helper

一个使用 PyQt6 和 OpenCV 构建的桌面应用程序，旨在辅助《明日方舟》玩家识别游戏画面中的怪物并展示其属性。

## 功能

*   通过按钮触发屏幕区域选择。
*   实时捕获用户选择的屏幕区域。
*   使用 `data/image/` 中的模板图片在捕获的区域中识别怪物。
*   从 `data/monster.csv` 加载怪物属性数据。
*   在 UI 界面上展示识别出的怪物列表、数量及其详细属性。

## 项目结构

```
ArknightALL/
├── data/
│   ├── image/            # 存放怪物模板图片 (obj_1.png, obj_2.png, ...)
│   └── monster.csv       # 存放怪物属性数据
├── src/
│   ├── __init__.py
│   ├── core/             # 核心逻辑模块
│   │   ├── __init__.py
│   │   ├── data_loader.py  # 加载 monster.csv 数据
│   │   ├── image_recognition.py # 图像识别逻辑 (OpenCV)
│   │   └── screen_capture.py # 屏幕区域捕获逻辑
│   ├── ui/               # UI界面相关模块
│   │   ├── __init__.py
│   │   ├── main_window.py  # 主窗口界面定义
│   │   └── selection_overlay.py # (可选) 用于屏幕选择的透明覆盖层
│   └── models/           # 数据模型 (例如 Monster 类)
│       ├── __init__.py
│       └── monster.py
├── main.py               # 应用程序入口
├── requirements.txt      # 项目依赖库
└── README.md             # 项目说明文档
```

## 核心流程

```mermaid
graph LR
    A[用户启动程序] --> B(显示主窗口 main_window.py);
    B --> C{点击 "选择区域" 按钮};
    C --> D[触发屏幕捕获 screen_capture.py];
    D --> E{用户在屏幕上框选区域};
    E --> F[获取选中区域截图];
    F --> G[调用图像识别 image_recognition.py];
    G -- 传入截图和模板图片 --> H{使用OpenCV模板匹配};
    H --> I[识别出的怪物列表及数量];
    I --> J[加载怪物数据 data_loader.py];
    J -- 从 monster.csv 读取属性 --> K[获取怪物详细属性];
    K --> L[在主窗口UI上展示结果];
    L --> B; # 返回主窗口等待下一次操作
```

## 技术栈

*   **UI**: PyQt6
*   **图像识别**: OpenCV (`opencv-python`)
*   **数据存储**: CSV (`data/monster.csv`)
*   **屏幕捕获**: (待定，可能使用 `mss` 或其他库)