# 明日方舟斗蛐蛐错题册 (Arknights Combat Analysis & Mistake Book)

## 简介

本项目是一个基于 Python 和 PyQt6 的桌面应用程序，旨在帮助《明日方舟》博士分析“斗蛐蛐”或其他需要快速识别敌方单位并评估对抗情况的场景。

它通过屏幕识别技术自动识别指定区域内的敌方单位，展示其详细属性，并提供一个“错题本”功能，方便博士记录和回顾失误的对局组合。

<del>本项目不能直接预测胜率，但能通过数值展示和运算，错题回顾等提高博士们的斗蛐蛐水平</del>

**现在已添加坎诺特计算器的预测功能，并优化了大量UI**

有关坎诺特计算器，请前往[CannotMax](https://github.com/Ancientea/CannotMax)


<del> 本相关99.9%的代码由AI生成，作者不对代码质量与实现方法负责</del>

## 主要功能

*   **屏幕区域识别 (ROI Selection & Recognition)**:
    *   允许用户在屏幕上选择一个特定区域 (ROI)。
    *   实时或手动触发捕获该区域的截图。
    *   使用图像识别技术 (OpenCV) 自动识别区域内出现的明日方舟敌方单位（基于 `data/image/` 下的模板图片）。
    *   在界面上显示带有识别框标注的截图。

*   **怪物信息展示 (Monster Information Display)**:
    *   在识别后，将左右两侧的怪物分别以卡片形式展示。
    *   怪物卡片显示怪物的图标、名称、以及关键属性（生命、攻击、防御、法抗、攻速、攻击范围、特殊能力等，数据来源于 `data/monster.csv`）(部分数据来源 bilibili@碱式碳酸咸鱼)。
    *   攻击力根据伤害类型（物理/法术）以不同颜色显示。

*   **手动添加怪物 (Manual Monster Addition)**:
    *   提供所有已知怪物的图标列表。
    *   用户可以手动选择怪物并将其添加到左侧或右侧的显示区域。

*   **双向伤害计算 (Bidirectional Damage Calculation)**:
    *   点击任一侧的怪物卡片，可以弹出窗口显示该怪物与另一侧所有怪物之间的双向伤害计算结果（考虑攻击、防御、法抗等因素）。

*   **错题本管理 (Mistake Book Management)**:
    *   **添加记录**: 可以将当前屏幕识别或手动配置的怪物组合（包括左右两侧）保存到错题本（SQLite 数据库 `data/mistakes.db`）。添加时可以记录备注、标签、对局结果（赢/输）等信息。
    *   **查询记录**: 快速查询当前显示的怪物组合是否已存在于错题本中。
    *   **浏览历史**: 查看所有已保存的错题记录，包括当时的怪物组合、截图（如果识别时保存了）、备注等。
    *   **自动匹配提示**: 每次屏幕识别完成后，自动检查识别出的怪物组合是否在错题本中有匹配记录，并进行提示。

## 截图
**最新版本**:

![图片](https://github.com/user-attachments/assets/5d71f580-62b1-4b92-aad1-ace63f1e6964)

![图片](https://github.com/user-attachments/assets/f9b9b9b5-ddd6-4253-bd08-f84f8c8c497d)
![图片](https://github.com/user-attachments/assets/337db46d-b1d8-44f6-b489-e3eb0a7ce84c)
![图片](https://github.com/user-attachments/assets/2bd9e2cf-b217-48c9-abf4-b35459b7b3be)
![图片](https://github.com/user-attachments/assets/1410b333-e3e6-4470-a32a-30b1f5187ddd)


## 技术栈

*   **语言**: Python 3
*   **GUI**: PyQt6
*   **图像处理与识别**: OpenCV-Python, MSS (截图)
*   **数据库**: SQLite (用于错题本)
*   **日志**: colorlog, logging

## 运行环境要求

*   **Python**: 3.8 或更高版本 (建议)
*   **操作系统**: Windows
*   **依赖库**: 见 `requirements.txt`

## 安装与运行
1.  **克隆或下载仓库**:
    ```bash
    git clone https://github.com/xinhai-ai/arknightall # 或者直接下载 ZIP 解压
    cd arknightall
    ```
    
2.  **创建并激活虚拟环境** (可选；推荐):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **运行程序**:
    ```bash
    python main.py
    ```

## 文件结构简述

```
ArknightALL/
│
├── main.py                 # 程序主入口脚本
├── README.md               # 本文档
├── requirements.txt        # Python 依赖列表
│
├── data/                   # 数据文件目录
│   ├── image/              # 怪物识别模板图片 (.png)
│   ├── monster.csv         # 怪物属性数据
│   └── mistakes.db         # 错题本 SQLite 数据库文件
│
├── src/                    # 源代码目录
│   ├── core/               # 核心功能模块 (截图, 识别, 数据加载, 日志等)
│   ├── models/             # 数据模型定义 (如 Monster 类)
│   └── ui/                 # 用户界面相关代码 (主窗口, 对话框等)
│
└── log/                    # 日志文件目录
    └── app.log             # 应用程序日志文件
```

## 注意事项

*   识别效果依赖于模板图片的质量、游戏内怪物的显示清晰度以及选择的识别区域。
*   怪物与数量识别不保证百分百准确
