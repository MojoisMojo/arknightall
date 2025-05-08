import sys
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)
import sqlite3
import json
import torch
import numpy as np
from src.ui.mistake_book_manager import MistakeBookManager
from src.core.prediction import load_id_mapping, MONSTER_COUNT, UnitAwareTransformer
import copy
import os
import time

def load_mistake_data():
    """从错题集数据库中加载训练数据"""
    manager = MistakeBookManager()
    mistakes = manager.load_all_mistakes()
    id_mapping = load_id_mapping()

    training_data = []
    for entry in mistakes:
        try:
            # 解析怪物数据
            left_monsters = entry["left"]["monsters"]
            right_monsters = entry["right"]["monsters"]
            outcome = entry["outcome"]  # 'left_win', 'right_win', 或 'draw'

            # 转换为模型输入格式
            left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
            right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

            # 填充左侧数据
            for _ui_id, count in left_monsters.items():
                ui_id = _ui_id[4:]
                if ui_id in id_mapping and id_mapping[ui_id].isdigit():
                    model_id_idx = int(id_mapping[ui_id]) - 1
                    if 0 <= model_id_idx < MONSTER_COUNT:
                        left_counts[model_id_idx] = count
                else:
                    print(f"UI ID {ui_id} 在 id_mapping 中未找到或无效")

            # 填充右侧数据
            for _ui_id, count in right_monsters.items():
                ui_id = _ui_id[4:]
                if ui_id in id_mapping and id_mapping[ui_id].isdigit():
                    model_id_idx = int(id_mapping[ui_id]) - 1
                    if 0 <= model_id_idx < MONSTER_COUNT:
                        right_counts[model_id_idx] = count

            # 目标输出 (0: 左胜, 1: 右胜, 0.5: 平局)
            target = (
                0.0
                if outcome == "left_win"
                else (1.0 if outcome == "right_win" else 0.5)
            )

            training_data.append(
                {
                    "left_counts": left_counts,
                    "right_counts": right_counts,
                    "target": target,
                }
            )
        except Exception as e:
            print(f"处理错题记录时出错: {e}")
            continue
    print(f"加载了 {len(training_data)} 条训练数据")

    return training_data


def fine_tune_model(epochs=30, learning_rate=0.0002, batch_size=8):
    """微调现有模型"""
    from src.core.prediction import load_prediction_model, get_device, MODEL_PATH
    import torch.optim as optim
    import torch.nn as nn

    # 加载当前模型
    model = load_prediction_model()
    device = get_device()

    if model is None:
        print("模型加载失败，无法进行微调")
        return False

    # 准备训练数据
    training_data = load_mistake_data()
    if not training_data:
        print("未找到有效的错题记录，无法进行微调")
        return False

    # 分为训练和验证集
    np.random.shuffle(training_data)  # 打乱数据顺序
    split_index = int(len(training_data) * 0.8)
    train_data = training_data[:split_index]
    val_data = training_data[split_index:]

    print(f"训练数据集大小: {len(train_data)}, 验证数据集大小: {len(val_data)}")

    # 设置为训练模式
    model.train()

    # 跟踪最佳模型
    best_val_accuracy = 0.0
    best_model_state = None

    # 时间戳用于模型保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # 二元交叉熵损失

    # 训练循环
    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        model.train()
        total_loss = 0
        np.random.shuffle(training_data)  # 打乱数据顺序

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i : i + batch_size]

            # 准备批次数据
            left_signs_batch = []
            left_counts_batch = []
            right_signs_batch = []
            right_counts_batch = []
            targets_batch = []

            for item in batch:
                left_counts = item["left_counts"]
                right_counts = item["right_counts"]

                left_signs = np.sign(left_counts)
                right_signs = np.sign(right_counts)

                left_signs_batch.append(left_signs)
                left_counts_batch.append(np.abs(left_counts))
                right_signs_batch.append(right_signs)
                right_counts_batch.append(np.abs(right_counts))
                targets_batch.append(item["target"])

            # 转换为张量
            left_signs_tensor = torch.tensor(
                np.array(left_signs_batch), dtype=torch.int16
            ).to(device)
            left_counts_tensor = torch.tensor(
                np.array(left_counts_batch), dtype=torch.int16
            ).to(device)
            right_signs_tensor = torch.tensor(
                np.array(right_signs_batch), dtype=torch.int16
            ).to(device)
            right_counts_tensor = torch.tensor(
                np.array(right_counts_batch), dtype=torch.int16
            ).to(device)
            targets_tensor = torch.tensor(targets_batch, dtype=torch.float32).to(device)

            # 前向传播
            outputs = model(
                left_signs_tensor,
                left_counts_tensor,
                right_signs_tensor,
                right_counts_tensor,
            )
            # 确保输出是张量格式并进行截断到[0,1]范围
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() > 0 and outputs.size(0) > 1:  # 如果是批次输出
                    # 应用截断到每个批次元素
                    outputs = torch.clamp(outputs, 0.0, 1.0)
                else:  # 单一输出
                    outputs = torch.clamp(outputs, 0.0, 1.0)
            else:  # 非张量输出
                # 转换为张量后截断
                outputs = torch.tensor(
                    [min(max(0.0, float(outputs)), 1.0)], device=device
                )

            # 确保不存在NaN值
            if torch.isnan(outputs).any():
                print("警告: 输出包含NaN值")
                outputs = torch.where(
                    torch.isnan(outputs), torch.zeros_like(outputs), outputs
                )

            # 计算损失
            loss = criterion(outputs, targets_tensor)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / ((len(training_data) + batch_size - 1) // batch_size)
        # ---------- 验证阶段 ----------
        model.eval()  # 切换到评估模式
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():  # 不计算梯度
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i : i + batch_size]

                # 准备验证批次数据
                left_signs_batch = []
                left_counts_batch = []
                right_signs_batch = []
                right_counts_batch = []
                targets_batch = []

                for item in batch:
                    left_counts = item["left_counts"]
                    right_counts = item["right_counts"]

                    left_signs = np.sign(left_counts)
                    right_signs = np.sign(right_counts)

                    left_signs_batch.append(left_signs)
                    left_counts_batch.append(np.abs(left_counts))
                    right_signs_batch.append(right_signs)
                    right_counts_batch.append(np.abs(right_counts))
                    targets_batch.append(item["target"])

                # 转换为张量
                left_signs_tensor = torch.tensor(
                    np.array(left_signs_batch), dtype=torch.float32
                ).to(device)
                left_counts_tensor = torch.tensor(
                    np.array(left_counts_batch), dtype=torch.float32
                ).to(device)
                right_signs_tensor = torch.tensor(
                    np.array(right_signs_batch), dtype=torch.float32
                ).to(device)
                right_counts_tensor = torch.tensor(
                    np.array(right_counts_batch), dtype=torch.float32
                ).to(device)
                targets_tensor = torch.tensor(targets_batch, dtype=torch.float32).to(
                    device
                )

                # 前向传播
                outputs = model(
                    left_signs_tensor,
                    left_counts_tensor,
                    right_signs_tensor,
                    right_counts_tensor,
                )

                # 确保输出是张量格式并进行截断到[0,1]范围
                if isinstance(outputs, torch.Tensor):
                    if outputs.dim() > 0 and outputs.size(0) > 1:  # 如果是批次输出
                        # 应用截断到每个批次元素
                        outputs = torch.clamp(outputs, 0.0, 1.0)
                    else:  # 单一输出
                        outputs = torch.clamp(outputs, 0.0, 1.0)
                else:  # 非张量输出
                    # 转换为张量后截断
                    outputs = torch.tensor(
                        [min(max(0.0, float(outputs)), 1.0)], device=device
                    )

                # 确保不存在NaN值
                if torch.isnan(outputs).any():
                    print("警告: 输出包含NaN值")
                    outputs = torch.where(
                        torch.isnan(outputs), torch.zeros_like(outputs), outputs
                    )

                loss = criterion(outputs, targets_tensor)
                val_loss += loss.item()
                probs = outputs.cpu().numpy()
                for j in range(len(batch)):
                    target = targets_batch[j]
                    if (probs[j] >= 0.53 and target == 1.0) or (
                        probs[j] < 0.47 and target == 0.0
                    ):
                        correct += 1
                total += len(batch)
        # 计算验证准确率和平均损失
        val_accuracy = correct / total if total > 0 else 0
        avg_val_loss = val_loss / ((len(val_data) + batch_size - 1) // batch_size)
        
        print(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}, 验证准确率: {val_accuracy:.4f}")
        
        # 保存验证准确率最高的模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            
            # 保存最佳模型
            best_model_path = MODEL_PATH.replace(".pth", f"_best_{timestamp}.pth")
            torch.save(model, best_model_path)
            print(f"新的最佳模型已保存，验证准确率: {best_val_accuracy:.4f}")

    # 恢复为评估模式
    model.eval()
    return True


if __name__ == "__main__":
    fine_tune_model(epochs=30, learning_rate=0.0002, batch_size=8)
    print("微调完成")
