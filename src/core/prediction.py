import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
import numpy as np
import torch
from PyQt6.QtWidgets import QMessageBox # For showing errors if needed
from src.core.log import logger # Import logger for logging errors and info
# Adjust path to import from CannotMax and project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
MONSTER_COUNT = 56
# Define paths relative to the project root (d:/Project/python/ArknightALL)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
MODEL_PATH = os.path.join(DATA_DIR, 'models', 'best_model_full.pth') # Adjusted path
MAPPING_PATH = os.path.join(DATA_DIR, 'mapping.json')

# Global cache for model and mapping to avoid reloading
_prediction_model = None
_id_mapping = None
_device = None

class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 注意力层与FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.enemy_ffn.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim * 2, embed_dim)
            ))

            # 友方注意力层
            self.friend_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.friend_ffn.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim * 2, embed_dim)
            ))

            # 初始化注意力层参数
            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1)
        )


    def forward(self, left_sign, left_count, right_sign, right_count):
        # 提取Top3兵种特征
        left_values, left_indices = torch.topk(left_count, k=3, dim=1)
        right_values, right_indices = torch.topk(right_count, k=3, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)  # (B, 3, 128)
        right_feat = self.unit_embed(right_indices)  # (B, 3, 128)

        embed_dim = self.embed_dim

        # 前x维不变，后y维 *= 数量，但使用缩放后的值
        left_feat = torch.cat([
            left_feat[..., :embed_dim // 2],  # 前x维
            left_feat[..., embed_dim // 2:] * left_values.unsqueeze(-1)  # 后y维乘数量
        ], dim=-1)
        right_feat = torch.cat([
            right_feat[..., :embed_dim // 2],
            right_feat[..., embed_dim // 2:] * right_values.unsqueeze(-1)
        ], dim=-1)

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, 3) 0.1防一手可能的浮点误差
        left_mask = (left_values > 0.1)
        right_mask = (right_values > 0.1)

        for i in range(self.num_layers):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            delta_left, _ = self.friend_attentions[i](
                query=left_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)

        # 输出战斗力
        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask

        # 计算战斗力差输出概率，'L': 0, 'R': 1，R大于L时输出大于0.5
        output = torch.sigmoid(R.sum(1) - L.sum(1))

        return output

def get_device():
    """Gets the torch device."""
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Prediction using device: {_device}")
    return _device

def load_id_mapping() -> Optional[Dict[str, str]]:
    """Loads the UI ID to Model ID mapping from mapping.json."""
    global _id_mapping
    if _id_mapping is not None:
        return _id_mapping

    if not os.path.exists(MAPPING_PATH):
        logger.error(f"Error: Mapping file not found at {MAPPING_PATH}")
        QMessageBox.critical(None, "错误", f"ID映射文件丢失: {MAPPING_PATH}")
        return None
    try:
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            _id_mapping = json.load(f)
        logger.info("ID mapping loaded successfully.")
        return _id_mapping
    except json.JSONDecodeError:
        logger.error(f"Error: Failed to decode JSON from {MAPPING_PATH}")
        QMessageBox.critical(None, "错误", f"ID映射文件格式错误: {MAPPING_PATH}")
        return None
    except Exception as e:
        logger.error(f"Error loading mapping file: {e}")
        QMessageBox.critical(None, "错误", f"加载ID映射文件时出错: {e}")
        return None

def load_prediction_model(use_fine_tuned=True):
    """Loads the prediction model."""
    global _prediction_model
    if _prediction_model is not None:
        return _prediction_model

    device = get_device()
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Error: Model file not found at {MODEL_PATH}")
        QMessageBox.critical(None, "错误", f"预测模型文件丢失: {MODEL_PATH}\n请确保模型已训练并放置在正确位置。")
        return None

    # 尝试加载微调模型
    fine_tuned_path = MODEL_PATH.replace('.pth', '_fine_tuned.pth')
    if use_fine_tuned and os.path.exists(fine_tuned_path):
        try:
            model = torch.load(fine_tuned_path, map_location=device)
            model.eval()
            _prediction_model = model
            logger.info("已加载微调后的预测模型")
            return _prediction_model
        except Exception as e:
            logger.warning(f"加载微调模型失败: {e}，将尝试加载原始模型")
    
    try:
        # Try loading with weights_only=False first for custom classes
        try:
            model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        except TypeError: # Handle older PyTorch versions
            model = torch.load(MODEL_PATH, map_location=device)
        except AttributeError as ae: # Handle potential issues if class definition changed
             logger.error(f"AttributeError loading model (potential class mismatch): {ae}")
             QMessageBox.warning(None, "模型加载警告", f"加载模型时遇到属性错误 (可能类定义不匹配): {ae}\n尝试仅加载权重...")
             # Fallback: Try loading just the state dict if the class is available
            
        except Exception as load_e: # Catch other loading errors
             logger.error(f"General error loading model: {load_e}")
             QMessageBox.critical(None, "模型加载失败", f"加载模型时发生一般错误: {load_e}")
             return None


        model.eval()
        _prediction_model = model.to(device)
        logger.debug(f"Prediction model loaded successfully onto {device}.")
        return _prediction_model
    except FileNotFoundError:
        logger.error(f"Error: Model file not found at {MODEL_PATH}")
        QMessageBox.critical(None, "错误", f"预测模型文件丢失: {MODEL_PATH}")
        return None
    except Exception as e:
        error_msg = f"模型加载失败: {str(e)}"
        if "missing keys" in str(e) or "unexpected keys" in str(e):
            error_msg += "\n可能是模型结构不匹配，请确保使用的模型文件正确或重新训练。"
        elif "No such file or directory" in str(e):
             error_msg = f"模型文件未找到: {MODEL_PATH}"
        logger.error(f"Error: {error_msg}")
        QMessageBox.critical(None, "严重错误", error_msg)
        return None


def predict_outcome(left_monster_ui_ids: List[str],
                    right_monster_ui_ids: List[str]) -> Optional[float]:
    """
    Performs prediction based on UI monster IDs using the loaded model and mapping.

    Args:
        left_monster_ui_ids: List of UI monster IDs (strings, e.g., '36') on the left.
        right_monster_ui_ids: List of UI monster IDs (strings, e.g., '58') on the right.

    Returns:
        Predicted probability for the right side winning (float between 0 and 1),
        or None if prediction fails.
    """
    model = load_prediction_model()
    id_mapping = load_id_mapping()
    device = get_device()

    if model is None or id_mapping is None:
        print("Prediction failed: Model or ID mapping not loaded.")
        return None

    try:
        # 1. Count occurrences of each UI ID
        left_ui_counts = {ui_id: left_monster_ui_ids.count(ui_id) for ui_id in set(left_monster_ui_ids)}
        right_ui_counts = {ui_id: right_monster_ui_ids.count(ui_id) for ui_id in set(right_monster_ui_ids)}

        # 2. Prepare mapped input arrays (initialized to zeros)
        # These arrays will be indexed by the *model's* ID (0 to MONSTER_COUNT-1)
        left_counts_mapped = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right_counts_mapped = np.zeros(MONSTER_COUNT, dtype=np.int16)
        skipped_ids = []

        # 3. Map left side counts
        for ui_id, count in left_ui_counts.items():
            model_id_str = id_mapping.get(ui_id)
            if model_id_str and model_id_str.isdigit():
                model_id_idx = int(model_id_str) - 1 # Assuming model IDs in mapping are 1-based
                if 0 <= model_id_idx < MONSTER_COUNT:
                    left_counts_mapped[model_id_idx] = count
                else:
                    skipped_ids.append(f"左侧 UI ID {ui_id} -> 模型 ID {model_id_str} (越界)")
            else:
                skipped_ids.append(f"左侧 UI ID {ui_id} (无映射或无效: '{model_id_str}')")

        # 4. Map right side counts
        for ui_id, count in right_ui_counts.items():
            model_id_str = id_mapping.get(ui_id)
            if model_id_str and model_id_str.isdigit():
                model_id_idx = int(model_id_str) - 1 # Assuming model IDs in mapping are 1-based
                if 0 <= model_id_idx < MONSTER_COUNT:
                    right_counts_mapped[model_id_idx] = count
                else:
                    skipped_ids.append(f"右侧 UI ID {ui_id} -> 模型 ID {model_id_str} (越界)")
            else:
                skipped_ids.append(f"右侧 UI ID {ui_id} (无映射或无效: '{model_id_str}')")

        if skipped_ids:
            logger.error(f"警告：预测时跳过以下怪物ID：{', '.join(skipped_ids)}")
            # Optionally show a non-critical warning to the user
            # QMessageBox.warning(None, "映射警告", f"部分怪物ID无法映射，预测可能不准确:\n{', '.join(skipped_ids)}")

        # 5. Convert to tensors and prepare model input (similar to CannotMax/main.py)
        # Ensure dtype is appropriate for the model's embedding layers (often torch.long or torch.int)
        # Using int16 based on CannotMax/main.py, but check model requirements.
        left_signs = torch.sign(torch.tensor(left_counts_mapped, dtype=torch.int16)).unsqueeze(0).to(device)
        left_counts_abs = torch.abs(torch.tensor(left_counts_mapped, dtype=torch.int16)).unsqueeze(0).to(device)
        right_signs = torch.sign(torch.tensor(right_counts_mapped, dtype=torch.int16)).unsqueeze(0).to(device)
        right_counts_abs = torch.abs(torch.tensor(right_counts_mapped, dtype=torch.int16)).unsqueeze(0).to(device)

        # 6. Perform prediction
        prediction = 0.5 # Default value
        with torch.no_grad():
            # The input arguments MUST match the model's forward method signature
            # Based on CannotMax/main.py, it seems to be (left_signs, left_counts, right_signs, right_counts)
            output = model(left_signs, left_counts_abs, right_signs, right_counts_abs)
            # Assuming the model output is a single value (or can be converted to one)
            if isinstance(output, torch.Tensor):
                 prediction = output.item()
            elif isinstance(output, (float, int)):
                 prediction = float(output)
            else:
                 logger.warning(f"Warning: Unexpected model output type: {type(output)}. Using default prediction.")
                 prediction = 0.5


            # Ensure prediction is within [0, 1]
            if np.isnan(prediction) or np.isinf(prediction):
                logger.warning("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                prediction = 0.5
            else:
                prediction = max(0.0, min(1.0, prediction))

        logger.info(f"预测成功，右侧胜率: {prediction:.4f}")
        return prediction

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        QMessageBox.critical(None, "预测错误", f"执行预测时发生错误: {e}")
        return None

if __name__ == '__main__':
    # Basic test
    print("Testing prediction module...")
    mapping = load_id_mapping()
    model = load_prediction_model()

    if mapping and model:
        print("\n--- Test Case 1: Basic ---")
        # Example UI IDs (replace with actual IDs from your UI for testing)
        # Assuming '36' maps to model ID 13, '58' maps to model ID 48
        test_left_ids = ['36', '36', '1'] # Two '36', one '1' (maps to 7)
        test_right_ids = ['58']           # One '58'
        print(f"Input Left UI IDs: {test_left_ids}")
        print(f"Input Right UI IDs: {test_right_ids}")
        result = predict_outcome(test_left_ids, test_right_ids)
        if result is not None:
            print(f"Predicted Right Win Probability: {result:.4f}")
        else:
            print("Prediction failed.")

        print("\n--- Test Case 2: Empty Side ---")
        test_left_ids_empty = []
        test_right_ids_empty = ['58', '45'] # '45' maps to 49
        print(f"Input Left UI IDs: {test_left_ids_empty}")
        print(f"Input Right UI IDs: {test_right_ids_empty}")
        result_empty = predict_outcome(test_left_ids_empty, test_right_ids_empty)
        if result_empty is not None:
            print(f"Predicted Right Win Probability: {result_empty:.4f}")
        else:
            print("Prediction failed.")

        print("\n--- Test Case 3: Unmapped ID ---")
        test_left_ids_unmapped = ['36', '999'] # '999' is not in mapping
        test_right_ids_unmapped = ['58']
        print(f"Input Left UI IDs: {test_left_ids_unmapped}")
        print(f"Input Right UI IDs: {test_right_ids_unmapped}")
        result_unmapped = predict_outcome(test_left_ids_unmapped, test_right_ids_unmapped)
        if result_unmapped is not None:
            print(f"Predicted Right Win Probability: {result_unmapped:.4f}")
        else:
            print("Prediction failed.")

    else:
        print("Could not run tests because model or mapping failed to load.")