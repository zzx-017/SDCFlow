
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

#  CausalTimeDataset作为加载2N维数据和H图的源
class CausalTimeDataset(Dataset):
    def __init__(self, data_path: str, graph_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found at: {graph_path}")

        print(f"Loading 2N-dimensional time series data (X̃) from: {data_path}")
        # data_full 现在被理解为 X̃，形状: (Sample_num, Time_step, 2N)
        self.data_full_2N = np.load(data_path) 
        
        print(f"Loading ground truth graph H (N x N) from: {graph_path}")
        # H_ground_truth 形状: (N, N)
        self.H_ground_truth_np = np.load(graph_path) 

        # 从加载的数据中推断形状和参数
        self.num_samples, self.time_steps_per_sample, self.total_nodes_2N = self.data_full_2N.shape
        self.num_original_nodes_N = self.H_ground_truth_np.shape[0]

        #严格要求输入数据的节点数必须是 H 图节点数的两倍
        if self.total_nodes_2N != (2 * self.num_original_nodes_N):
            raise ValueError(
                f"Data mismatch! The time series file has {self.total_nodes_2N} nodes, "
                f"but the ground truth H implies {self.num_original_nodes_N} original nodes. "
                f"This dataloader requires the time series to have exactly 2N = {2 * self.num_original_nodes_N} nodes (X and X^R)."
            )
        
        if self.H_ground_truth_np.shape[0] != self.H_ground_truth_np.shape[1]:
            raise ValueError(f"Ground truth H must be a square matrix, but got shape {self.H_ground_truth_np.shape}")

        # 模型将使用完整的2N维数据，并假设数据是全观测的
        self.X_tilde_for_model = self.data_full_2N
        self.mask_for_model = np.ones_like(self.X_tilde_for_model)

        print(f"Dataset initialized for SDCFlow (2N-dimensional input):")
        print(f"  Total samples: {self.num_samples}")
        print(f"  Timesteps per sample: {self.time_steps_per_sample}")
        print(f"  Model input shape (Sample_num, Time_step, 2N): {self.X_tilde_for_model.shape}")
        print(f"  Ground truth H shape (N, N): {self.H_ground_truth_np.shape}")
        print(f"  Number of original nodes (N): {self.num_original_nodes_N}")
        print(f"  Number of model nodes (2N): {self.total_nodes_2N}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        sample_X_tilde = self.X_tilde_for_model[idx]
        sample_X_tilde_tensor = torch.from_numpy(sample_X_tilde).float()
        H_tensor = torch.from_numpy(self.H_ground_truth_np).float()
        return sample_X_tilde_tensor, H_tensor


class SDCFlowAdapterDataset(Dataset):
    """
    将原始时间序列数据 (来自 CausalTimeDataset 的 2N 维数据) 适配为 SDCFlow 的序列。
    """
    def __init__(self, raw_data_np: np.ndarray, raw_mask_np: np.ndarray,
                seq_len: int, pred_len: int, 
                original_n_nodes: int, 
                step_size: int = 1,
                supervision_policy: str = "masked"):
        """
        Args:
            raw_data_np (np.ndarray): 完整的 2N 维数据, 形状 (N_samples, T_total, 2N) 或 (T_total, 2N)。
            raw_mask_np (np.ndarray): 对应的观测掩码。
            seq_len (int): 输入序列长度 (历史)。
            pred_len (int): 预测序列长度 (未来)。
            step_size (int): 滑动窗口步长。
            original_n_nodes (int): 原始因果节点的数量 N (不是2N)。
            supervision_policy (str): 如何处理预测的掩码。
        """
        # 确保 N_features 维度存在
        if raw_data_np.ndim == 3:
            raw_data_np = np.expand_dims(raw_data_np, axis=-1)
            raw_mask_np = np.expand_dims(raw_mask_np, axis=-1)
        elif raw_data_np.ndim == 2:
            raw_data_np = np.expand_dims(raw_data_np, axis=(0, -1))
            raw_mask_np = np.expand_dims(raw_mask_np, axis=(0, -1))
        
        self.data = torch.from_numpy(raw_data_np).float() # (N_samples, T_total, 2N, 1)
        self.mask = torch.from_numpy(raw_mask_np).float()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step_size = step_size
        
        # 区分模型节点数和评估节点数
        self.n_nodes_for_model = self.data.shape[2]           # 模型处理的节点数，即 2N
        self.original_n_nodes_for_eval = original_n_nodes  # 用于评估的原始节点数，即 N
        
        if self.n_nodes_for_model != 2 * self.original_n_nodes_for_eval:
             raise ValueError("Adapter Dataset: Model nodes (2N) must be twice the original nodes for evaluation (N).")

        self.supervision_policy = supervision_policy

        self.sequences = []
        for sample_idx in range(self.data.shape[0]):
            current_sample_data = self.data[sample_idx] # (T_total, 2N, 1)
            current_sample_mask = self.mask[sample_idx]

            for i in range(0, current_sample_data.shape[0] - self.seq_len - self.pred_len + 1, self.step_size):
                x_history = current_sample_data[i : i + self.seq_len]  # (seq_len, 2N, 1)
                y_true = current_sample_data[i + self.seq_len : i + self.seq_len + self.pred_len] # (pred_len, 2N, 1)
                
                mask_x = current_sample_mask[i : i + self.seq_len]
                mask_y = current_sample_mask[i + self.seq_len : i + self.seq_len + self.pred_len]

                # original_x 和 original_y 现在只包含前 N 维
                original_x = x_history[:, :self.original_n_nodes_for_eval, :] # (seq_len, N, 1)
                original_y = y_true[:, :self.original_n_nodes_for_eval, :]   # (pred_len, N, 1)

                self.sequences.append((x_history, y_true, original_x, original_y, mask_x, mask_y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def load_data_from_causal_time_dataset(
    causal_time_dataset_obj: CausalTimeDataset, # 这是一个 CausalTimeDataset 的实例
    batch_size: int,
    seq_len: int,
    pred_len: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    step_size: int = 1,
    supervision_policy: str = "masked"
):
    """
    加载 CausalTimeDataset 数据并为 SDCFlow 训练适配。
    """
    # 获取 2N 维数据 ---
    data_for_model_np = causal_time_dataset_obj.X_tilde_for_model # (Sample_num, Time_step, 2N)
    mask_for_model_np = causal_time_dataset_obj.mask_for_model    # (Sample_num, Time_step, 2N)

    # 地面真实图 (N, N) 和原始节点数 N
    H_ground_truth_np = causal_time_dataset_obj.H_ground_truth_np 
    num_original_nodes = causal_time_dataset_obj.num_original_nodes_N

    full_sdcflow_dataset = SDCFlowAdapterDataset(
        raw_data_np=data_for_model_np, # 传入 2N 维数据
        raw_mask_np=mask_for_model_np,
        seq_len=seq_len,
        pred_len=pred_len,
        step_size=step_size,
        original_n_nodes=num_original_nodes, # 传入 N，而不是 2N
        supervision_policy=supervision_policy
    )

    # 数据集划分
    dataset_size = len(full_sdcflow_dataset)
    test_val_size = int((test_size + val_size) * dataset_size)
    train_size = dataset_size - test_val_size
    val_size_actual = int(val_size * dataset_size)
    test_size_actual = test_val_size - val_size_actual

    if train_size + val_size_actual + test_size_actual != dataset_size:
        train_size = dataset_size - (val_size_actual + test_size_actual)

    train_dataset, val_dataset, test_dataset = random_split(
        full_sdcflow_dataset, [train_size, val_size_actual, test_size_actual]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    return train_loader, test_loader, val_loader, data_for_model_np, mask_for_model_np, H_ground_truth_np
