import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tqdm
from copy import deepcopy
import os
import math
import random

from torchdiffeq import odeint_adjoint as odeint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from dataloader import SDCFlowAdapterDataset, DataLoader, load_data_from_causal_time_dataset, CausalTimeDataset



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def h_DAG(adj_matrix):
    if adj_matrix.dim() != 3 or adj_matrix.shape[-1] != adj_matrix.shape[-2]:
        # 只处理(B, N, N)的方阵
        return torch.tensor(0.0, device=adj_matrix.device)
    # 使用einsum进行高效的对角线求和，避免bmm
    # A^2的对角线: sum_j A_ij * A_ji
    trace_A_sq = torch.einsum('bii->b', torch.einsum('bij,bjk->bik', adj_matrix, adj_matrix)).sum()
    # A^3的对角线: sum_{j,k} A_ij * A_jk * A_ki
    A_sq = torch.bmm(adj_matrix, adj_matrix)
    trace_A_cube = torch.einsum('bii->b', torch.bmm(A_sq, adj_matrix)).sum()
    
    # 对角线元素惩罚
    diag_penalty = torch.einsum('bii->b', adj_matrix).abs().sum()
    
    # 结合两种循环惩罚
    cycle_penalty = trace_A_sq.abs() + trace_A_cube.abs()
    return diag_penalty * 0.5 + cycle_penalty * 0.1

def calc_and_log_metrics(predicted_graph, true_graph, log_writer, global_step):
    y_true, y_scores = true_graph.cpu().numpy().flatten(), predicted_graph.cpu().numpy().flatten()
    diag_mask = ~np.eye(true_graph.shape[0], dtype=bool).flatten()
    y_true, y_scores = y_true[diag_mask], y_scores[diag_mask]
    if len(np.unique(y_true)) < 2: return 0.0, 0.0
    try: auroc = roc_auc_score(y_true, y_scores)
    except ValueError: auroc = 0.5
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    if log_writer: print(f"Metrics - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f} at step {global_step}")
    return auroc, auprc


class HybridAttentionEncoder(nn.Module):
    """
    SDCFlow: 全局-局部混合编码器
    - 使用节点级RNN捕捉局部动态。
    - 使用自注意力机制捕捉全局交互。
    - 输出丰富的节点嵌入和全局摘要。
    """
    def __init__(self, n_features, n_nodes, per_node_hidden_dim, global_summary_dim, num_heads=4):
        super().__init__()
        print("SDCFlow V2: Initializing HybridAttentionEncoder.")
        self.n_nodes = n_nodes
        # 1. 节点级RNN，独立处理每个节点的时序特征
        self.node_level_rnn = nn.GRU(n_features, per_node_hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        
        # 2. 多头自注意力机制，用于节点间的交互
        self.attention = nn.MultiheadAttention(embed_dim=per_node_hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.layer_norm1 = nn.LayerNorm(per_node_hidden_dim)
        
        # 3. 前馈网络增强表示
        self.ffn = nn.Sequential(
            nn.Linear(per_node_hidden_dim, per_node_hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(per_node_hidden_dim * 4, per_node_hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(per_node_hidden_dim)
        
        # 4. 全局摘要生成器
        self.global_aggregator = nn.GRU(n_nodes * per_node_hidden_dim, global_summary_dim, batch_first=True)

    def forward(self, x_history):
        # x_history: (B, T_in, N, F)
        B, T_in, N, F = x_history.shape
        
        # Reshape for node-level RNN: (B*N, T_in, F)
        x_flat = x_history.permute(0, 2, 1, 3).reshape(B * N, T_in, F)
        
        # 1. 局部动态编码
        _, h_nodes_local = self.node_level_rnn(x_flat)
        h_nodes_local = h_nodes_local[-1].view(B, N, -1) # (B, N, H_node)
        
        # 2. 全局交互编码
        # Add a residual connection and layer norm before attention
        attn_input = self.layer_norm1(h_nodes_local)
        # Attention expects (B, N, H), which is our current shape
        h_nodes_global, _ = self.attention(attn_input, attn_input, attn_input)
        # Add residual connection
        h_fused = h_nodes_local + h_nodes_global
        
        # 3. FFN
        ffn_input = self.layer_norm2(h_fused)
        final_node_embeddings = h_fused + self.ffn(ffn_input) # (B, N, H_node)
        
        # 4. 全局摘要生成
        # (B, N, H_node) -> (B, N * H_node) -> (B, 1, N * H_node) for GRU
        global_input = final_node_embeddings.reshape(B, 1, -1)
        _, global_summary = self.global_aggregator(global_input)
        
        return final_node_embeddings, global_summary.squeeze(0) # (B, N, H_node), (B, H_global)


class StructuredODEFunc(nn.Module):
    """
    结构化解耦的ODE函数
    - 将图隐空间拆分为 z_intra 和 z_inter。
    - 为每个隐变量设计独立的ODE函数。
    - 引入弱化的跨空间交互。
    """
    def __init__(self, data_latent_dim, intra_latent_dim, inter_latent_dim, interaction_strength=0.1):
        super().__init__()
        print("SDCFlow V2: Initializing StructuredODEFunc.")
        self.data_dim = data_latent_dim
        self.intra_dim = intra_latent_dim
        self.inter_dim = inter_latent_dim
        self.interaction_strength = interaction_strength

        # 独立的动力学网络
        self.f_data = self._build_net(data_latent_dim, data_latent_dim)
        self.f_intra = self._build_net(intra_latent_dim, intra_latent_dim)
        self.f_inter = self._build_net(inter_latent_dim, inter_latent_dim)
        
        # 弱化的跨空间交互层
        if self.interaction_strength > 0:
            self.intra_to_inter = nn.Linear(intra_latent_dim, inter_latent_dim, bias=False)
            self.inter_to_intra = nn.Linear(inter_latent_dim, intra_latent_dim, bias=False)

    def _build_net(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim + 1, (input_dim + 1) * 2),
            nn.Tanh(),
            nn.Linear((input_dim + 1) * 2, output_dim)
        )

    def forward(self, t, z):
        # z: (B, D_data + D_intra + D_inter)
        z_data = z[:, :self.data_dim]
        z_intra = z[:, self.data_dim : self.data_dim + self.intra_dim]
        z_inter = z[:, self.data_dim + self.intra_dim :]
        
        t_exp = t.expand(z.size(0), 1)
        
        # 计算主导动态
        dz_data = self.f_data(torch.cat([z_data, t_exp], dim=-1))
        dz_intra = self.f_intra(torch.cat([z_intra, t_exp], dim=-1))
        dz_inter = self.f_inter(torch.cat([z_inter, t_exp], dim=-1))
        
        # 添加弱交互
        if self.interaction_strength > 0:
            dz_intra = dz_intra + self.interaction_strength * self.inter_to_intra(z_inter)
            dz_inter = dz_inter + self.interaction_strength * self.intra_to_inter(z_intra)
            
        return torch.cat([dz_data, dz_intra, dz_inter], dim=-1)

class RelationalReasonerV2(nn.Module):
    """
      关系推理器
    - 输入事解耦的 z_intra 和 z_inter。
    """
    def __init__(self, node_embedding_dim, intra_latent_dim, inter_latent_dim, num_lags):
        super().__init__()
        print("SDCFlow V2: Initializing RelationalReasonerV2.")
        self.num_lags = num_lags
        self.lag_embedding = nn.Embedding(num_lags, 8)

        # 独立的推理器
        self.instantaneous_reasoner = self._build_reasoner_net(node_embedding_dim * 2 + intra_latent_dim)
        self.lagged_reasoner = self._build_reasoner_net(node_embedding_dim * 2 + inter_latent_dim + 8)

    def _build_reasoner_net(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, node_embeddings, z_intra_t, z_inter_t):
        B, N, D_node = node_embeddings.shape
        
        # 预计算节点对表示
        h_i = node_embeddings.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = node_embeddings.unsqueeze(1).expand(-1, N, -1, -1)
        pairs = torch.cat([h_i, h_j], dim=-1).reshape(B, N * N, -1)

        # Intra-slice (W_t) 推理
        z_intra_exp = z_intra_t.unsqueeze(1).expand(-1, N * N, -1)
        inst_input = torch.cat([pairs, z_intra_exp], dim=-1)
        W_logits = self.instantaneous_reasoner(inst_input).view(B, N, N)

        # Inter-slice (A_t) 推理 (向量化)
        z_inter_exp_lags = z_inter_t.unsqueeze(1).expand(-1, self.num_lags, -1).unsqueeze(2).expand(-1, -1, N*N, -1)
        pairs_expanded = pairs.unsqueeze(1).expand(-1, self.num_lags, -1, -1)
        
        lag_indices = torch.arange(self.num_lags, device=node_embeddings.device)
        embedded_lags = self.lag_embedding(lag_indices).view(1, self.num_lags, 1, -1).expand(B, -1, N * N, -1)
        
        lag_input = torch.cat([pairs_expanded, z_inter_exp_lags, embedded_lags], dim=-1)
        A_lags_logits = self.lagged_reasoner(lag_input).view(B, self.num_lags, N, N)

        mask_diag = 1.0 - torch.eye(N, device=node_embeddings.device)
        W_t = torch.sigmoid(W_logits) * mask_diag
        A_lags_t = torch.sigmoid(A_lags_logits) * mask_diag
        return W_t, A_lags_t

class DynamicGraphForecasterV2(nn.Module):
    """
    动态图预测器
    """
    def __init__(self, n_features, node_embedding_dim, data_latent_dim):
        super().__init__()
        print("SDCFlow V2: Initializing DynamicGraphForecasterV2 (FIXED).")
        
        combined_dim = node_embedding_dim + data_latent_dim
        
        self.fusion_gate = nn.Linear(combined_dim, node_embedding_dim)
        self.fusion_transform = nn.Linear(combined_dim, node_embedding_dim)
        
        self.inst_projection = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.lag_projection = nn.Linear(n_features, node_embedding_dim)
        self.output_decoder = nn.Sequential(
            nn.Linear(node_embedding_dim, node_embedding_dim * 2), nn.ReLU(),
            nn.Linear(node_embedding_dim * 2, n_features)
        )
        
    def forward(self, final_node_embeddings, x_history, z_data_t, W_t, A_lags_t):
        B, N, D_node = final_node_embeddings.shape
        
        # 融合节点静态嵌入和数据动态隐状态
        z_data_exp = z_data_t.unsqueeze(1).expand(-1, N, -1)
        
        combined_repr = torch.cat([final_node_embeddings, z_data_exp], dim=-1)
        
        gate = torch.sigmoid(self.fusion_gate(combined_repr))
        transformed_repr = F.relu(self.fusion_transform(combined_repr))
        current_node_state = (1 - gate) * final_node_embeddings + gate * transformed_repr
        
        # 消息传递
        inst_messages = self.inst_projection(current_node_state)
        inst_agg = torch.bmm(W_t, inst_messages)

        num_lags = A_lags_t.shape[1]
        history_len = x_history.shape[1]
        
        lag_agg = torch.zeros_like(current_node_state)
        if num_lags > 0 and history_len > 0:
            usable_lags = min(num_lags, history_len)
            relevant_history = x_history[:, -usable_lags:]
            lag_messages = self.lag_projection(relevant_history)
            relevant_A = A_lags_t[:, :usable_lags]
            lag_agg = torch.einsum('blik,blkj->bij', relevant_A, lag_messages)

        final_node_repr = current_node_state + inst_agg + lag_agg
        return self.output_decoder(final_node_repr)


class SDCFlowModelV2(nn.Module):
    """
    SDCFlow 主模型
    """
    def __init__(self, args, n_nodes, n_features, input_step, pred_step, num_lags, log_writer=None):
        super().__init__()
        self.args, self.log, self.device = args, log_writer, args.device
        self.n_nodes, self.n_features = n_nodes, n_features
        self.input_step, self.pred_step, self.num_lags = input_step, pred_step, num_lags
        
        total_latent_dim = args.data_latent_dim + args.intra_latent_dim + args.inter_latent_dim
        
        self.encoder = HybridAttentionEncoder(
            n_features, n_nodes, args.node_embedding_dim, args.global_summary_dim
        ).to(self.device)
        self.summary_to_latent = nn.Linear(args.global_summary_dim, total_latent_dim * 2).to(self.device)
        
        self.ode_func = StructuredODEFunc(
            args.data_latent_dim, args.intra_latent_dim, args.inter_latent_dim, args.interaction_strength
        ).to(self.device)
        
        self.reasoner = RelationalReasonerV2(
            args.node_embedding_dim, args.intra_latent_dim, args.inter_latent_dim, num_lags
        ).to(self.device)

        # 步骤3: 将正确的维度传入 Forecaster
        self.forecaster = DynamicGraphForecasterV2(
            n_features, args.node_embedding_dim, args.data_latent_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)

        self.log_var_pred = nn.Parameter(torch.tensor([0.0])).to(self.device)
        self.log_var_kl = nn.Parameter(torch.tensor([0.0])).to(self.device)
        self.log_var_dag = nn.Parameter(torch.tensor([0.0])).to(self.device)
        self.log_var_sparse = nn.Parameter(torch.tensor([0.0])).to(self.device)

    def forward(self, x_history_batch):
        # 1. 编码
        final_node_embeddings, global_summary = self.encoder(x_history_batch)
        
        # 2. 初始化隐状态
        mu_logvar_z0 = self.summary_to_latent(global_summary)
        latent_dims = self.args.data_latent_dim + self.args.intra_latent_dim + self.args.inter_latent_dim
        mu_z0, logvar_z0 = mu_logvar_z0[:, :latent_dims], mu_logvar_z0[:, latent_dims:]
        z0 = mu_z0 + torch.randn_like(mu_z0) * torch.exp(0.5 * logvar_z0) if self.training else mu_z0
        
        # 3. ODE 演化
        t_span = torch.linspace(0, self.pred_step - 1, self.pred_step).to(self.device)
        z_trajectory = odeint(self.ode_func, z0, t_span, method='dopri5', rtol=1e-3, atol=1e-4)
        
        # 4. 解码与预测
        data_pred_all, W_t_all, A_lags_t_all = [], [], []
        
        data_dim = self.args.data_latent_dim
        intra_dim = self.args.intra_latent_dim
        
        for t_step in range(self.pred_step):
            z_t = z_trajectory[t_step]
            z_data_t = z_t[:, :data_dim]
            z_intra_t = z_t[:, data_dim : data_dim + intra_dim]
            z_inter_t = z_t[:, data_dim + intra_dim :]
            
            W_t, A_lags_t = self.reasoner(final_node_embeddings, z_intra_t, z_inter_t)
            y_pred_t = self.forecaster(final_node_embeddings, x_history_batch, z_data_t, W_t, A_lags_t)
            
            data_pred_all.append(y_pred_t)
            W_t_all.append(W_t)
            A_lags_t_all.append(A_lags_t)
            
        return torch.stack(data_pred_all, 1), mu_z0, logvar_z0, torch.stack(W_t_all, 0), torch.stack(A_lags_t_all, 0)

    def calculate_losses(self, data_pred_all, y_true_batch, mask_y_batch,
                         mu, logvar, W_t_trajectory, A_lags_t_trajectory):
        losses = {}
        batch_size = mu.shape[0]

        # 预测损失
        mse_loss_raw = F.mse_loss(data_pred_all * mask_y_batch, y_true_batch * mask_y_batch, reduction='sum') / (torch.sum(mask_y_batch) + 1e-6)
        losses['L_pred'] = 0.5 * torch.exp(-self.log_var_pred) * mse_loss_raw + 0.5 * self.log_var_pred

        # KL 损失
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        losses['L_KL'] = self.args.lambda_kl * (torch.exp(-self.log_var_kl) * kl_div + self.log_var_kl)
        
        # DAG 损失 (向量化)
        T_pred, B, N, _ = W_t_trajectory.shape
        W_flat_for_dag = W_t_trajectory.reshape(T_pred * B, N, N)
        dag_loss_sum = h_DAG(W_flat_for_dag)
        losses['L_DAG'] = self.args.lambda_dag * (torch.exp(-self.log_var_dag) * dag_loss_sum / (T_pred * B) + self.log_var_dag)

        # 稀疏性损失
        sparse_loss = torch.mean(W_t_trajectory.abs()) + self.args.lambda_sparse_lag * torch.mean(A_lags_t_trajectory.abs())
        losses['L_sparse'] = self.args.lambda_sparse * (torch.exp(-self.log_var_sparse) * sparse_loss + self.log_var_sparse)
        
        losses['total_loss'] = sum(losses.values())
        losses['mse_pred_raw'] = mse_loss_raw.item()
        return losses

    def train_epoch(self, train_loader, epoch_i):
        self.train()
        total_loss_epoch = 0
        # --- 这里可以实现分阶段优化逻辑 ---
        # if epoch_i < WARMUP_EPOCHS: # 预热阶段
        #     # 调整损失权重
        # elif epoch_i < ALTERNATING_EPOCHS: # 交替优化阶段
        #     # 实现两个optimizer分别step
        # else: # 联合微调
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch_i+1}/{self.args.total_epoch}", leave=False)
        for x_history_batch, y_true_batch, _, _, _, mask_y_batch in pbar:
            x_history_batch, y_true_batch, mask_y_batch = [t.to(self.device) for t in (x_history_batch, y_true_batch, mask_y_batch)]
            
            self.optimizer.zero_grad()
            data_pred, mu, logvar, W_traj, A_traj = self(x_history_batch)
            losses = self.calculate_losses(data_pred, y_true_batch, mask_y_batch, mu, logvar, W_traj, A_traj)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip_norm)
            self.optimizer.step()

            total_loss_epoch += losses['total_loss'].item()
            pbar.set_postfix({'Loss': f'{losses["total_loss"].item():.4f}', 'MSE': f'{losses["mse_pred_raw"]:.4f}'})
        
        avg_loss = total_loss_epoch / len(train_loader)
        self.scheduler.step(avg_loss) # 使用ReduceLROnPlateau

    def get_learned_H_graph(self, data_loader):
        """
        在2N维数据上运行模型，但只提取和平均左上角 N x N 的部分。
        并在评估前对小值进行阈值处理。
        """
        self.eval()
        
        # 从模型自身获取 N 和 2N
        n_nodes_model = self.n_nodes # 这是 2N
        n_nodes_original = n_nodes_model // 2 # 这是 N

        avg_A_lag1_H_part = torch.zeros(n_nodes_original, n_nodes_original, device=self.device)
        total_batches = 0
        with torch.no_grad():
            # data_loader 现在提供的是 2N 维的数据
            for x, _, _, _, _, _ in data_loader:
                # 模型输出 2N x 2N 的图
                _, _, _, _, A_traj = self(x.to(self.device)) # A_traj shape: (T_pred, B, L, 2N, 2N)
                
                # 对 T_pred 和 B 维度取平均
                avg_A_traj = torch.mean(A_traj, dim=[0, 1]) # (L, 2N, 2N)
                
                # 只取 lag=1 的图 (索引为0)
                avg_A_lag1 = avg_A_traj[0] # (2N, 2N)

                H_part = avg_A_lag1[:n_nodes_original, :n_nodes_original]
                
                avg_A_lag1_H_part += H_part
                total_batches += 1
                
        final_H_graph = avg_A_lag1_H_part / total_batches if total_batches > 0 else avg_A_lag1_H_part

        return final_H_graph

    def run_training(self, train_loader, val_loader, true_graph_H_np):
        true_graph_H_dev = torch.from_numpy(true_graph_H_np).float().to(self.device)
        print(f"\n--- Starting SDCFlow V2 Model Training (2N input) on {self.device} ---")
        for epoch_i in range(self.args.total_epoch):
            self.train_epoch(train_loader, epoch_i)
            if self.log:
                learned_H_graph = self.get_learned_H_graph(val_loader)
                # 用预测的 H 和真实的 H 计算指标
                calc_and_log_metrics(learned_H_graph, true_graph_H_dev, self.log, epoch_i)
        print("\nTraining finished. Performing final graph inference...")
        return self.get_learned_H_graph(train_loader)



if __name__ == "__main__":
    from dataloader import SDCFlowAdapterDataset, DataLoader, load_data_from_causal_time_dataset, CausalTimeDataset
    
    SEED = 3407
    seed_everything(SEED)

    class Args:
        node_embedding_dim = 512
        global_summary_dim = 2048
        data_latent_dim = 512
        intra_latent_dim = 512
        inter_latent_dim = 512
        interaction_strength = 0.1
        
        # 损失函数权重
        lambda_kl = 0.1
        lambda_dag = 5.0
        lambda_sparse = 5.0
        lambda_sparse_lag = 5.0 



        # 训练参数
        lr = 5e-4
        #lr = 1e-3
        total_epoch = 20 
        batch_size = 64
        grad_clip_norm = 1.0

        # 数据 & 模型时间线
        input_step, pred_step, num_lags = 10, 3, 1
        
        # 系统
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_dir_path = "./pm25/" 
        data_file, graph_file = "data.npy", "graph.npy"
        
    args = Args()
    class MockLog:
        def add_scalar(self, *args, **kwargs): pass
    log_writer = MockLog()

    # 1. 加载数据
    dataset = CausalTimeDataset(os.path.join(args.data_dir_path, args.data_file), os.path.join(args.data_dir_path, args.graph_file))
    train_loader, _, val_loader, _, _, true_graph_H = load_data_from_causal_time_dataset(dataset, args.batch_size, args.input_step, args.pred_step)

    # 2. 获取正确的节点数
    # n_nodes_for_model 应该是 2N
    # n_features 保持不变
    _, _, n_nodes_for_model, n_features = next(iter(train_loader))[0].shape # [0] 是 x_history

    # 3. 实例化模型
    # 传入 2N 作为模型的节点数
    model = SDCFlowModelV2(args, n_nodes_for_model, n_features, args.input_step, args.pred_step, args.num_lags, log_writer).to(args.device)

    # 4. 训练模型
    final_graph = model.run_training(train_loader, val_loader, true_graph_H)

    print("--- SDCFlow V2 Model Training Finished ---")
    print(f"Final learned graph (Lag 1) shape: {final_graph.shape}\n{final_graph[:5, :5].cpu().numpy()}")
