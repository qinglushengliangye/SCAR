"""
创新点2：交互式特征提取 + 有监督对比学习 (Supervised Contrastive Learning, SupCon)
===================================================================================

【核心思想】
针对零样本场景下模型难以区分细粒度相似关系（如"创始人"与"联合创始人"）的难题，
本模块在特征投影空间中：
  - 极力拉近同类关系样本的距离
  - 推远语义高度相似但类别不同的"难负样本"对

不同于传统分类损失仅关注决策边界，本模块显式优化特征分布结构，迫使模型学到
更本质的语义差异，提升零样本场景下的泛化与抗干扰能力。

【与原始设计的适配】
原描述只谈"特征投影 + 拉近/推远"，未利用 GLiREL 的交互式编码能力，
也未定义"难负样本"。本实现做了三处修正：

1. 加入 Label-Aware Cross-Attention 层：让实体对表示与当前 batch 所有标签
   表示做一次交互，得到"标签感知"的对比表示，对应"交互式特征提取"。
2. 难负样本定义为 batch 内标签文本余弦相似度较高的负标签样本，通过在
   SupCon 分母中对此类负样本做乘性加权（w = 1 + β·max(0, cos)）显式放大
   其竞争压力。
3. 用线性 warmup 让对比损失在 warmup_steps 内从 0 增长到 loss_weight_max，
   避免训练早期破坏预训练 rel_rep 结构。

【数值稳定性】
- 投影表示 L2 归一化后用温度缩放（τ=0.1）
- logits 在 softmax 前做 clamp（±50）防止 AMP fp16 溢出
- 若 batch 内无有效正样本对，返回零损失但保留计算图
- NaN/Inf 检测：异常时退化为零损失，不干扰主任务
"""

from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger


class LabelAwareCrossAttention(nn.Module):
    """
    轻量单头 Cross-Attention：
        Q = rel_rep  [B, P, D]
        K,V = rel_type_rep  [B, L, D]
    输出：标签感知的实体对表示 [B, P, D]（residual + LayerNorm）

    参数量约 3·D² + D ≈ 1.77M（D=768）
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_size ** -0.5

    def forward(
        self,
        rel_rep: Tensor,          # [B, P, D]
        rel_type_rep: Tensor,     # [B, L, D]
        rel_type_mask: Tensor,    # [B, L]  (True = valid label)
    ) -> Tensor:
        q = self.q_proj(rel_rep)                    # [B, P, D]
        k = self.k_proj(rel_type_rep)               # [B, L, D]
        v = self.v_proj(rel_type_rep)               # [B, L, D]

        attn_logits = torch.einsum('BPD,BLD->BPL', q, k) * self.scale  # [B, P, L]

        if rel_type_mask is not None:
            inv_mask = ~rel_type_mask.unsqueeze(1)  # [B, 1, L]
            attn_logits = attn_logits.masked_fill(inv_mask, float('-inf'))

        attn_logits = attn_logits.clamp(min=-50.0, max=50.0)
        attn = F.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # 若整行被 mask，softmax 会产生 NaN
        attn = self.dropout(attn)

        ctx = torch.einsum('BPL,BLD->BPD', attn, v)  # [B, P, D]
        ctx = self.out_proj(ctx)
        out = self.layer_norm(rel_rep + ctx)         # residual
        return out


class ProjectionHead(nn.Module):
    """
    对比空间投影头：
        Linear(D, 2d_c) -> LayerNorm -> GELU -> Dropout -> Linear(2d_c, d_c)
    输出 L2 归一化。参数量 ≈ D·2d_c + 2d_c·d_c ≈ 230K。
    """

    def __init__(self, hidden_size: int, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class SupervisedContrastiveModule(nn.Module):
    """
    有监督对比学习模块（含难负加权与 warmup）

    输入：
      rel_rep          [B, P, D]    实体对表示（含 mask padding 位置）
      rel_type_rep     [B, L, D]    标签文本表示
      rel_type_mask    [B, L]       有效标签 mask
      rel_labels       [B, P]       标签 id（1-indexed，0 为无关系，-1 为 padding）
      class_to_ids     List[Dict]   batch 内每个样本的 {rel_text: local_id} 映射

    输出：
      {
        'supcon_loss': scalar (已乘以当前 warmup 权重)
        'raw_loss'   : scalar (未乘权重，调试用)
        'loss_weight': float
        'n_anchors'  : int
      }
    """

    def __init__(
        self,
        hidden_size: int,
        proj_dim: int = 128,
        temperature: float = 0.1,
        hard_neg_beta: float = 1.0,
        warmup_steps: int = 300,
        loss_weight_max: float = 0.05,
        dropout: float = 0.1,
        global_alignment: bool = True,
        cross_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.hard_neg_beta = hard_neg_beta
        self.warmup_steps = warmup_steps
        self.loss_weight_max = loss_weight_max
        self.global_alignment = global_alignment
        self.cross_attention = cross_attention

        self.cross_attn = LabelAwareCrossAttention(hidden_size, dropout)
        self.pair_proj = ProjectionHead(hidden_size, proj_dim, dropout)
        self.label_proj = ProjectionHead(hidden_size, proj_dim, dropout)

        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    @property
    def loss_weight(self) -> float:
        """warmup: 0 → loss_weight_max 线性增长；达到 warmup_steps 后稳定"""
        if self.warmup_steps <= 0:
            return self.loss_weight_max
        progress = min(1.0, self._step.item() / max(1, self.warmup_steps))
        return self.loss_weight_max * progress

    @staticmethod
    def _build_global_label_ids(
        class_to_ids_batch: List[Dict[str, int]],
    ):
        """
        将每个样本的本地标签 id 映射为跨 batch 统一的全局 id。
        返回：
          global_id_per_local : List[List[int]]  每个 batch 样本的 [L_i] 全局 id（按 local id 升序）
          num_global          : int              全局唯一标签数
        """
        text_to_global: Dict[str, int] = {}
        global_id_per_local: List[List[int]] = []
        for mapping in class_to_ids_batch:
            if mapping is None:
                global_id_per_local.append([])
                continue
            # 按 local id 升序排序，抵消 coref 特殊索引 (local 可能为 -50)
            # 只保留 local > 0 的正常标签（与 rel_labels>0 对齐）
            sorted_items = sorted(
                [(v, k) for k, v in mapping.items() if v > 0],
                key=lambda x: x[0],
            )
            labels_for_this = []
            # local_id 从 1 开始递增且连续；我们按位置填充
            max_local = sorted_items[-1][0] if sorted_items else 0
            # 用 list 占位 max_local 个位置（index = local_id - 1）
            placeholder = [-1] * max_local
            for local_id, text in sorted_items:
                if text not in text_to_global:
                    text_to_global[text] = len(text_to_global)
                placeholder[local_id - 1] = text_to_global[text]
            global_id_per_local.append(placeholder)
        return global_id_per_local, len(text_to_global)

    def forward(
        self,
        rel_rep: Tensor,                      # [B, P, D]
        rel_type_rep: Tensor,                 # [B, L, D]
        rel_type_mask: Tensor,                # [B, L]
        rel_labels: Optional[Tensor],         # [B, P]  1-indexed，0/-1 非正样本
        class_to_ids_batch: Optional[List[Dict[str, int]]] = None,
    ) -> dict:
        device = rel_rep.device
        zero = rel_rep.sum() * 0.0  # 保持计算图

        # 训练才推进 step 与计算损失（eval 时返回 0）
        if not self.training or rel_labels is None:
            return {
                'supcon_loss': zero,
                'raw_loss'   : zero,
                'loss_weight': 0.0,
                'n_anchors'  : 0,
            }

        self._step += 1

        B, P, D = rel_rep.shape
        L = rel_type_rep.shape[1]

        # 1) 交互式特征：让实体对表示感知当前 batch 的标签集合
        if self.cross_attention:
            interacted = self.cross_attn(rel_rep, rel_type_rep, rel_type_mask)  # [B, P, D]
        else:
            interacted = rel_rep

        # 2) 投影到对比空间
        z_pair = self.pair_proj(interacted)                                 # [B, P, d_c]
        z_label = self.label_proj(rel_type_rep)                             # [B, L, d_c]

        # 3) 收集 anchor：仅 rel_labels ∈ [1, L] 的位置为有效正标签样本
        #    同时排除 padding (-1) 与负样本对 (0)
        valid_anchor_mask = (rel_labels > 0) & (rel_labels <= L)            # [B, P]
        n_anchors = int(valid_anchor_mask.sum().item())
        if n_anchors < 2:
            # 无法构造正样本对
            return {
                'supcon_loss': zero,
                'raw_loss'   : zero,
                'loss_weight': self.loss_weight,
                'n_anchors'  : n_anchors,
            }

        # 4) 将 batch 内标签映射为全局 id，用于跨样本识别同类正样本
        if not self.global_alignment or class_to_ids_batch is None:
            global_ids_per_sample = [list(range(1, L + 1)) for _ in range(B)]
            num_global = L
        else:
            global_ids_per_sample, num_global = self._build_global_label_ids(class_to_ids_batch)

        # 5) 为每个 anchor 收集全局标签 id（用于正负样本划分）
        #    并收集 anchor 对应的标签嵌入（用于难负加权）
        anchor_pair_indices = []   # flat idx in [B*P]
        anchor_global_ids = []
        anchor_local_ids = []
        anchor_batch_ids = []
        anchor_positions = valid_anchor_mask.nonzero(as_tuple=False)  # [N_anchor, 2] (b, p)
        for (b, p) in anchor_positions.tolist():
            local_id = int(rel_labels[b, p].item())
            labels_map = global_ids_per_sample[b]
            if local_id - 1 >= len(labels_map):
                continue
            g_id = labels_map[local_id - 1]
            if g_id < 0:
                continue
            anchor_pair_indices.append(b * P + p)
            anchor_global_ids.append(g_id)
            anchor_local_ids.append(local_id - 1)
            anchor_batch_ids.append(b)

        if len(anchor_pair_indices) < 2:
            return {
                'supcon_loss': zero,
                'raw_loss'   : zero,
                'loss_weight': self.loss_weight,
                'n_anchors'  : len(anchor_pair_indices),
            }

        anchor_idx_tensor = torch.tensor(anchor_pair_indices, device=device, dtype=torch.long)
        anchor_global_tensor = torch.tensor(anchor_global_ids, device=device, dtype=torch.long)
        anchor_local_tensor = torch.tensor(anchor_local_ids, device=device, dtype=torch.long)
        anchor_batch_tensor = torch.tensor(anchor_batch_ids, device=device, dtype=torch.long)

        N = anchor_idx_tensor.size(0)
        if N < 2:
            return {
                'supcon_loss': zero,
                'raw_loss'   : zero,
                'loss_weight': self.loss_weight,
                'n_anchors'  : N,
            }

        # 6) 抽取 anchor 投影向量
        z_flat = z_pair.reshape(B * P, self.proj_dim)       # [B*P, d_c]
        z_anchors = z_flat[anchor_idx_tensor]               # [N, d_c]

        # 7) 抽取 anchor 对应的"当前样本所属标签"嵌入（用于难负加权）
        #    从各自样本的 z_label 中取第 (local_id-1) 行
        z_label_anchor = z_label[anchor_batch_tensor, anchor_local_tensor]  # [N, d_c]

        # 8) 计算 anchor-anchor 相似度矩阵（对比核心）
        sim = torch.matmul(z_anchors, z_anchors.t())        # [N, N]
        sim = sim / self.temperature
        sim = sim.clamp(min=-50.0, max=50.0)

        # 9) 构造 positive mask（同全局 id，但不含自身）
        pos_mask = anchor_global_tensor.unsqueeze(0).eq(anchor_global_tensor.unsqueeze(1))  # [N,N]
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        pos_mask = pos_mask & (~self_mask)

        has_pos = pos_mask.any(dim=1)                       # [N]
        if has_pos.sum().item() < 1:
            return {
                'supcon_loss': zero,
                'raw_loss'   : zero,
                'loss_weight': self.loss_weight,
                'n_anchors'  : int(N),
            }

        # 10) 难负样本加权：neg_mask = (!pos_mask & !self_mask)；对 neg 计算
        #     anchor 标签嵌入的余弦相似度，越相似越"难"
        neg_mask = (~pos_mask) & (~self_mask)               # [N, N]

        # 标签语义相似度（在对比空间中 L2 归一化向量的点积即 cos）
        label_sim = torch.matmul(z_label_anchor, z_label_anchor.t())  # [N, N]
        label_sim = label_sim.clamp(min=-1.0, max=1.0)
        hard_weight = 1.0 + self.hard_neg_beta * label_sim.clamp(min=0.0)  # [N, N] ≥ 1

        # logits 减去每行最大值（数值稳定性）
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()
        exp_sim = torch.exp(sim_stable)

        # 对 negative 乘加权；positive 和 self 不加权
        weights = torch.ones_like(exp_sim)
        weights = torch.where(neg_mask, hard_weight, weights)

        # 分母：所有非自身项（positive + negative）加权后求和
        valid_mask = ~self_mask                              # [N, N]
        denom = (exp_sim * weights * valid_mask.float()).sum(dim=1) + 1e-8  # [N]

        # 分子：每个 anchor 对其所有 positive 的 log prob 求均值
        log_prob = sim_stable - torch.log(denom).unsqueeze(1)  # [N, N]
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0) # [N]
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count  # [N]

        # 仅对含 positive 的 anchor 取损失（其余为 0，不参与平均）
        loss_per_anchor = -mean_log_prob_pos                 # [N]
        loss_per_anchor = torch.where(has_pos, loss_per_anchor, torch.zeros_like(loss_per_anchor))
        n_valid = has_pos.float().sum().clamp(min=1.0)
        raw_loss = loss_per_anchor.sum() / n_valid

        # 异常检测：NaN/Inf 时退化为零损失
        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            logger.warning(
                f"[SupCon] step={self._step.item()} raw_loss NaN/Inf, fallback to zero"
            )
            raw_loss = zero

        lw = self.loss_weight
        supcon_loss = lw * raw_loss

        # 周期性日志
        if self.training and (self._step.item() % 500 == 1):
            logger.info(
                f"[SupCon] step={self._step.item()} "
                f"loss_weight={lw:.4f} raw_loss={raw_loss.item():.4f} "
                f"n_anchors={N} n_with_pos={int(has_pos.sum().item())}"
            )

        return {
            'supcon_loss': supcon_loss,
            'raw_loss'   : raw_loss,
            'loss_weight': lw,
            'n_anchors'  : int(N),
        }
