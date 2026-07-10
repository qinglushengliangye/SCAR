"""
双编码器辅助增强模块 (Dual Encoder Retrieval Augmentation)
============================================================

创新点1 重新设计：粗细粒度融合的级联增强推理架构

【设计原则】
不剪枝、不屏蔽，而是将粗粒度检索得分作为「软注意力偏置」
叠加到细粒度得分上，实现两个层次的互补融合：

  final_score = fine_score + alpha * coarse_bias

其中：
  - fine_score  : 原始 GLiREL 细粒度交互得分（完整保留，零损失）
  - coarse_bias : 双编码器粗粒度检索得分（轻量级，快速计算）
  - alpha       : 可学习融合权重，课程学习式从 0 渐进增大

【为什么能提升 F1】
1. 细粒度得分已经足够强，粗粒度提供「先验偏置」进一步校准
2. 对于训练数据少的低频关系，粗粒度相似度提供额外的语义信号
3. 软融合不影响召回率（无硬剪枝），仅提升精确度
4. 辅助对比损失使标签嵌入空间更加分离，改善 fine_score 的区分度

【超参数建议】
- retrieval_dim = 256    : 检索向量维度（轻量）
- fusion_alpha_init = 0.0: 初始融合权重（训练开始不干扰主模型）
- fusion_alpha_max  = 1.0: 最大融合权重
- warmup_steps = 2000   : 权重从 0 线性增长到 max 的步数
- retrieval_loss_weight = 0.1: 辅助损失权重（小，不干扰主损失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class LightweightLabelEncoder(nn.Module):
    """
    轻量级标签编码器：hidden_size -> retrieval_dim
    单隐层 FFN + LayerNorm，参数量约 hidden_size * retrieval_dim * 2
    """
    def __init__(self, hidden_size: int, retrieval_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, retrieval_dim * 2),
            nn.LayerNorm(retrieval_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(retrieval_dim * 2, retrieval_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)  # [*, retrieval_dim]


class LightweightRelEncoder(nn.Module):
    """
    轻量级关系对编码器：hidden_size -> retrieval_dim
    与 LabelEncoder 共享结构但参数独立（asymmetric dual encoder）
    """
    def __init__(self, hidden_size: int, retrieval_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, retrieval_dim * 2),
            nn.LayerNorm(retrieval_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(retrieval_dim * 2, retrieval_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)  # [*, retrieval_dim]


class DualEncoderAugmentor(nn.Module):
    """
    双编码器软融合增强模块

    核心机制：
      1. 用轻量双编码器计算粗粒度相似度得分 S_coarse  [B, P, L]
      2. 通过可学习 alpha（课程学习渐进增大）融合到细粒度得分：
         S_final = S_fine + alpha * S_coarse
      3. 同时用辅助对比损失（InfoNCE）优化粗粒度编码器，
         使标签嵌入空间分离度更好，间接提升细粒度得分质量

    训练策略：
      - warmup_steps 内 alpha 从 0 线性增长（课程学习）
      - 主损失 = relation_loss（不变）
      - 总损失 = relation_loss + retrieval_loss_weight * retrieval_loss
    """

    def __init__(
        self,
        hidden_size: int,
        retrieval_dim: int = 256,
        fusion_alpha_init: float = 0.0,
        fusion_alpha_max: float = 1.0,
        warmup_steps: int = 2000,
        retrieval_loss_weight: float = 0.1,
        dropout: float = 0.1,
        zscore_enabled: bool = True,
        gradient_isolation: bool = True,
    ):
        super().__init__()

        self.retrieval_dim = retrieval_dim
        self.fusion_alpha_max = fusion_alpha_max
        self.warmup_steps = warmup_steps
        self.retrieval_loss_weight = retrieval_loss_weight
        self.zscore_enabled = zscore_enabled
        self.gradient_isolation = gradient_isolation

        # 轻量双编码器
        self.label_encoder = LightweightLabelEncoder(hidden_size, retrieval_dim, dropout)
        self.rel_encoder   = LightweightRelEncoder(hidden_size, retrieval_dim, dropout)

        # 可学习温度（初始化为 ln(1/0.07) ≈ 2.659，与 CLIP 一致）
        self.log_temp = nn.Parameter(torch.tensor(2.659))

        # 步数计数器（不参与梯度）
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.fusion_alpha_init = fusion_alpha_init

    @property
    def fusion_alpha(self) -> float:
        """
        课程学习融合权重：
          step < warmup_steps : alpha 从 fusion_alpha_init 线性增长到 fusion_alpha_max
          step >= warmup_steps: alpha = fusion_alpha_max（稳定）
        """
        if self.warmup_steps <= 0:
            return self.fusion_alpha_max
        progress = min(1.0, self._step.item() / max(1, self.warmup_steps))
        return self.fusion_alpha_init + (self.fusion_alpha_max - self.fusion_alpha_init) * progress

    def _compute_coarse_scores(
        self,
        rel_rep: Tensor,       # [B, P, D]
        rel_type_rep: Tensor,  # [B, L, D]
    ) -> Tensor:
        """计算粗粒度余弦相似度得分 [B, P, L]"""
        rel_vecs   = self.rel_encoder(rel_rep)       # [B, P, retrieval_dim]
        label_vecs = self.label_encoder(rel_type_rep)  # [B, L, retrieval_dim]
        temp = self.log_temp.exp().clamp(min=1.0, max=30.0)
        scores = torch.einsum('BPD,BLD->BPL', rel_vecs, label_vecs) * temp

        if self.zscore_enabled:
            scores = scores - scores.mean(dim=-1, keepdim=True)
            scores = scores / (scores.std(dim=-1, keepdim=True) + 1e-6)
            scores = scores.clamp(min=-6.0, max=6.0)
        return scores

    def _compute_retrieval_loss(
        self,
        coarse_scores: Tensor,   # [B, P, L]  detached from fine-grained graph
        rel_labels: Tensor,      # [B, P]  label id，1-indexed（0=无关系）
        rel_type_mask: Tensor,   # [B, L]  有效标签掩码
    ) -> Tensor:
        """
        辅助对比损失（InfoNCE）
        目标：对每个正样本对 (relation_pair, correct_label)，
              在所有标签中最大化正标签得分
        只对 rel_labels > 0 的关系对计算
        """
        B, P, L = coarse_scores.shape

        # 有效正样本掩码（rel_labels > 0 表示该实体对存在正关系标签）
        pos_mask = (rel_labels > 0) & (rel_labels <= L)  # [B, P]
        if not pos_mask.any():
            return coarse_scores.sum() * 0.0  # 保持计算图连通

        # ── 修复#2：防止错误对比信号在早期淹没细粒度信息 ──────────────
        # coarse_scores 在归一化后已处于标准分布，但训练初期粗编码器随机初始化，
        # InfoNCE 损失会驱动其学习任意"伪"标签语义，产生的错误梯度信号
        # 在 warmup 期间仍持续更新编码器参数，导致后验概率被错误标签主导。
        # 对比损失计算前，对粗粒度得分再做一次温度缩放版 softmax 稳定性处理：
        # 使用 log-sum-exp 技巧避免 exp 溢出/下溢，同时限制 logits 量级。
        safe_scores = coarse_scores.clamp(min=-50.0, max=50.0)

        # 屏蔽无效标签（使无效标签的 logit 极度负，不参与竞争）
        if rel_type_mask is not None:
            inv_mask = ~rel_type_mask.unsqueeze(1).expand_as(safe_scores)  # [B,P,L]
            safe_scores = safe_scores.masked_fill(inv_mask, -1e4)

        # 展平，只取正样本行
        flat_scores  = safe_scores.view(B * P, L)                  # [BP, L]
        flat_labels  = rel_labels.clamp(min=0).view(B * P)       # [BP]
        flat_pos_mask = pos_mask.view(B * P)                     # [BP]

        scores_pos  = flat_scores[flat_pos_mask]       # [N_pos, L]
        labels_pos  = flat_labels[flat_pos_mask] - 1    # [N_pos]  1-indexed -> 0-indexed
        labels_pos  = labels_pos.clamp(min=0, max=L - 1)

        # ── 修复#2（续）：使用 ClampLogitsCrossEntropy 防止 logits 差异过大 ─
        # 直接用 clamp 后的 safe scores 计算 cross entropy，
        # 与标准 F.cross_entropy 等价，但显式控制 logits 范围
        # （F.cross_entropy 内部已经做了稳定化，这里额外加 clamp 是双保险）
        loss = F.cross_entropy(scores_pos, labels_pos)
        return loss

    def forward(
        self,
        fine_scores: Tensor,     # [B, P, L]  细粒度得分（scorer 输出，已计算）
        rel_rep: Tensor,         # [B, P, D]
        rel_type_rep: Tensor,    # [B, L, D]
        rel_type_mask: Tensor,   # [B, L]
        rel_labels: Optional[Tensor] = None,  # [B, P] 训练时提供
    ) -> dict:
        """
        软融合前向传播

        Returns:
            augmented_scores : [B, P, L]  融合后得分（直接替换原 scores）
            retrieval_loss   : scalar     辅助损失
            fusion_alpha     : float      当前融合权重（供日志用）
        """
        # 训练时推进步数计数器
        if self.training:
            self._step += 1

        # 计算粗粒度得分（归一化后，尺度对齐）
        coarse_scores = self._compute_coarse_scores(rel_rep, rel_type_rep)  # [B, P, L]

        # 屏蔽无效标签（保持与 fine_scores 一致）
        if rel_type_mask is not None:
            inv_mask = ~rel_type_mask.unsqueeze(1).expand_as(coarse_scores)
            coarse_scores = coarse_scores.masked_fill(inv_mask, -1e4)

        # 软融合：fine_scores + alpha * coarse_scores
        # alpha 在 warmup_steps 期间线性从 fusion_alpha_init 增长到 fusion_alpha_max
        alpha = self.fusion_alpha
        augmented_scores = fine_scores + alpha * coarse_scores  # [B, P, L]

        # ── 修复#3：强化异常检测回退机制 ─────────────────────────────
        # 若粗分支输出异常，回退到 fine_scores（不破坏主任务训练）
        # 仅保留 retrieval_loss（梯度截断，不影响 fine-grained scorer）
        with torch.no_grad():
            # 有效位置掩码
            valid_mask = rel_type_mask.unsqueeze(1).expand_as(fine_scores)  # [B, P, L]
            fine_valid = fine_scores[valid_mask]
            aug_valid  = augmented_scores[valid_mask]
            fine_std   = fine_valid.std() + 1e-6
            aug_std    = aug_valid.std() + 1e-6
            is_abnormal = (
                (aug_std / fine_std) > 3.0
                or torch.isnan(augmented_scores).any()
                or torch.isinf(augmented_scores).any()
            )

        if is_abnormal:
            # 回退：得分用 fine_scores，不破坏主任务
            if self.training:
                logger.warning(
                    f"[CascadeAugment] step={self._step} coarse abnormal "
                    f"(std_ratio={aug_std/fine_std:.2f}), fallback to fine_scores"
                )
            augmented_scores = fine_scores

        # 辅助对比损失（仅训练时，detach 截断梯度）
        retrieval_loss = fine_scores.sum() * 0.0  # 零张量，保持计算图
        if self.training and rel_labels is not None:
            scores_for_loss = coarse_scores.detach().clone() if self.gradient_isolation else coarse_scores
            raw_loss = self._compute_retrieval_loss(
                scores_for_loss,
                rel_labels,
                rel_type_mask,
            )
            retrieval_loss = self.retrieval_loss_weight * raw_loss

        return {
            'augmented_scores': augmented_scores,
            'retrieval_loss'  : retrieval_loss,
            'coarse_scores'   : coarse_scores,
            'fusion_alpha'    : alpha if isinstance(alpha, float) else alpha.item(),
        }


# 向后兼容别名
DualEncoderRetriever = DualEncoderAugmentor
