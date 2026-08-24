"""Interactive Supervised Contrastive Learning (ISCL).

Joint-encoding ZSRE struggles to separate fine-grained relations that share
an entity-type signature (e.g. *founder* vs. *co-founder*). This module
reshapes the projection space so that same-relation pairs are pulled together
and semantically close but differently-labelled "hard negatives" are pushed
apart. Unlike the primary BCE loss, which only constrains the decision
boundary, this explicitly constrains the global geometry of the space.

Three components adapt supervised contrastive learning to joint encoding:

1. A label-aware cross-attention layer lets each entity-pair representation
   attend over all candidate label representations in the batch, producing a
   label-conditioned contrastive representation.
2. Hard negatives are the negatives whose label embeddings are most similar
   to the anchor's; they are up-weighted in the SupCon denominator by
   w = 1 + beta * max(0, cos).
3. A linear curriculum warmup ramps the loss weight from 0 to
   loss_weight_max, so the contrastive term cannot disrupt the pretrained
   entity-pair representations early in training.

Numerical safeguards: projections are L2-normalised and temperature-scaled;
logits are clamped to +-50 before the softmax to avoid fp16 overflow under
AMP; if a batch contains no valid positive pair, or if the loss becomes
non-finite, a zero loss is returned with the graph kept intact.
"""

from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger


class LabelAwareCrossAttention(nn.Module):
    """Lightweight single-head cross-attention.

    Q = rel_rep [B, P, D]; K, V = rel_type_rep [B, L, D]. Returns
    label-aware entity-pair representations [B, P, D] with a residual
    connection and LayerNorm. About 3*D^2 + D ~= 1.77M parameters at D=768.
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
        attn = torch.nan_to_num(attn, nan=0.0)  # a fully masked row yields NaN
        attn = self.dropout(attn)

        ctx = torch.einsum('BPL,BLD->BPD', attn, v)  # [B, P, D]
        ctx = self.out_proj(ctx)
        out = self.layer_norm(rel_rep + ctx)         # residual
        return out


class ProjectionHead(nn.Module):
    """Projection head into the contrastive space.

    Linear(D, 2d_c) -> LayerNorm -> GELU -> Dropout -> Linear(2d_c, d_c),
    followed by L2 normalisation. About D*2d_c + 2d_c*d_c ~= 230K parameters.
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
    """Supervised contrastive loss with hard-negative weighting and warmup.

    Args:
      rel_rep       [B, P, D]  entity-pair representations (padded positions included)
      rel_type_rep  [B, L, D]  label representations
      rel_type_mask [B, L]     mask over valid labels
      rel_labels    [B, P]     label ids (1-indexed; 0 = no relation, -1 = padding)
      class_to_ids  List[Dict] per-sample {rel_text: local_id} mapping

    Returns a dict with the warmup-scaled loss, the unscaled loss (for
    logging), the current loss weight, and the number of anchors.
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
        """Linear warmup from 0 to loss_weight_max, then held constant."""
        if self.warmup_steps <= 0:
            return self.loss_weight_max
        progress = min(1.0, self._step.item() / max(1, self.warmup_steps))
        return self.loss_weight_max * progress

    @staticmethod
    def _build_global_label_ids(
        class_to_ids_batch: List[Dict[str, int]],
    ):
        """Map each sample's local label ids to batch-global ids.

        This is what makes anchors of the same relation in different samples
        of the batch count as positives.

        Returns:
          global_id_per_local: per sample, the [L_i] global ids ordered by local id
          num_global         : number of distinct labels in the batch
        """
        text_to_global: Dict[str, int] = {}
        global_id_per_local: List[List[int]] = []
        for mapping in class_to_ids_batch:
            if mapping is None:
                global_id_per_local.append([])
                continue
            # Sort by ascending local id and keep only local > 0, which
            # drops the special coref indices (local can be -50) and matches
            # the rel_labels > 0 convention.
            sorted_items = sorted(
                [(v, k) for k, v in mapping.items() if v > 0],
                key=lambda x: x[0],
            )
            labels_for_this = []
            # local_id is 1-based and contiguous, so fill positionally.
            max_local = sorted_items[-1][0] if sorted_items else 0
            # Reserve max_local slots (index = local_id - 1).
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
        rel_labels: Optional[Tensor],         # [B, P]  1-indexed; 0/-1 are not positives
        class_to_ids_batch: Optional[List[Dict[str, int]]] = None,
    ) -> dict:
        device = rel_rep.device
        zero = rel_rep.sum() * 0.0  # keeps the graph connected

        # Only step and compute the loss during training; return 0 at eval.
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

        # 1) Interactive features: attend over the batch's label set.
        if self.cross_attention:
            interacted = self.cross_attn(rel_rep, rel_type_rep, rel_type_mask)  # [B, P, D]
        else:
            interacted = rel_rep

        # 2) Project into the contrastive space.
        z_pair = self.pair_proj(interacted)                                 # [B, P, d_c]
        z_label = self.label_proj(rel_type_rep)                             # [B, L, d_c]

        # 3) Anchors are positions with rel_labels in [1, L]; padding (-1)
        #    and negative pairs (0) are excluded.
        valid_anchor_mask = (rel_labels > 0) & (rel_labels <= L)            # [B, P]
        n_anchors = int(valid_anchor_mask.sum().item())
        if n_anchors < 2:
            # Cannot form a positive pair.
            return {
                'supcon_loss': zero,
                'raw_loss'   : zero,
                'loss_weight': self.loss_weight,
                'n_anchors'  : n_anchors,
            }

        # 4) Map labels to batch-global ids to find positives across samples.
        if not self.global_alignment or class_to_ids_batch is None:
            global_ids_per_sample = [list(range(1, L + 1)) for _ in range(B)]
            num_global = L
        else:
            global_ids_per_sample, num_global = self._build_global_label_ids(class_to_ids_batch)

        # 5) For each anchor collect its global label id (to split positives
        #    from negatives) and its label embedding (for hard-negative weighting).
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

        # 6) Gather the anchor projection vectors.
        z_flat = z_pair.reshape(B * P, self.proj_dim)       # [B*P, d_c]
        z_anchors = z_flat[anchor_idx_tensor]               # [N, d_c]

        # 7) Gather each anchor's own label embedding, taking row
        #    (local_id - 1) from that sample's z_label.
        z_label_anchor = z_label[anchor_batch_tensor, anchor_local_tensor]  # [N, d_c]

        # 8) Anchor-anchor similarity matrix.
        sim = torch.matmul(z_anchors, z_anchors.t())        # [N, N]
        sim = sim / self.temperature
        sim = sim.clamp(min=-50.0, max=50.0)

        # 9) Positive mask: same global id, excluding the anchor itself.
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

        # 10) Hard-negative weighting: for each negative, the cosine
        #     similarity between anchor label embeddings measures hardness.
        neg_mask = (~pos_mask) & (~self_mask)               # [N, N]

        # Label similarity; the vectors are L2-normalised, so the dot product is cosine.
        label_sim = torch.matmul(z_label_anchor, z_label_anchor.t())  # [N, N]
        label_sim = label_sim.clamp(min=-1.0, max=1.0)
        hard_weight = 1.0 + self.hard_neg_beta * label_sim.clamp(min=0.0)  # [N, N] ≥ 1

        # Subtract the row max for numerical stability.
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()
        exp_sim = torch.exp(sim_stable)

        # Weight negatives only; positives and the self term are unweighted.
        weights = torch.ones_like(exp_sim)
        weights = torch.where(neg_mask, hard_weight, weights)

        # Denominator: weighted sum over all non-self terms.
        valid_mask = ~self_mask                              # [N, N]
        denom = (exp_sim * weights * valid_mask.float()).sum(dim=1) + 1e-8  # [N]

        # Numerator: mean log-probability over each anchor's positives.
        log_prob = sim_stable - torch.log(denom).unsqueeze(1)  # [N, N]
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0) # [N]
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count  # [N]

        # Only anchors that have a positive contribute to the mean.
        loss_per_anchor = -mean_log_prob_pos                 # [N]
        loss_per_anchor = torch.where(has_pos, loss_per_anchor, torch.zeros_like(loss_per_anchor))
        n_valid = has_pos.float().sum().clamp(min=1.0)
        raw_loss = loss_per_anchor.sum() / n_valid

        # Fall back to a zero loss if the result is non-finite.
        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            logger.warning(
                f"[SupCon] step={self._step.item()} raw_loss NaN/Inf, fallback to zero"
            )
            raw_loss = zero

        lw = self.loss_weight
        supcon_loss = lw * raw_loss

        # Periodic logging.
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
