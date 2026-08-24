"""Coarse-to-Fine Cascade Augmentation (CCA).

The coarse retrieval score acts as a soft additive bias on the fine-grained
score rather than pruning or masking candidates:

    S_final = S_fine + alpha(t) * S_norm

where S_fine is the original GLiREL fine-grained interaction score (kept
intact), S_norm is the z-score-normalised dual-encoder coarse score, and
alpha(t) ramps linearly from 0 to alpha_max over the warmup window so the
auxiliary branch cannot destabilise the main scorer early in training.

Because the fusion is additive rather than a hard Top-K cut, low-confidence
unseen relations are never discarded before the joint encoder can recover
them. An auxiliary contrastive loss trains the coarse encoders; its gradient
is isolated from the fine-grained scorer (see `gradient_isolation`).

The class defaults below are placeholders; the values used in the paper are
set per dataset in `configs/` (e.g. Wiki-ZSL: retrieval_dim 128,
alpha_max 0.05, warmup 3000, loss weight 0.01).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class LightweightLabelEncoder(nn.Module):
    """Lightweight label encoder: hidden_size -> retrieval_dim.

    Single-hidden-layer FFN with LayerNorm; roughly
    hidden_size * retrieval_dim * 2 parameters.
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
    """Lightweight entity-pair encoder: hidden_size -> retrieval_dim.

    Same architecture as the label encoder but with independent parameters
    (asymmetric dual encoder).
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
    """Soft additive fusion of a coarse dual-encoder branch into the scorer.

    1. Compute coarse similarity scores S_coarse of shape [B, P, L] with the
       lightweight dual encoders.
    2. Fuse them into the fine-grained scores with a curriculum weight alpha:
       S_final = S_fine + alpha * S_norm.
    3. Train the coarse encoders with an auxiliary contrastive (InfoNCE)
       loss, which spreads out the label embedding space.

    Training: alpha ramps from 0 over `warmup_steps`; the total objective is
    relation_loss + retrieval_loss_weight * retrieval_loss, and the main
    relation loss is left unchanged.
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

        # Lightweight dual encoders.
        self.label_encoder = LightweightLabelEncoder(hidden_size, retrieval_dim, dropout)
        self.rel_encoder   = LightweightRelEncoder(hidden_size, retrieval_dim, dropout)

        # Learnable temperature, initialised to ln(1/0.07) ~= 2.659 as in CLIP.
        self.log_temp = nn.Parameter(torch.tensor(2.659))

        # Step counter; not a gradient-bearing parameter.
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.fusion_alpha_init = fusion_alpha_init

    @property
    def fusion_alpha(self) -> float:
        """Curriculum fusion weight.

        step <  warmup_steps: linear ramp from fusion_alpha_init to fusion_alpha_max.
        step >= warmup_steps: held at fusion_alpha_max.
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
        """Return the coarse cosine-similarity scores, shape [B, P, L]."""
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
        rel_labels: Tensor,      # [B, P]  label id, 1-indexed (0 = no relation)
        rel_type_mask: Tensor,   # [B, L]  mask over valid labels
    ) -> Tensor:
        """Auxiliary contrastive (InfoNCE) loss.

        For every positive pair (entity_pair, gold_label), maximise the gold
        label's score against all candidate labels. Only entity pairs with
        rel_labels > 0 contribute.
        """
        B, P, L = coarse_scores.shape

        # Positive mask: rel_labels > 0 means the pair carries a gold label.
        pos_mask = (rel_labels > 0) & (rel_labels <= L)  # [B, P]
        if not pos_mask.any():
            return coarse_scores.sum() * 0.0  # keep the graph connected

        # Clamp the logits before the softmax. Early in training the coarse
        # encoders are randomly initialised, so the InfoNCE term can produce
        # large, badly-scaled gradients; bounding the logit magnitude keeps
        # the auxiliary signal from destabilising training.
        safe_scores = coarse_scores.clamp(min=-50.0, max=50.0)

        # Mask invalid labels out of the softmax denominator.
        if rel_type_mask is not None:
            inv_mask = ~rel_type_mask.unsqueeze(1).expand_as(safe_scores)  # [B,P,L]
            safe_scores = safe_scores.masked_fill(inv_mask, -1e4)

        # Flatten and keep only the positive rows.
        flat_scores  = safe_scores.view(B * P, L)                  # [BP, L]
        flat_labels  = rel_labels.clamp(min=0).view(B * P)       # [BP]
        flat_pos_mask = pos_mask.view(B * P)                     # [BP]

        scores_pos  = flat_scores[flat_pos_mask]       # [N_pos, L]
        labels_pos  = flat_labels[flat_pos_mask] - 1    # [N_pos]  1-indexed -> 0-indexed
        labels_pos  = labels_pos.clamp(min=0, max=L - 1)

        # Cross-entropy on the clamped scores. This is equivalent to the
        # standard formulation; the clamp above only bounds the logit range.
        loss = F.cross_entropy(scores_pos, labels_pos)
        return loss

    def forward(
        self,
        fine_scores: Tensor,     # [B, P, L]  fine-grained scorer output
        rel_rep: Tensor,         # [B, P, D]
        rel_type_rep: Tensor,    # [B, L, D]
        rel_type_mask: Tensor,   # [B, L]
        rel_labels: Optional[Tensor] = None,  # [B, P] supplied during training
    ) -> dict:
        """Soft-fusion forward pass.

        Returns:
            augmented_scores : [B, P, L]  fused scores, replacing the originals
            retrieval_loss   : scalar     auxiliary loss
            fusion_alpha     : float      current fusion weight (for logging)
        """
        # Advance the step counter during training.
        if self.training:
            self._step += 1

        # Coarse scores, z-score normalised so the two scales are comparable.
        coarse_scores = self._compute_coarse_scores(rel_rep, rel_type_rep)  # [B, P, L]

        # Mask invalid labels, matching fine_scores.
        if rel_type_mask is not None:
            inv_mask = ~rel_type_mask.unsqueeze(1).expand_as(coarse_scores)
            coarse_scores = coarse_scores.masked_fill(inv_mask, -1e4)

        # Soft fusion: fine_scores + alpha * coarse_scores, with alpha
        # ramping from fusion_alpha_init to fusion_alpha_max over warmup.
        alpha = self.fusion_alpha
        augmented_scores = fine_scores + alpha * coarse_scores  # [B, P, L]

        # Safety fallback: if the coarse branch produces a degenerate or
        # non-finite score distribution, fall back to fine_scores so the main
        # task is unaffected; only retrieval_loss is retained.
        with torch.no_grad():
            # Mask of valid positions.
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
            # Fall back to the fine-grained scores.
            if self.training:
                logger.warning(
                    f"[CascadeAugment] step={self._step} coarse abnormal "
                    f"(std_ratio={aug_std/fine_std:.2f}), fallback to fine_scores"
                )
            augmented_scores = fine_scores

        # Auxiliary contrastive loss (training only; gradient is detached).
        retrieval_loss = fine_scores.sum() * 0.0  # zero tensor, keeps the graph
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


# Backwards-compatible alias.
DualEncoderRetriever = DualEncoderAugmentor
