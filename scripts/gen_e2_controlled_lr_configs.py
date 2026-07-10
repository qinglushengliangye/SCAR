"""Generate E2 (controlled learning-rate) configs for the rebuttal.

Isolates the Wiki-ZSL LR confound (ppaf) with two clean controls under the
STANDARD m=15 protocol (directly comparable to the paper's main table):

  (a) baseline_lowlr : Baseline run at the CCA/SCAR LR (2e-6/2e-5, warmup 0.03)
                       -> substantiates the paper's claim that low LR hurts Baseline.
  (b) scar_highlr    : SCAR run at the Baseline LR (1e-5/1e-4, warmup 0.1)
                       -> shows SCAR's gain is not an LR artifact.
  (c) cca_highlr     : CCA at the Baseline LR (completeness).

All use the identical 7-point threshold grid. 3 relation splits (seeds) each.
Written to configs/e2_controlled_lr/.
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "configs", "e2_controlled_lr")
os.makedirs(OUT, exist_ok=True)

SEEDS = {"exp1": 1, "exp2": 2, "exp3": 3}
THRESHOLD = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6]

COMMON = """# ==== E2 rebuttal: controlled learning-rate (Wiki-ZSL m=15, standard split) ====
weight_decay_encoder: 0.0
weight_decay_other: 0.0
num_steps: 20000
train_batch_size: 24
eval_every: 300
gradient_accumulation: null
eval_batch_size: 48
num_layers_freeze: null
early_stopping_patience: 12
early_stopping_delta: 0.0
threshold_search_metric: "macro_f1"
max_saves: 1
max_width: 12
model_name: /root/autodl-tmp/models/deberta-v3-large
fine_tune: true
subtoken_pooling: first
hidden_size: 768
scorer: "dot"
rel_mode: marker
span_marker_mode: markerv1
refine_prompt: false
refine_relation: false
ffn_mul: 4
dropout: 0.4
scheduler: "cosine_with_warmup"
loss_func: "binary_cross_entropy_loss"
alpha: 0.3
gamma: 3
label_embed_strategy: "both"
coref_classifier: false
coref_loss_weight: 10.0
coreference_label: "SELF"
entity_start_token: "[E]"
entity_end_token: "[/E]"
dataset_name: "wiki_zsl"
train_data: "data/wiki_zsl_all.jsonl"
prev_path: /root/autodl-tmp/models/glirel-large-v0
size_sup: -1
num_train_rel_types: 25
num_unseen_rel_types: 15
top_k: 1
random_drop: true
max_len: 512
max_entity_pair_distance: null
fixed_relation_types: true
positive_weight: 2.0
negative_weight: 1.0
"""

CCA_BLOCK = """cascade_retrieval: true
retrieval_dim: 128
fusion_alpha_init: 0.0
fusion_alpha_max: 0.05
cascade_warmup_steps: 3000
retrieval_loss_weight: 0.01
"""

ISCL_BLOCK = """supcon_enabled: true
supcon_proj_dim: 128
supcon_temperature: 0.1
supcon_hard_neg_beta: 1.0
supcon_warmup_steps: 1500
supcon_loss_weight: 0.05
"""

VARIANTS = {
    # name: (lr_encoder, lr_others, warmup_ratio, extra_blocks, run_name)
    "baseline_lowlr": ("2e-6", "2e-5", 0.03, "", "e2_baseline_lowlr"),
    "cca_highlr": ("1e-5", "1e-4", 0.1, CCA_BLOCK, "e2_cca_highlr"),
    "scar_highlr": ("1e-5", "1e-4", 0.1, CCA_BLOCK + ISCL_BLOCK, "e2_scar_highlr"),
}


def main():
    thr = "eval_threshold:\n" + "".join(f"  - {t}\n" for t in THRESHOLD)
    n = 0
    for variant, (lre, lro, wu, extra, rname) in VARIANTS.items():
        for exp, seed in SEEDS.items():
            root = f"/root/autodl-tmp/logs-e2/{variant}_wikizsl"
            cfg = (
                COMMON
                + thr
                + f"lr_encoder: {lre}\n"
                + f"lr_others: {lro}\n"
                + f"warmup_ratio: {wu}\n"
                + f'name: "{rname}"\n'
                + extra
                + f"root_dir: {root}\n"
                + f"seed: {seed}\n"
            )
            path = os.path.join(OUT, f"config_{variant}_{exp}.yaml")
            with open(path, "w") as f:
                f.write(cfg)
            n += 1
            print("wrote", path)
    print(f"done: {n} configs in {OUT}")


if __name__ == "__main__":
    main()
