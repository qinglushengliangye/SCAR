"""Generate E1 (independent dev-split) configs for the rebuttal.

For Wiki-ZSL m=15 we hold out an ADDITIONAL 15 relations as a dev set
(num_dev_rel_types=15), disjoint from both train and the 15 test relations.
Threshold + checkpoint selection run on dev; test is reported at the
dev-selected threshold (handled in train.py).

Three methods x three relation splits (seeds) = 9 configs, written to
configs/e1_devsplit/. Baseline uses the identical 7-point threshold grid as
CCA/SCAR (0.05 added) so the only controlled difference is the method.
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "configs", "e1_devsplit")
os.makedirs(OUT, exist_ok=True)

SEEDS = {"exp1": 1, "exp2": 2, "exp3": 3}
THRESHOLD = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6]

COMMON = """# ==== E1 rebuttal: independent dev-split (Wiki-ZSL m=15) ====
# Selection (threshold + checkpoint) on dev; test reported at dev-selected threshold.
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
num_dev_rel_types: 15
top_k: 1
random_drop: true
max_len: 512
max_entity_pair_distance: null
fixed_relation_types: true
positive_weight: 2.0
negative_weight: 1.0
"""

METHOD_BLOCKS = {
    "repro": """warmup_ratio: 0.1
lr_encoder: 1e-5
lr_others: 1e-4
name: "e1_baseline"
""",
    "cascade": """warmup_ratio: 0.03
lr_encoder: 2e-6
lr_others: 2e-5
name: "e1_cca"
cascade_retrieval: true
retrieval_dim: 128
fusion_alpha_init: 0.0
fusion_alpha_max: 0.05
cascade_warmup_steps: 3000
retrieval_loss_weight: 0.01
""",
    "innovation2": """warmup_ratio: 0.03
lr_encoder: 2e-6
lr_others: 2e-5
name: "e1_scar"
cascade_retrieval: true
retrieval_dim: 128
fusion_alpha_init: 0.0
fusion_alpha_max: 0.05
cascade_warmup_steps: 3000
retrieval_loss_weight: 0.01
supcon_enabled: true
supcon_proj_dim: 128
supcon_temperature: 0.1
supcon_hard_neg_beta: 1.0
supcon_warmup_steps: 1500
supcon_loss_weight: 0.05
""",
}


def main():
    thr = "eval_threshold:\n" + "".join(f"  - {t}\n" for t in THRESHOLD)
    n = 0
    for method, block in METHOD_BLOCKS.items():
        for exp, seed in SEEDS.items():
            root = f"/root/autodl-tmp/logs-e1/{method}_wikizsl"
            cfg = (
                COMMON
                + thr
                + block
                + f"root_dir: {root}\n"
                + f"seed: {seed}\n"
            )
            path = os.path.join(OUT, f"config_wiki_zsl_{method}_dev_{exp}.yaml")
            with open(path, "w") as f:
                f.write(cfg)
            n += 1
            print("wrote", path)
    print(f"done: {n} configs in {OUT}")


if __name__ == "__main__":
    main()
