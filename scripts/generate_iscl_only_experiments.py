#!/usr/bin/env python3
"""
Generate config YAML files and shell scripts for ISCL-only ablation experiments.

ISCL-only = Baseline + ISCL (no CCA), to verify ISCL's independent contribution.

Covers: WikiZSL / FewRel x m=5/10/15 x 5 seeds = 30 configs + 30 scripts + 1 batch runner
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DST = PROJECT_ROOT / "configs_iscl_only"
SCRIPTS_DST = PROJECT_ROOT / "scripts_iscl_only"

M_VALUES = [5, 10, 15]
SEEDS = {
    1: 11222333,
    2: 457365,
    3: 99887766,
    4: 11223344,
    5: 55667788,
}

LOG_ROOT = "/root/autodl-tmp/logs-iscl-only"

# ── Dataset-specific base parameters ──
# These mirror the Baseline (repro) configs exactly, with ISCL params appended.
DATASET_PARAMS = {
    "wiki_zsl": {
        "lr_encoder": "1e-5",
        "lr_others": "1e-4",
        "warmup_ratio": 0.1,
        "train_batch_size": 24,
        "eval_batch_size": 48,
        "alpha": 0.3,           # focal loss alpha
        "dataset_name": "wiki_zsl",
        "train_data": "data/wiki_zsl_all.jsonl",
        "eval_threshold_extra": "  - 0.05\n",  # wiki_zsl innovation2 includes 0.05
    },
    "few_rel": {
        "lr_encoder": "1e-5",
        "lr_others": "1e-4",
        "warmup_ratio": 0.1,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "alpha": 0.75,
        "dataset_name": "few_rel",
        "train_data": "data/few_rel_all.jsonl",
        "eval_threshold_extra": "",
    },
}

# ── ISCL hyperparameters (same as CCA+ISCL, from innovation2 configs) ──
ISCL_PARAMS = {
    "supcon_proj_dim": 128,
    "supcon_temperature": 0.1,
    "supcon_hard_neg_beta": 1.0,
    "supcon_warmup_steps": 1500,
    "supcon_loss_weight": 0.05,
}


def generate_config(dataset: str, m: int, seed: int, exp_n: int) -> str:
    dp = DATASET_PARAMS[dataset]
    dataset_label = "Wiki-ZSL" if dataset == "wiki_zsl" else "FewRel"
    log_subdir = f"iscl_only_{dataset}"

    return f"""# ============================================================
# ISCL-Only 消融实验 - {dataset_label} | m={m} | seed={seed}
# Baseline + ISCL（无 CCA），验证 ISCL 独立于 CCA 的效果
# ============================================================

# Learning Rate（与 Baseline 一致）
lr_encoder: {dp['lr_encoder']}
lr_others: {dp['lr_others']}
weight_decay_encoder: 0.0
weight_decay_other: 0.0

# Training Parameters
num_steps: 20000
warmup_ratio: {dp['warmup_ratio']}
train_batch_size: {dp['train_batch_size']}
eval_every: 300
gradient_accumulation: null
eval_batch_size: {dp['eval_batch_size']}
num_layers_freeze: null
early_stopping_patience: 12
early_stopping_delta: 0.0
threshold_search_metric: "macro_f1"
max_saves: 1

# Model Configuration
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
alpha: {dp['alpha']}
gamma: 3
label_embed_strategy: "both"

# Coreference Resolution
coref_classifier: false
coref_loss_weight: 10.0
coreference_label: "SELF"

# Entity markers
entity_start_token: "[E]"
entity_end_token: "[/E]"

# Directory Paths
dataset_name: "{dp['dataset_name']}"
root_dir: "{LOG_ROOT}/{log_subdir}"
train_data: "{dp['train_data']}"

# Pretrained GLiREL model as starting point
prev_path: /root/autodl-tmp/models/glirel-large-v0

# Training Specifics
size_sup: -1
num_train_rel_types: 25
num_unseen_rel_types: {m}
top_k: 1
random_drop: true
max_len: 512
eval_threshold:
  - 0.01
{dp['eval_threshold_extra']}  - 0.1
  - 0.2
  - 0.3
  - 0.5
  - 0.6
max_entity_pair_distance: null
fixed_relation_types: true

name: "large_iscl_only_m{m}"

positive_weight: 2.0
negative_weight: 1.0

# ============================================================
# CCA 禁用（纯 Baseline 打分路径）
# ============================================================
cascade_retrieval: false

# ============================================================
# ISCL：交互式有监督对比学习（与 CCA+ISCL 实验超参完全一致）
# ============================================================
supcon_enabled: true
supcon_proj_dim: {ISCL_PARAMS['supcon_proj_dim']}
supcon_temperature: {ISCL_PARAMS['supcon_temperature']}
supcon_hard_neg_beta: {ISCL_PARAMS['supcon_hard_neg_beta']}
supcon_warmup_steps: {ISCL_PARAMS['supcon_warmup_steps']}
supcon_loss_weight: {ISCL_PARAMS['supcon_loss_weight']}

seed: {seed}
"""


def generate_shell_script(dataset: str, m: int, seed: int, exp_n: int, config_filename: str) -> str:
    dataset_label = "Wiki-ZSL" if dataset == "wiki_zsl" else "FewRel"
    dataset_short = "wikizsl" if dataset == "wiki_zsl" else "fewrel"
    log_dir = f"{LOG_ROOT}/iscl_only_{dataset}/m{m}_exp{exp_n}"

    return f"""#!/bin/bash
# ============================================================
# ISCL-Only - {dataset_label} | m={m} 实验{exp_n}（seed={seed}）
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOG_DIR="{log_dir}"
mkdir -p "$LOG_DIR"

echo "========================================="
echo " ISCL-Only {dataset_label} m={m} exp{exp_n} | seed={seed}"
echo " 日志目录: $LOG_DIR"
echo " 开始时间: $(date)"
echo "========================================="

cd /root/GLiREL

python3 train.py \\
    --config configs_iscl_only/{config_filename} \\
    --log_dir "$LOG_DIR" \\
    2>&1 | tee "$LOG_DIR/train.log"

echo "========================================="
echo " 训练完成 | 结束时间: $(date)"
echo "========================================="
"""


def generate_batch_runner(all_scripts: list[dict]) -> str:
    header = """#!/bin/bash
# ============================================================
# 批量运行 ISCL-Only 消融实验
# 用法:
#   bash scripts_iscl_only/run_all.sh               # 全部运行
#   bash scripts_iscl_only/run_all.sh wikizsl        # 只运行 Wiki-ZSL
#   bash scripts_iscl_only/run_all.sh fewrel         # 只运行 FewRel
#   bash scripts_iscl_only/run_all.sh m5             # 只运行 m=5
#   bash scripts_iscl_only/run_all.sh m10            # 只运行 m=10
#   bash scripts_iscl_only/run_all.sh m15            # 只运行 m=15
# ============================================================
set -e

FILTER="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOTAL=0
DONE=0

"""
    body_lines = []
    for s in all_scripts:
        name = s["filename"]
        body_lines.append(f'if [ -z "$FILTER" ] || echo "{name}" | grep -q "$FILTER"; then')
        body_lines.append(f'    echo ">>> [{s["dataset"]} m={s["m"]} exp{s["n"]}] {name}"')
        body_lines.append(f'    bash "$SCRIPT_DIR/{name}"')
        body_lines.append(f'    DONE=$((DONE + 1))')
        body_lines.append(f'fi')
        body_lines.append(f'TOTAL=$((TOTAL + 1))')
        body_lines.append("")

    footer = """
echo "========================================="
echo " 全部完成: $DONE / $TOTAL 个实验已执行"
echo "========================================="
"""
    return header + "\n".join(body_lines) + footer


def main():
    CONFIGS_DST.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DST.mkdir(parents=True, exist_ok=True)

    all_scripts = []
    config_count = 0

    for dataset in ["wiki_zsl", "few_rel"]:
        dataset_short = "wikizsl" if dataset == "wiki_zsl" else "fewrel"
        for m in M_VALUES:
            for n, seed in SEEDS.items():
                config_filename = f"config_{dataset}_iscl_only_m{m}_exp{n}.yaml"
                script_filename = f"run_iscl_only_{dataset_short}_m{m}_exp{n}.sh"

                config_content = generate_config(dataset, m, seed, n)
                config_path = CONFIGS_DST / config_filename
                config_path.write_text(config_content, encoding="utf-8")

                script_content = generate_shell_script(dataset, m, seed, n, config_filename)
                script_path = SCRIPTS_DST / script_filename
                script_path.write_text(script_content, encoding="utf-8")
                script_path.chmod(0o755)

                all_scripts.append({
                    "filename": script_filename,
                    "dataset": dataset_short,
                    "m": m,
                    "n": n,
                })
                config_count += 1

    batch_content = generate_batch_runner(all_scripts)
    batch_path = SCRIPTS_DST / "run_all.sh"
    batch_path.write_text(batch_content, encoding="utf-8")
    batch_path.chmod(0o755)

    print(f"Generated {config_count} config files  -> {CONFIGS_DST}/")
    print(f"Generated {config_count} shell scripts  -> {SCRIPTS_DST}/")
    print(f"Generated batch runner              -> {batch_path}")
    print()
    print("Quick start:")
    print(f"  bash scripts_iscl_only/run_all.sh           # run all 30 experiments")
    print(f"  bash scripts_iscl_only/run_all.sh m15       # run only m=15 (10 experiments)")
    print(f"  bash scripts_iscl_only/run_all.sh wikizsl   # run only Wiki-ZSL (15 experiments)")


if __name__ == "__main__":
    main()
