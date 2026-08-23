#!/bin/bash
# ============================================================
# E1 (rebuttal): independent dev-split protocol, Wiki-ZSL m=15.
# 9 runs = {baseline, cca, scar} x {exp1, exp2, exp3}, assigned
# round-robin across available GPUs. Each run selects threshold +
# checkpoint on DEV and writes test-at-dev-threshold to
# <log_dir>/dev_test_results.json.
#
# Usage:
#   bash scripts/launch_e1_devsplit.sh "0 1 2 3"   # GPU ids to use
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /root/GLiREL

GPUS=(${1:-0})
NGPU=${#GPUS[@]}

METHODS=(repro cascade innovation2)
EXPS=(exp1 exp2 exp3)

i=0
for m in "${METHODS[@]}"; do
  for e in "${EXPS[@]}"; do
    gpu=${GPUS[$((i % NGPU))]}
    cfg="configs/e1_devsplit/config_wiki_zsl_${m}_dev_${e}.yaml"
    log_dir="/root/autodl-tmp/logs-e1/${m}_wikizsl/${e}"
    mkdir -p "$log_dir"
    echo "[E1] launching $m $e on GPU $gpu -> $log_dir"
    CUDA_VISIBLE_DEVICES="$gpu" python3 train_leakage_free.py \
      --config "$cfg" \
      --log_dir "$log_dir" \
      >"$log_dir/train.log" 2>&1 &
    i=$((i + 1))
  done
done

wait
echo "[E1] all runs finished. Collect with: python3 scripts/collect_e1_results.py"
