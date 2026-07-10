#!/bin/bash
# ============================================================
# E2 (rebuttal): controlled learning-rate, Wiki-ZSL m=15.
# 9 runs = {baseline_lowlr, cca_highlr, scar_highlr} x {exp1,exp2,exp3},
# round-robin across available GPUs. Standard (dev==test) protocol, so
# results are directly comparable to the paper's Table (main).
#
# Usage:
#   bash scripts/launch_e2_controlled_lr.sh "0 1 2 3"
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /root/GLiREL

GPUS=(${1:-0})
NGPU=${#GPUS[@]}

VARIANTS=(baseline_lowlr cca_highlr scar_highlr)
EXPS=(exp1 exp2 exp3)

i=0
for v in "${VARIANTS[@]}"; do
  for e in "${EXPS[@]}"; do
    gpu=${GPUS[$((i % NGPU))]}
    cfg="configs/e2_controlled_lr/config_${v}_${e}.yaml"
    log_dir="/root/autodl-tmp/logs-e2/${v}_wikizsl/${e}"
    mkdir -p "$log_dir"
    echo "[E2] launching $v $e on GPU $gpu -> $log_dir"
    CUDA_VISIBLE_DEVICES="$gpu" python3 train.py \
      --config "$cfg" \
      --log_dir "$log_dir" \
      >"$log_dir/train.log" 2>&1 &
    i=$((i + 1))
  done
done

wait
echo "[E2] all runs finished. Best dev macro-F1 per run is in each train.log."
