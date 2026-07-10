#!/bin/bash
# ============================================================
# CCA 消融实验：去除梯度隔离 - FewRel | seed=11223344
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOG_DIR="/root/autodl-tmp/logs/logs_ablation_fewrel_cca/ablation_no_grad_iso"
mkdir -p "$LOG_DIR"

echo "========================================="
echo " CCA 消融：去除梯度隔离 (FewRel) | seed=11223344"
echo " 日志目录: $LOG_DIR"
echo " 开始时间: $(date)"
echo "========================================="

cd /root/GLiREL

python3 train.py \
    --config configs/config_ablation_fewrel_cca_no_grad_iso.yaml \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "========================================="
echo " 训练完成，日志保存于 $LOG_DIR/train.log"
echo " 结束时间: $(date)"
echo "========================================="
