#!/bin/bash
# ============================================================
# ISCL 消融实验：去除交叉注意力 | seed=20260421
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOG_DIR="/root/autodl-tmp/logs/logs_ablation_iscl/ablation_no_cross_attn"
mkdir -p "$LOG_DIR"

echo "========================================="
echo " ISCL 消融：去除交叉注意力 | seed=20260421"
echo " 日志目录: $LOG_DIR"
echo " 开始时间: $(date)"
echo "========================================="

cd /root/GLiREL

python3 train.py \
    --config configs/config_ablation_iscl_no_cross_attn.yaml \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "========================================="
echo " 训练完成，日志保存于 $LOG_DIR/train.log"
echo " 结束时间: $(date)"
echo "========================================="
