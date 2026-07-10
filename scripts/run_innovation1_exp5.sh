#!/bin/bash
# ============================================================
# 创新点1实验 - 实验5（Wiki-ZSL，seed=55667788）
# ============================================================
set -e

LOG_DIR="/root/autodl-tmp/logs_innovation1_exp5"
mkdir -p $LOG_DIR

echo "========================================="
echo " 创新点1实验5 - Wiki-ZSL | seed=55667788"
echo " 日志目录: $LOG_DIR"
echo "========================================="

cd /root/GLiREL

PYTHONPATH=. python train.py \
    --config configs/config_wiki_zsl_cascade_exp5.yaml \
    --log_dir $LOG_DIR \
    2>&1 | tee $LOG_DIR/train_stdout.log

echo "训练完成，日志保存于 $LOG_DIR/train_stdout.log"
