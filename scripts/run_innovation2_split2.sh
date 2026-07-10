#!/bin/bash
# ============================================================
# 创新点2 第二组数据划分实验 - Wiki-ZSL
# 验证 Innovation 2 在不同 train/eval 关系划分下的稳定性
# seed=11223344（与主实验 seed=20260421 不同）
# ============================================================
set -e

LOG_DIR="/root/autodl-tmp/logs_innovation2_split2"
mkdir -p $LOG_DIR

echo "========================================="
echo " 创新点2 split2 实验 - Wiki-ZSL | seed=11223344"
echo " 日志目录: $LOG_DIR"
echo "========================================="

cd /root/GLiREL

PYTHONPATH=. python train.py \
    --config configs/config_wiki_zsl_innovation2_split2.yaml \
    --log_dir $LOG_DIR \
    2>&1 | tee $LOG_DIR/train_stdout.log

echo "训练完成，日志保存于 $LOG_DIR/train_stdout.log"
