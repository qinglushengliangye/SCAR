#!/bin/bash
# ============================================================
# 级联推理架构微调脚本 - FewRel 数据集
# 创新点1：双编码器预筛选 + 细粒度交互式计算
# ============================================================

set -e

LOG_DIR="logs_cascade/fewrel"
mkdir -p $LOG_DIR

echo "========================================="
echo " 级联推理架构训练 - FewRel"
echo " 日志目录: $LOG_DIR"
echo "========================================="

PYTHONPATH=. python train.py \
    --config configs/config_few_rel_cascade.yaml \
    --log_dir $LOG_DIR \
    2>&1 | tee $LOG_DIR/train.log

echo "训练完成，日志保存于 $LOG_DIR/train.log"
