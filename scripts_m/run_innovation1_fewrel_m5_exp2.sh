#!/bin/bash
# ============================================================
# 创新点1（CCA） - FewRel | m=5 实验2（seed=457365）
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOG_DIR="/root/autodl-tmp/logs-m=5/cascade_fewrel/exp2"
mkdir -p "$LOG_DIR"

echo "========================================="
echo " 创新点1（CCA） FewRel m=5 实验2 | seed=457365"
echo " 日志目录: $LOG_DIR"
echo " 开始时间: $(date)"
echo "========================================="

cd /root/GLiREL

python3 train.py \
    --config configs_m/config_few_rel_cascade_m5_exp2.yaml \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "========================================="
echo " 训练完成，日志保存于 $LOG_DIR/train.log"
echo " 结束时间: $(date)"
echo "========================================="
