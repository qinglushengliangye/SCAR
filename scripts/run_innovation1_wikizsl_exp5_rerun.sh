#!/bin/bash
# ============================================================
# 创新点1（CCA）- WikiZSL 实验5 重跑（seed=55667788）
# 原实验在 step 3781 中断，清除旧日志后重新训练
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOG_DIR="/root/autodl-tmp/logs/logs_innovation1_wikizsl/logs_innovation1_exp5"

echo "========================================="
echo " CCA WikiZSL 实验5 重跑 | seed=55667788"
echo " 日志目录: $LOG_DIR"
echo " 开始时间: $(date)"
echo "========================================="

if [ -d "$LOG_DIR" ]; then
    echo "清除旧的不完整日志: $LOG_DIR"
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

cd /root/GLiREL

python3 train.py \
    --config configs/config_wiki_zsl_cascade_exp5.yaml \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "========================================="
echo " 训练完成，日志保存于 $LOG_DIR/train.log"
echo " 结束时间: $(date)"
echo "========================================="
