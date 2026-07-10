#!/bin/bash
# ============================================================
# 创新点2 第五组数据划分实验 - Wiki-ZSL (split5, seed=77889900)
# 日志目录：/root/autodl-tmp/logs_innovation2_split5
# ============================================================

set -e

LOG_DIR="/root/autodl-tmp/logs_innovation2_split5"
mkdir -p "${LOG_DIR}"

cd /root/GLiREL

echo "=========================================="
echo "启动 GLiREL 创新点2 实验 (Wiki-ZSL / split5)"
echo "配置文件: configs/config_wiki_zsl_innovation2_split5.yaml"
echo "日志目录: ${LOG_DIR}"
echo "开始时间: $(date)"
echo "=========================================="

python train.py \
    --config configs/config_wiki_zsl_innovation2_split5.yaml \
    --log_dir "${LOG_DIR}" \
    2>&1 | tee "${LOG_DIR}/train_stdout.log"

echo "=========================================="
echo "结束时间: $(date)"
echo "=========================================="
