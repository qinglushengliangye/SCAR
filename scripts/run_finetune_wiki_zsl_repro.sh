#!/bin/bash
# =============================================================================
# GLiREL Wiki-ZSL 微调脚本（复现用）
# =============================================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /root/GLiREL

LOG_DIR="logs_repro/wiki_zsl"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "GLiREL Wiki-ZSL 微调" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "配置文件: configs/config_wiki_zsl_repro.yaml" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 检查数据文件
if [ ! -f "data/wiki_zsl_all.jsonl" ]; then
    echo "ERROR: data/wiki_zsl_all.jsonl 不存在"
    exit 1
fi
echo "Wiki-ZSL 数据行数: $(wc -l < data/wiki_zsl_all.jsonl)" | tee -a "$LOG_FILE"

# 检查模型文件
if [ ! -f "/root/autodl-tmp/models/glirel-large-v0/pytorch_model.bin" ]; then
    echo "ERROR: 预训练模型未找到"
    exit 1
fi
echo "预训练模型: /root/autodl-tmp/models/glirel-large-v0" | tee -a "$LOG_FILE"

python3 train.py \
    --config configs/config_wiki_zsl_repro.yaml \
    --log_dir "$LOG_DIR" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "========================================" | tee -a "$LOG_FILE"
echo "Wiki-ZSL 微调结束时间: $(date)" | tee -a "$LOG_FILE"
echo "退出码: $EXIT_CODE" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 1 ]; then
    BEST_CKPT=$(ls -d "$LOG_DIR"/model_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$BEST_CKPT" ]; then
        echo "最佳 checkpoint: $BEST_CKPT" | tee -a "$LOG_FILE"
        echo "$BEST_CKPT" > "$LOG_DIR/best_checkpoint.txt"
    fi
fi

echo "Wiki-ZSL 微调完成！" | tee -a "$LOG_FILE"
