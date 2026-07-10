#!/bin/bash
# =============================================================================
# GLiREL FewRel 微调脚本（复现用）
# 使用已下载的预训练模型 /root/autodl-tmp/models/glirel-large-v0
# 以及 DeBERTa-v3-large /root/autodl-tmp/models/deberta-v3-large
# =============================================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /root/GLiREL

LOG_DIR="logs_repro/fewrel"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "GLiREL FewRel 微调" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "配置文件: configs/config_few_rel_repro.yaml" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 检查数据文件
if [ ! -f "data/few_rel_all.jsonl" ]; then
    echo "ERROR: data/few_rel_all.jsonl 不存在，请先运行数据准备"
    exit 1
fi
echo "FewRel 数据行数: $(wc -l < data/few_rel_all.jsonl)" | tee -a "$LOG_FILE"

# 检查模型文件
if [ ! -f "/root/autodl-tmp/models/glirel-large-v0/pytorch_model.bin" ]; then
    echo "ERROR: 预训练模型未找到"
    exit 1
fi
echo "预训练模型: /root/autodl-tmp/models/glirel-large-v0" | tee -a "$LOG_FILE"

python3 train.py \
    --config configs/config_few_rel_repro.yaml \
    --log_dir "$LOG_DIR" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "========================================" | tee -a "$LOG_FILE"
echo "FewRel 微调结束时间: $(date)" | tee -a "$LOG_FILE"
echo "退出码: $EXIT_CODE" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 1 ]; then
    # 找到最佳 checkpoint
    BEST_CKPT=$(ls -d "$LOG_DIR"/model_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$BEST_CKPT" ]; then
        echo "最佳 checkpoint: $BEST_CKPT" | tee -a "$LOG_FILE"
        echo "$BEST_CKPT" > "$LOG_DIR/best_checkpoint.txt"
    fi
fi

echo "FewRel 微调完成！" | tee -a "$LOG_FILE"
