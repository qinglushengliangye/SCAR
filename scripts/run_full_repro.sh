#!/bin/bash
# =============================================================================
# GLiREL 完整复现一键脚本
#
# 执行顺序：
#   1. 检查环境和数据
#   2. 微调 FewRel
#   3. 微调 Wiki-ZSL
#   4. 在两个数据集上推理评估（使用各自微调模型）
#
# 用法:
#   bash run_full_repro.sh                    # 完整流程
#   bash run_full_repro.sh --skip-finetune    # 跳过微调，直接用预训练模型评估
#   bash run_full_repro.sh --eval-only        # 仅评估（需指定 checkpoint）
#
# =============================================================================
set -e

cd /root/GLiREL
export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PRETRAINED_MODEL="/root/autodl-tmp/models/glirel-large-v0"
FEWREL_DATA="data/few_rel_all.jsonl"
WIKI_DATA="data/wiki_zsl_all.jsonl"
FEWREL_LOG_DIR="logs_repro/fewrel"
WIKI_LOG_DIR="logs_repro/wiki_zsl"
MAIN_LOG="logs_repro/full_repro_$(date +%Y%m%d_%H%M%S).log"

mkdir -p logs_repro

SKIP_FINETUNE=false
EVAL_ONLY=false
FEWREL_CKPT=""
WIKI_CKPT=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-finetune)
            SKIP_FINETUNE=true
            shift ;;
        --eval-only)
            EVAL_ONLY=true
            SKIP_FINETUNE=true
            shift ;;
        --fewrel-ckpt)
            FEWREL_CKPT="$2"
            shift 2 ;;
        --wiki-ckpt)
            WIKI_CKPT="$2"
            shift 2 ;;
        *)
            echo "未知参数: $1"
            exit 1 ;;
    esac
done

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$MAIN_LOG"
}

log "============================================================"
log "GLiREL 完整复现流程开始"
log "============================================================"
log "主日志文件: $MAIN_LOG"
log "跳过微调: $SKIP_FINETUNE"

# =============================================================================
# STEP 1: 环境检查
# =============================================================================
log ""
log "STEP 1: 环境检查"

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {p.name} | {p.total_memory // 1024**3} GB')
" | tee -a "$MAIN_LOG"

# 检查数据
for f in "$FEWREL_DATA" "$WIKI_DATA"; do
    if [ ! -f "$f" ]; then
        log "ERROR: 数据文件不存在: $f"
        exit 1
    fi
done
log "FewRel 数据: $(wc -l < $FEWREL_DATA) 条"
log "Wiki-ZSL 数据: $(wc -l < $WIKI_DATA) 条"

# 检查预训练模型
if [ ! -f "$PRETRAINED_MODEL/pytorch_model.bin" ]; then
    log "ERROR: 预训练模型不存在: $PRETRAINED_MODEL"
    exit 1
fi
log "预训练模型: $PRETRAINED_MODEL [OK]"

# =============================================================================
# STEP 2: 微调 FewRel
# =============================================================================
if [ "$SKIP_FINETUNE" = false ]; then
    log ""
    log "STEP 2: 微调 FewRel"
    log "配置: configs/config_few_rel_repro.yaml"
    log "日志目录: $FEWREL_LOG_DIR"
    mkdir -p "$FEWREL_LOG_DIR"

    python3 train.py \
        --config configs/config_few_rel_repro.yaml \
        --log_dir "$FEWREL_LOG_DIR" \
        2>&1 | tee -a "$FEWREL_LOG_DIR/train.log" "$MAIN_LOG" || true

    log "FewRel 微调完成"
else
    log "STEP 2: 跳过 FewRel 微调"
fi

# =============================================================================
# STEP 3: 微调 Wiki-ZSL
# =============================================================================
if [ "$SKIP_FINETUNE" = false ]; then
    log ""
    log "STEP 3: 微调 Wiki-ZSL"
    log "配置: configs/config_wiki_zsl_repro.yaml"
    log "日志目录: $WIKI_LOG_DIR"
    mkdir -p "$WIKI_LOG_DIR"

    python3 train.py \
        --config configs/config_wiki_zsl_repro.yaml \
        --log_dir "$WIKI_LOG_DIR" \
        2>&1 | tee -a "$WIKI_LOG_DIR/train.log" "$MAIN_LOG" || true

    log "Wiki-ZSL 微调完成"
else
    log "STEP 3: 跳过 Wiki-ZSL 微调"
fi

# =============================================================================
# STEP 4: 确定 checkpoint 路径
# =============================================================================
log ""
log "STEP 4: 确定评估用 checkpoint"

# FewRel checkpoint
if [ -z "$FEWREL_CKPT" ]; then
    if [ -f "$FEWREL_LOG_DIR/best_checkpoint.txt" ]; then
        FEWREL_CKPT=$(cat "$FEWREL_LOG_DIR/best_checkpoint.txt")
    else
        FEWREL_CKPT=$(ls -d "$FEWREL_LOG_DIR"/model_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || echo "")
    fi
fi

# Wiki-ZSL checkpoint
if [ -z "$WIKI_CKPT" ]; then
    if [ -f "$WIKI_LOG_DIR/best_checkpoint.txt" ]; then
        WIKI_CKPT=$(cat "$WIKI_LOG_DIR/best_checkpoint.txt")
    else
        WIKI_CKPT=$(ls -d "$WIKI_LOG_DIR"/model_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || echo "")
    fi
fi

# 如果没有微调 checkpoint，回退到预训练模型
[ -z "$FEWREL_CKPT" ] && FEWREL_CKPT="$PRETRAINED_MODEL"
[ -z "$WIKI_CKPT" ]   && WIKI_CKPT="$PRETRAINED_MODEL"

log "FewRel  使用模型: $FEWREL_CKPT"
log "Wiki-ZSL 使用模型: $WIKI_CKPT"

# =============================================================================
# STEP 5: 推理评估
# =============================================================================
log ""
log "STEP 5: 推理评估（完整数据集，搜索最优 threshold）"

python3 infer_and_eval.py \
    --fewrel-model "$FEWREL_CKPT" \
    --wiki-model   "$WIKI_CKPT" \
    --datasets both \
    --fewrel-data  "$FEWREL_DATA" \
    --wiki-data    "$WIKI_DATA" \
    --eval-batch-size 32 \
    --search-threshold \
    --top-k 1 \
    --num-unseen-rel-types 15 \
    2>&1 | tee -a "$MAIN_LOG"

log ""
log "============================================================"
log "全流程完成！"
log "主日志: $MAIN_LOG"
log "FewRel 训练日志: $FEWREL_LOG_DIR/train.log"
log "Wiki-ZSL 训练日志: $WIKI_LOG_DIR/train.log"
log "推理评估日志: logs_repro/eval_*.log"
log "============================================================"
