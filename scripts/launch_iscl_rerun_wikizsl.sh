#!/bin/bash
# ============================================================
# 重跑 Wiki-ZSL ISCL 三个消融，全部 seed=11223344
# 分别在 GPU 0/1/2 上并行
# ============================================================

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

BASE_LOG=/root/autodl-tmp/logs-m=15/logs_ablation_iscl_seed11223344
mkdir -p $BASE_LOG/ablation_no_cross_attn
mkdir -p $BASE_LOG/ablation_no_global_align
mkdir -p $BASE_LOG/ablation_no_supcon_warmup

cd /root/GLiREL

echo "[$(date)] Launching GPU 0: no_cross_attn"
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py \
    --config configs/config_ablation_iscl_no_cross_attn.yaml \
    --log_dir $BASE_LOG/ablation_no_cross_attn \
    > $BASE_LOG/ablation_no_cross_attn/nohup.out 2>&1 &
PID0=$!
echo "  PID=$PID0"

sleep 2

echo "[$(date)] Launching GPU 1: no_global_align"
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py \
    --config configs/config_ablation_iscl_no_global_align.yaml \
    --log_dir $BASE_LOG/ablation_no_global_align \
    > $BASE_LOG/ablation_no_global_align/nohup.out 2>&1 &
PID1=$!
echo "  PID=$PID1"

sleep 2

echo "[$(date)] Launching GPU 2: no_supcon_warmup"
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py \
    --config configs/config_ablation_iscl_no_supcon_warmup.yaml \
    --log_dir $BASE_LOG/ablation_no_supcon_warmup \
    > $BASE_LOG/ablation_no_supcon_warmup/nohup.out 2>&1 &
PID2=$!
echo "  PID=$PID2"

echo ""
echo "============================================================"
echo "All 3 jobs launched."
echo "  GPU 0 / no_cross_attn      PID=$PID0"
echo "  GPU 1 / no_global_align    PID=$PID1"
echo "  GPU 2 / no_supcon_warmup   PID=$PID2"
echo ""
echo "Log dirs under: $BASE_LOG"
echo "Monitor with:   tail -f $BASE_LOG/ablation_no_*/train.log"
echo "Or check GPU:   nvidia-smi"
echo "============================================================"
