#!/bin/bash
# ============================================================
# Wiki-ZSL ablation 3-seed 重复验证 launcher
# 5 个 ablation 模块 (no_cascade_warmup 已弃用)
#   CCA  × {no_grad_iso, no_zscore}                  × 3 seed
#   ISCL × {no_cross_attn, no_global_align, no_supcon_warmup} × 3 seed
#
# 用法: bash launch_rerun_batch.sh <batch_id>
#   batch 1: CCA  × seed 11222333  (2 jobs → GPU 0, 1)
#   batch 2: CCA  × seed 55667788  (2 jobs → GPU 0, 1)
#   batch 3: ISCL × seed 11223344  (3 jobs → GPU 0, 1, 2)
#   batch 4: ISCL × seed 33445566  (3 jobs → GPU 0, 1, 2)
# ============================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash $0 <batch_id>  (1, 2, 3, or 4)"
    echo ""
    echo "  batch 1: CCA  ablations × seed 11222333  (2 jobs)"
    echo "  batch 2: CCA  ablations × seed 55667788  (2 jobs)"
    echo "  batch 3: ISCL ablations × seed 11223344  (3 jobs)"
    echo "  batch 4: ISCL ablations × seed 33445566  (3 jobs)"
    exit 1
fi

BATCH=$1
CFG_DIR=/root/GLiREL/configs/rerun_ablation
BASE_LOG=/root/autodl-tmp/logs-m=15/logs_ablation_rerun

case $BATCH in
    1)
        SEED=11222333
        CONFIGS=("cca_no_grad_iso_seed${SEED}" "cca_no_zscore_seed${SEED}")
        ;;
    2)
        SEED=55667788
        CONFIGS=("cca_no_grad_iso_seed${SEED}" "cca_no_zscore_seed${SEED}")
        ;;
    3)
        SEED=11223344
        CONFIGS=("iscl_no_cross_attn_seed${SEED}" "iscl_no_global_align_seed${SEED}" "iscl_no_supcon_warmup_seed${SEED}")
        ;;
    4)
        SEED=33445566
        CONFIGS=("iscl_no_cross_attn_seed${SEED}" "iscl_no_global_align_seed${SEED}" "iscl_no_supcon_warmup_seed${SEED}")
        ;;
    *)
        echo "ERROR: batch must be 1, 2, 3, or 4 (got: $BATCH)"
        exit 1
        ;;
esac

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================================"
echo " Batch $BATCH  (seed=$SEED, ${#CONFIGS[@]} jobs)"
echo " Configs:"
for c in "${CONFIGS[@]}"; do echo "   - $c"; done
echo "============================================================"

cd /root/GLiREL

PIDS=()
for i in "${!CONFIGS[@]}"; do
    cfg=${CONFIGS[$i]}
    log_dir=$BASE_LOG/$cfg
    mkdir -p $log_dir
    echo "[$(date +%H:%M:%S)] GPU $i  ←  $cfg"
    CUDA_VISIBLE_DEVICES=$i nohup python3 train.py \
        --config $CFG_DIR/${cfg}.yaml \
        --log_dir $log_dir \
        > $log_dir/nohup.out 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "   PID=$pid  log_dir=$log_dir"
    sleep 2
done

echo ""
echo "============================================================"
echo "Batch $BATCH launched.  PIDs: ${PIDS[*]}"
echo ""
echo "Monitor:"
echo "  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
echo ""
echo "Check best F1 when done:"
echo "  for c in ${CONFIGS[*]}; do"
echo "    echo \"=\$c=\"; grep 'Macro F1:' $BASE_LOG/\$c/train.log | awk -F'Macro F1: ' '{print \$2}' | sort -nr | head -1"
echo "  done"
echo "============================================================"
