#!/bin/bash
# ============================================================
# 生成 10 个新的 FewRel ablation config 文件
# 每个 ablation 在 2 个新 seed 上重复（除原 seed 外）
#
# CCA 2 ablations:  新 seeds = 11222333, 55667788  (原 seed: 11223344)
# ISCL 3 ablations: 新 seeds = 11223344, 99887766  (原 seed: 11222333)
# ============================================================
set -e

SRC_DIR=/root/GLiREL/configs
OUT_DIR=/root/GLiREL/configs/rerun_ablation
mkdir -p "$OUT_DIR"

CCA_ABLATIONS=(no_grad_iso no_zscore)
CCA_SEEDS=(11222333 55667788)
CCA_ORIG_SEED=11223344

ISCL_ABLATIONS=(no_cross_attn no_global_align no_supcon_warmup)
ISCL_SEEDS=(11223344 99887766)
ISCL_ORIG_SEED=11222333

count=0
for abl in "${CCA_ABLATIONS[@]}"; do
    src=$SRC_DIR/config_ablation_fewrel_cca_${abl}.yaml
    for new_seed in "${CCA_SEEDS[@]}"; do
        dst=$OUT_DIR/fewrel_cca_${abl}_seed${new_seed}.yaml
        sed "s/^seed: ${CCA_ORIG_SEED}/seed: ${new_seed}/" "$src" > "$dst"
        actual=$(grep "^seed:" "$dst" | awk '{print $2}')
        if [ "$actual" != "$new_seed" ]; then
            echo "ERROR: seed substitution failed for $dst (got $actual)"
            exit 1
        fi
        count=$((count+1))
        echo "  [$count] $(basename $dst)  seed=$actual"
    done
done

for abl in "${ISCL_ABLATIONS[@]}"; do
    src=$SRC_DIR/config_ablation_fewrel_iscl_${abl}.yaml
    for new_seed in "${ISCL_SEEDS[@]}"; do
        dst=$OUT_DIR/fewrel_iscl_${abl}_seed${new_seed}.yaml
        sed "s/^seed: ${ISCL_ORIG_SEED}/seed: ${new_seed}/" "$src" > "$dst"
        actual=$(grep "^seed:" "$dst" | awk '{print $2}')
        if [ "$actual" != "$new_seed" ]; then
            echo "ERROR: seed substitution failed for $dst (got $actual)"
            exit 1
        fi
        count=$((count+1))
        echo "  [$count] $(basename $dst)  seed=$actual"
    done
done

echo ""
echo "Generated $count FewRel config files under $OUT_DIR"
