#!/bin/bash
set -e
cd /root/GLiREL

BS=192
LOG=/root/GLiREL/paper/figures/inference_log.txt

echo "=== Full-dataset inference pipeline (93K, bs=$BS) ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

echo "=== [1/3] Baseline ===" | tee -a $LOG
python3 analysis/full_dataset_analysis.py \
  --method baseline \
  --ckpt-dir /root/autodl-tmp/logs-m=15/logs_repro_wikizsl/logs_repro_exp1/model_6300 \
  --gpu 0 --batch-size $BS 2>&1 | tee -a $LOG

echo "=== [2/3] CCA ===" | tee -a $LOG
python3 analysis/full_dataset_analysis.py \
  --method cca \
  --ckpt-dir /root/autodl-tmp/logs-m=15/logs_innovation1_wikizsl/logs_innovation1_exp5/model_3900 \
  --gpu 0 --batch-size $BS 2>&1 | tee -a $LOG

echo "=== [3/3] CCA+ISCL ===" | tee -a $LOG
python3 analysis/full_dataset_analysis.py \
  --method cca_iscl \
  --ckpt-dir /root/autodl-tmp/logs-m=15/logs_innovation2_wikizsl/logs_innovation2_split1/model_11400 \
  --gpu 0 --batch-size $BS 2>&1 | tee -a $LOG

echo "=== Generating figures ===" | tee -a $LOG
python3 analysis/generate_figures.py 2>&1 | tee -a $LOG

echo "=== All complete: $(date) ===" | tee -a $LOG
