#!/bin/bash
# Fine-tune GLiREL on Wiki-ZSL dataset
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export PYTHONPATH=/root/GLiREL

cd /root/GLiREL

echo "========================================"
echo "Fine-tuning GLiREL on Wiki-ZSL"
echo "========================================"

python3 train.py --config configs/config_wiki_zsl.yaml 2>&1 | tee logs/finetune_wiki_zsl.log

echo "Wiki-ZSL fine-tuning complete!"
