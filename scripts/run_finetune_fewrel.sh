#!/bin/bash
# Fine-tune GLiREL on FewRel dataset
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export PYTHONPATH=/root/GLiREL

cd /root/GLiREL

echo "========================================"
echo "Fine-tuning GLiREL on FewRel"
echo "========================================"

# Find the downloaded deberta-v3-large model path
DEBERTA_PATH=$(python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
from huggingface_hub import snapshot_download
path = snapshot_download('microsoft/deberta-v3-large', cache_dir='/root/autodl-tmp/hf_cache', local_files_only=True)
print(path)
" 2>/dev/null)
echo "DeBERTa path: $DEBERTA_PATH"

# Find the GLiREL pretrained model path
GLIREL_PATH=$(python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
from huggingface_hub import snapshot_download
path = snapshot_download('jackboyla/glirel-large-v0', cache_dir='/root/autodl-tmp/hf_cache', local_files_only=True)
print(path)
" 2>/dev/null)
echo "GLiREL pretrained path: $GLIREL_PATH"

python3 train.py --config configs/config_few_rel.yaml 2>&1 | tee logs/finetune_fewrel.log

echo "FewRel fine-tuning complete!"
