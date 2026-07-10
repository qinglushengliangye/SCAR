#!/bin/bash
# GLiREL Full Reproduction Script
# Uses hf-mirror.com for model downloads (China-accessible)
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_VERBOSITY=info
export HF_HOME=/root/autodl-tmp/hf_cache

echo "========================================"
echo "Step 1: Download DeBERTa-v3-large model"
echo "========================================"

mkdir -p /root/autodl-tmp/hf_cache
mkdir -p /root/autodl-tmp/models/deberta-v3-large

# Download deberta-v3-large using huggingface_hub
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
from huggingface_hub import snapshot_download
print('Downloading microsoft/deberta-v3-large...')
path = snapshot_download(
    repo_id='microsoft/deberta-v3-large',
    cache_dir='/root/autodl-tmp/hf_cache',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*', 'rust_model*'],
)
print(f'Downloaded to: {path}')
"

echo "========================================"
echo "Step 2: Download pretrained GLiREL model"
echo "========================================"

python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
from huggingface_hub import snapshot_download
print('Downloading jackboyla/glirel-large-v0...')
path = snapshot_download(
    repo_id='jackboyla/glirel-large-v0',
    cache_dir='/root/autodl-tmp/hf_cache',
)
print(f'Downloaded to: {path}')
"

echo "All downloads complete!"
echo "HF cache at: /root/autodl-tmp/hf_cache"
