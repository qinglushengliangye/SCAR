#!/bin/bash
# Data preparation script for FewRel and Wiki-ZSL
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTHONPATH=/root/GLiREL

cd /root/GLiREL/data

echo "========================================"
echo "Step 1: Prepare FewRel dataset"
echo "========================================"

# process_few_rel.py loads from HuggingFace datasets 'few_rel'
# Patch it to use mirror
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
from datasets import load_dataset, concatenate_datasets
import json

print('Loading FewRel dataset from HF mirror...')
dataset = load_dataset('few_rel', cache_dir='/root/autodl-tmp/hf_cache')
ds_train = dataset['train_wiki'].shuffle(seed=42)
ds_val = dataset['val_wiki'].shuffle(seed=42)
from datasets import concatenate_datasets
ds = concatenate_datasets([ds_train, ds_val])
print(f'Number of examples: {len(ds)}')

data = ds.to_dict()

def transform_few_rel(data):
    transformed_data = []
    for i in range(len(data['relation'])):
        ner_entries = []
        relations = []
        tokens = data['tokens'][i]
        head = data['head'][i]
        tail = data['tail'][i]
        relation = data['relation'][i]
        relation_text = data['names'][i][0]

        head_start, head_end = head['indices'][0][0], head['indices'][0][-1]
        head_text = ' '.join(tokens[head_start:head_end+1])
        ner_entries.append([head_start, head_end, head['type'], head_text])
        
        tail_start, tail_end = tail['indices'][0][0], tail['indices'][0][-1]
        tail_text = ' '.join(tokens[tail_start: tail_end+1])
        ner_entries.append([tail_start, tail_end, tail['type'], tail_text])
        
        relations.append({
            'head': {'mention': head_text, 'position': [head_start, head_end], 'type': head['type']},
            'tail': {'mention': tail_text, 'position': [tail_start, tail_end], 'type': tail['type']},
            'relation_id': relation,
            'relation_text': relation_text,
        })

        transformed_data.append({
            'ner': ner_entries,
            'relations': relations,
            'tokenized_text': tokens,
        })

    return transformed_data

transformed_data = transform_few_rel(data)
save_path = '/root/GLiREL/data/few_rel_all.jsonl'
with open(save_path, 'w') as f:
    for item in transformed_data:
        f.write(json.dumps(item) + '
')
print(f'Saved {len(transformed_data)} examples to {save_path}')
"

echo "FewRel data prepared!"

echo "========================================"
echo "Step 2: Prepare Wiki-ZSL dataset"
echo "========================================"

# Wiki-ZSL downloads via gdown (Google Drive) - we need an alternative
# The file ID is 1TMYvAbe9wsB5GiWcUL5bMAs9x6CpvnAj
# Try downloading via gdown with a proxy-free approach

cd /root/GLiREL/data

if [ ! -f wiki_all.json ]; then
    echo 'Downloading wiki_all.json via gdown...'
    python3 -c "
import gdown
file_id = '1TMYvAbe9wsB5GiWcUL5bMAs9x6CpvnAj'
gdown.download(id=file_id, output='wiki_all.json', quiet=False)
print('wiki_all.json downloaded!')
"
else
    echo 'wiki_all.json already exists, skipping download.'
fi

echo "Processing Wiki-ZSL..."
python3 /root/GLiREL/data/process_wiki_zsl.py

echo "Wiki-ZSL data prepared!"
echo "Data preparation complete!"
