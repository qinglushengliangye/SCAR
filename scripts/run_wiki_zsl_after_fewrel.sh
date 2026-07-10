#!/bin/bash
# Wait for FewRel training to finish, then start Wiki-ZSL training
echo "Waiting for FewRel training (PID monitoring)..."

while ps aux | grep -q '[t]rain.py.*config_few_rel_repro'; do
    sleep 60
    echo "$(date): FewRel still running..."
done

echo "$(date): FewRel finished! Starting Wiki-ZSL training..."
sleep 10  # Let GPU memory free up

cd /root/GLiREL
rm -rf logs_repro/wiki_zsl
mkdir -p logs_repro/wiki_zsl

OMP_NUM_THREADS=8 TRANSFORMERS_OFFLINE=1 python3 train.py \
    --config configs/config_wiki_zsl_repro.yaml \
    --log_dir logs_repro/wiki_zsl > /root/autodl-tmp/finetune_wiki_zsl.log 2>&1

echo "$(date): Wiki-ZSL training finished!"
