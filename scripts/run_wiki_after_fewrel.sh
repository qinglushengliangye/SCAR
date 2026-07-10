#!/bin/bash
# Wait for FewRel to finish, then launch Wiki-ZSL
FEWREL_PID=$(cat /root/GLiREL/logs_cascade/fewrel/train.pid 2>/dev/null)
echo "Waiting for FewRel (PID $FEWREL_PID) to finish..."
if [ -n "$FEWREL_PID" ]; then
    while kill -0 $FEWREL_PID 2>/dev/null; do
        sleep 30
    done
fi
echo "FewRel finished. Starting Wiki-ZSL cascade training..."
cd /root/GLiREL
python train.py --config configs/config_wiki_zsl_cascade.yaml --log_dir logs_cascade/wiki_zsl > logs_cascade/wiki_zsl/train.log 2>&1
echo "Wiki-ZSL cascade training complete."
