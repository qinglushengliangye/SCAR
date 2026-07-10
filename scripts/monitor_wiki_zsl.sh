#!/bin/bash
# 持续监控 WikiZSL 训练，每60秒打印最新评估
LOG=/root/GLiREL/logs_cascade/wiki_zsl/train.log
BESTF1=0
BESTSTEP=0
echo "[Monitor] 开始监控 $(date)"
while true; do
  sleep 60
  LATEST=$(grep -E '(Step=|Micro F1|Macro F1|Validation metric improved)' $LOG | tail -6)
  echo "[Monitor $(date +%H:%M:%S)] $LATEST"
  # 提取当前最佳 Micro F1
  CURF1=$(grep 'Micro F1' $LOG | grep -oP 'Micro F1: \K[0-9.]+' | sort -n | tail -1)
  if [ ! -z "$CURF1" ]; then
    if (( $(echo "$CURF1 > $BESTF1" | bc -l) )); then
      BESTF1=$CURF1
      echo "[Monitor] *** 新最高 Micro F1: $BESTF1 ***"
    fi
  fi
  # 检查进程是否还在
  PID=$(cat /root/GLiREL/logs_cascade/wiki_zsl/train.pid 2>/dev/null)
  if [ -z "$PID" ] || ! kill -0 $PID 2>/dev/null; then
    echo "[Monitor] 训练已结束！最终最佳 Micro F1=$BESTF1"
    break
  fi
done
