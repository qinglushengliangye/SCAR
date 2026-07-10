#!/bin/bash
# ============================================================
# 批量运行全部 m=5 / m=10 补充实验
# 用法:
#   bash scripts_m/run_all_m_experiments.sh           # 运行全部
#   bash scripts_m/run_all_m_experiments.sh m5        # 只运行 m=5
#   bash scripts_m/run_all_m_experiments.sh m10       # 只运行 m=10
#   bash scripts_m/run_all_m_experiments.sh repro     # 只运行基线
#   bash scripts_m/run_all_m_experiments.sh cascade   # 只运行创新点1
#   bash scripts_m/run_all_m_experiments.sh innovation2  # 只运行创新点2
# ============================================================
set -e

FILTER="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOTAL=0
DONE=0

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m5_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m5_exp1.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m5_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m5_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m5_exp2.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m5_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m5_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m5_exp3.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m5_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m5_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m5_exp4.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m5_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m5_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m5_exp5.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m5_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m10_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m10_exp1.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m10_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m10_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m10_exp2.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m10_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m10_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m10_exp3.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m10_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m10_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m10_exp4.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m10_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_wikizsl_m10_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_wikizsl_m10_exp5.sh ..."
    bash "$SCRIPT_DIR/run_repro_wikizsl_m10_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m5_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m5_exp1.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m5_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m5_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m5_exp2.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m5_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m5_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m5_exp3.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m5_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m5_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m5_exp4.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m5_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m5_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m5_exp5.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m5_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m10_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m10_exp1.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m10_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m10_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m10_exp2.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m10_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m10_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m10_exp3.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m10_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m10_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m10_exp4.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m10_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_repro_fewrel_m10_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_repro_fewrel_m10_exp5.sh ..."
    bash "$SCRIPT_DIR/run_repro_fewrel_m10_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m5_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m5_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m5_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m5_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m5_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m5_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m5_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m5_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m5_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m5_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m5_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m5_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m5_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m5_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m5_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m10_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m10_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m10_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m10_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m10_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m10_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m10_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m10_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m10_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m10_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m10_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m10_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_wikizsl_m10_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_wikizsl_m10_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_wikizsl_m10_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m5_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m5_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m5_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m5_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m5_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m5_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m5_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m5_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m5_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m5_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m5_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m5_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m5_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m5_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m5_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m10_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m10_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m10_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m10_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m10_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m10_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m10_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m10_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m10_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m10_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m10_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m10_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation1_fewrel_m10_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation1_fewrel_m10_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation1_fewrel_m10_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m5_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m5_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m5_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m5_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m5_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m5_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m5_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m5_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m5_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m5_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m5_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m5_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m5_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m5_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m5_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m10_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m10_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m10_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m10_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m10_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m10_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m10_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m10_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m10_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m10_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m10_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m10_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_wikizsl_m10_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_wikizsl_m10_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_wikizsl_m10_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m5_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m5_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m5_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m5_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m5_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m5_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m5_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m5_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m5_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m5_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m5_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m5_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m5_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m5_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m5_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m10_exp1.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m10_exp1.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m10_exp1.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m10_exp2.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m10_exp2.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m10_exp2.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m10_exp3.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m10_exp3.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m10_exp3.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m10_exp4.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m10_exp4.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m10_exp4.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

if [ -z "$FILTER" ] || echo "run_innovation2_fewrel_m10_exp5.sh" | grep -q "$FILTER"; then
    echo ">>> 运行 run_innovation2_fewrel_m10_exp5.sh ..."
    bash "$SCRIPT_DIR/run_innovation2_fewrel_m10_exp5.sh"
    DONE=$((DONE + 1))
fi
TOTAL=$((TOTAL + 1))

echo "========================================="
echo " 全部完成: $DONE / $TOTAL 个实验已执行"
echo "========================================="
