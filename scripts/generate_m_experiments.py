#!/usr/bin/env python3
"""
Generate config YAML files and shell scripts for m=5 and m=10 experiments.

Covers: Baseline / Innovation1 (CCA) / Innovation2 (ISCL)
        x WikiZSL / FewRel
        x 5 seeds each
= 60 configs + 60 shell scripts
"""

import os
import yaml
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_SRC = PROJECT_ROOT / "configs"
CONFIGS_DST = PROJECT_ROOT / "configs_m"
SCRIPTS_DST = PROJECT_ROOT / "scripts_m"

M_VALUES = [5, 10]
SEEDS = {
    1: 11222333,
    2: 457365,
    3: 99887766,
    4: 11223344,
    5: 55667788,
}

LOG_ROOT_FMT = "/root/autodl-tmp/logs-m={m}"

EXPERIMENTS = [
    {
        "method": "repro",
        "method_label": "Baseline",
        "dataset": "wiki_zsl",
        "dataset_label": "WikiZSL",
        "template": "config_wiki_zsl_repro.yaml",
        "config_fmt": "config_wiki_zsl_repro_m{m}_exp{n}.yaml",
        "script_fmt": "run_repro_wikizsl_m{m}_exp{n}.sh",
        "log_subdir": "repro_wikizsl",
        "name_base": "large",
    },
    {
        "method": "repro",
        "method_label": "Baseline",
        "dataset": "few_rel",
        "dataset_label": "FewRel",
        "template": "config_few_rel_repro.yaml",
        "config_fmt": "config_few_rel_repro_m{m}_exp{n}.yaml",
        "script_fmt": "run_repro_fewrel_m{m}_exp{n}.sh",
        "log_subdir": "repro_fewrel",
        "name_base": "large",
    },
    {
        "method": "cascade",
        "method_label": "Innovation1/CCA",
        "dataset": "wiki_zsl",
        "dataset_label": "WikiZSL",
        "template": "config_wiki_zsl_cascade_exp1.yaml",
        "config_fmt": "config_wiki_zsl_cascade_m{m}_exp{n}.yaml",
        "script_fmt": "run_innovation1_wikizsl_m{m}_exp{n}.sh",
        "log_subdir": "cascade_wikizsl",
        "name_base": "large_cascade_v2",
    },
    {
        "method": "cascade",
        "method_label": "Innovation1/CCA",
        "dataset": "few_rel",
        "dataset_label": "FewRel",
        "template": "config_few_rel_cascade_exp1.yaml",
        "config_fmt": "config_few_rel_cascade_m{m}_exp{n}.yaml",
        "script_fmt": "run_innovation1_fewrel_m{m}_exp{n}.sh",
        "log_subdir": "cascade_fewrel",
        "name_base": "large_cascade_v2",
    },
    {
        "method": "innovation2",
        "method_label": "Innovation2/ISCL",
        "dataset": "wiki_zsl",
        "dataset_label": "WikiZSL",
        "template": "config_wiki_zsl_innovation2.yaml",
        "config_fmt": "config_wiki_zsl_innovation2_m{m}_exp{n}.yaml",
        "script_fmt": "run_innovation2_wikizsl_m{m}_exp{n}.sh",
        "log_subdir": "innovation2_wikizsl",
        "name_base": "large_innovation2",
    },
    {
        "method": "innovation2",
        "method_label": "Innovation2/ISCL",
        "dataset": "few_rel",
        "dataset_label": "FewRel",
        "template": "config_few_rel_innovation2_exp1.yaml",
        "config_fmt": "config_few_rel_innovation2_m{m}_exp{n}.yaml",
        "script_fmt": "run_innovation2_fewrel_m{m}_exp{n}.sh",
        "log_subdir": "innovation2_fewrel",
        "name_base": "large_innovation2",
    },
]

METHOD_LABELS_CN = {
    "repro": "基线复现",
    "cascade": "创新点1（CCA）",
    "innovation2": "创新点2（ISCL）",
}

DATASET_LABELS_CN = {
    "wiki_zsl": "Wiki-ZSL",
    "few_rel": "FewRel",
}


def load_yaml_raw(path: Path) -> str:
    """Read template YAML as raw text."""
    return path.read_text(encoding="utf-8")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml_preserving_comments(template_text: str, overrides: dict, out_path: Path):
    """
    Write a YAML config by modifying specific fields in the raw template text.
    This preserves comments and formatting from the original template.
    Fields not found in the template are appended at the end.
    """
    lines = template_text.splitlines()
    new_lines = []
    found_keys = set()
    for line in lines:
        replaced = False
        for key, val in overrides.items():
            stripped = line.lstrip()
            if stripped.startswith(f"{key}:") and not stripped.startswith(f"{key}_"):
                indent = line[: len(line) - len(stripped)]
                new_lines.append(f"{indent}{key}: {val}")
                found_keys.add(key)
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    for key, val in overrides.items():
        if key not in found_keys:
            new_lines.append(f"{key}: {val}")

    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def generate_shell_script(
    exp: dict, m: int, n: int, seed: int, config_filename: str
) -> str:
    method_cn = METHOD_LABELS_CN[exp["method"]]
    dataset_cn = DATASET_LABELS_CN[exp["dataset"]]
    log_root = LOG_ROOT_FMT.format(m=m)
    log_dir = f"{log_root}/{exp['log_subdir']}/exp{n}"

    return f"""#!/bin/bash
# ============================================================
# {method_cn} - {dataset_cn} | m={m} 实验{n}（seed={seed}）
# ============================================================
set -e

export PYTHONPATH=/root/GLiREL
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LOG_DIR="{log_dir}"
mkdir -p "$LOG_DIR"

echo "========================================="
echo " {method_cn} {dataset_cn} m={m} 实验{n} | seed={seed}"
echo " 日志目录: $LOG_DIR"
echo " 开始时间: $(date)"
echo "========================================="

cd /root/GLiREL

python3 train.py \\
    --config configs_m/{config_filename} \\
    --log_dir "$LOG_DIR" \\
    2>&1 | tee "$LOG_DIR/train.log"

echo "========================================="
echo " 训练完成，日志保存于 $LOG_DIR/train.log"
echo " 结束时间: $(date)"
echo "========================================="
"""


def generate_batch_script(all_scripts: list[str]) -> str:
    header = """#!/bin/bash
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

"""
    body_lines = []
    for script_name in all_scripts:
        body_lines.append(f'if [ -z "$FILTER" ] || echo "{script_name}" | grep -q "$FILTER"; then')
        body_lines.append(f'    echo ">>> 运行 {script_name} ..."')
        body_lines.append(f'    bash "$SCRIPT_DIR/{script_name}"')
        body_lines.append(f'    DONE=$((DONE + 1))')
        body_lines.append(f'fi')
        body_lines.append(f'TOTAL=$((TOTAL + 1))')
        body_lines.append("")

    footer = """
echo "========================================="
echo " 全部完成: $DONE / $TOTAL 个实验已执行"
echo "========================================="
"""
    return header + "\n".join(body_lines) + footer


def main():
    CONFIGS_DST.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DST.mkdir(parents=True, exist_ok=True)

    all_script_names = []
    config_count = 0
    script_count = 0

    for exp in EXPERIMENTS:
        template_path = CONFIGS_SRC / exp["template"]
        template_text = load_yaml_raw(template_path)

        for m in M_VALUES:
            for n, seed in SEEDS.items():
                config_filename = exp["config_fmt"].format(m=m, n=n)
                script_filename = exp["script_fmt"].format(m=m, n=n)

                log_root = LOG_ROOT_FMT.format(m=m)
                root_dir_val = f"{log_root}/{exp['log_subdir']}"
                name_val = f'"{exp["name_base"]}_m{m}"'

                overrides = {
                    "num_unseen_rel_types": m,
                    "seed": seed,
                    "root_dir": root_dir_val,
                    "name": name_val,
                }

                config_out = CONFIGS_DST / config_filename
                dump_yaml_preserving_comments(template_text, overrides, config_out)
                config_count += 1

                script_content = generate_shell_script(exp, m, n, seed, config_filename)
                script_out = SCRIPTS_DST / script_filename
                script_out.write_text(script_content, encoding="utf-8")
                script_out.chmod(0o755)
                script_count += 1

                all_script_names.append(script_filename)

    batch_content = generate_batch_script(all_script_names)
    batch_out = SCRIPTS_DST / "run_all_m_experiments.sh"
    batch_out.write_text(batch_content, encoding="utf-8")
    batch_out.chmod(0o755)

    print(f"Generated {config_count} config files in {CONFIGS_DST}")
    print(f"Generated {script_count} shell scripts in {SCRIPTS_DST}")
    print(f"Generated batch runner: {batch_out}")


if __name__ == "__main__":
    main()
