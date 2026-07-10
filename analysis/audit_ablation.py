"""Compute the "direction" column for the ablation table.

For each (component, dataset) cell, Delta = ablated F1 - full-model F1 at the
same random split. The "Dir." column counts how many of the 3 runs have
Delta < 0 (i.e., the expected sign).

Outputs paper/figures/ablation_audit.json with per-run F1s, paired deltas,
mean/std, and direction counts. The paper's Table 2 mean +/- std values are
preserved unchanged; this script only re-checks them against the logs and
supplies the "Dir." column.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

LOG_ROOT = Path('/root/autodl-tmp/logs-m=15')
MACRO_F1_RE = re.compile(r"Macro F1:\s*([0-9]+\.[0-9]+)%")
SEED_RE = re.compile(r"Split on seed (\d+)")


def best_f1(p: Path) -> float | None:
    if not p.is_file():
        return None
    best = None
    with p.open('r', errors='ignore') as f:
        for line in f:
            m = MACRO_F1_RE.search(line)
            if m:
                v = float(m.group(1))
                if best is None or v > best:
                    best = v
    return best


def seed_of(p: Path) -> int | None:
    if not p.is_file():
        return None
    with p.open('r', errors='ignore') as f:
        for _ in range(20):
            line = f.readline()
            m = SEED_RE.search(line)
            if m:
                return int(m.group(1))
    return None


# Full-model F1 per seed (computed from logs)
FULL_F1 = {
    # CCA Wiki-ZSL (innovation1) per seed
    ('cca', 'wiki'): {
        11222333: 75.03, 457365: 82.86, 99887766: 82.80,
        11223344: 84.92, 55667788: 86.40,
    },
    # CCA FewRel per seed
    ('cca', 'few'): {
        11222333: 90.28, 457365: 89.58, 99887766: 90.08,
        11223344: 83.60, 55667788: 83.43,
    },
    # CCA+ISCL Wiki-ZSL (innovation2 splits)
    ('cca_iscl', 'wiki'): {
        20260421: 84.06, 11223344: 88.65, 42424242: 84.17,
        77889900: 80.45, 33445566: 77.88,
    },
    # CCA+ISCL FewRel
    ('cca_iscl', 'few'): {
        11222333: 83.59, 66: 88.96, 99887766: 85.27,
        11223344: 89.47, 55667788: 85.57,
    },
}

# Configurations: (component_label, dataset, list-of-log-paths)
CFG = [
    # CCA components, Wiki-ZSL
    ('cca/no_zscore', 'wiki', 'cca',  [
        LOG_ROOT / 'logs_ablation_cca' / 'ablation_no_zscore' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_cca' / 'cca_no_zscore_seed11222333' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_cca' / 'cca_no_zscore_seed55667788' / 'train.log',
    ]),
    ('cca/no_grad_iso', 'wiki', 'cca', [
        LOG_ROOT / 'logs_ablation_cca' / 'ablation_no_grad_iso' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_cca' / 'cca_no_grad_iso_seed11222333' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_cca' / 'cca_no_grad_iso_seed55667788' / 'train.log',
    ]),
    # CCA components, FewRel
    ('cca/no_zscore', 'few', 'cca', [
        LOG_ROOT / 'logs_ablation_fewrel_cca' / 'ablation_no_zscore' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_cca_no_zscore_seed11222333' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_cca_no_zscore_seed55667788' / 'train.log',
    ]),
    ('cca/no_grad_iso', 'few', 'cca', [
        LOG_ROOT / 'logs_ablation_fewrel_cca' / 'ablation_no_grad_iso' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_cca_no_grad_iso_seed11222333' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_cca_no_grad_iso_seed55667788' / 'train.log',
    ]),
    # ISCL components, Wiki-ZSL
    ('iscl/no_cross_attn', 'wiki', 'cca_iscl', [
        LOG_ROOT / 'logs_ablation_iscl' / 'ablation_no_cross_attn' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_iscl' / 'iscl_no_cross_attn_seed11223344' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_iscl' / 'iscl_no_cross_attn_seed33445566' / 'train.log',
    ]),
    ('iscl/no_global_align', 'wiki', 'cca_iscl', [
        LOG_ROOT / 'logs_ablation_iscl' / 'ablation_no_global_align' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_iscl' / 'iscl_no_global_align_seed11223344' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_iscl' / 'iscl_no_global_align_seed33445566' / 'train.log',
    ]),
    ('iscl/no_supcon_warmup', 'wiki', 'cca_iscl', [
        LOG_ROOT / 'logs_ablation_iscl' / 'ablation_no_supcon_warmup' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_iscl' / 'iscl_no_supcon_warmup_seed11223344' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun' / 'logs_ablation_iscl' / 'iscl_no_supcon_warmup_seed33445566' / 'train.log',
    ]),
    # ISCL components, FewRel
    ('iscl/no_cross_attn', 'few', 'cca_iscl', [
        LOG_ROOT / 'logs_ablation_fewrel_iscl' / 'ablation_no_cross_attn' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_iscl_no_cross_attn_seed11223344' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_iscl_no_cross_attn_seed99887766' / 'train.log',
    ]),
    ('iscl/no_global_align', 'few', 'cca_iscl', [
        LOG_ROOT / 'logs_ablation_fewrel_iscl' / 'ablation_no_global_align' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_iscl_no_global_align_seed11223344' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_iscl_no_global_align_seed99887766' / 'train.log',
    ]),
    ('iscl/no_supcon_warmup', 'few', 'cca_iscl', [
        LOG_ROOT / 'logs_ablation_fewrel_iscl' / 'ablation_no_supcon_warmup' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_iscl_no_supcon_warmup_seed11223344' / 'train.log',
        LOG_ROOT / 'logs_ablation_rerun_fewrel' / 'fewrel_iscl_no_supcon_warmup_seed99887766' / 'train.log',
    ]),
]


def main() -> None:
    rows = []
    print(f"{'component':<22} {'ds':<5} {'paired deltas (seed: dF1)':<60} "
          f"{'mean':<8} {'std':<7} {'dir/3':<6}")
    print('-' * 120)
    for component, ds, ref_key, paths in CFG:
        cell = []
        for p in paths:
            seed = seed_of(p)
            f1 = best_f1(p)
            full = FULL_F1[(ref_key, ds)].get(seed) if seed is not None else None
            delta = (f1 - full) if (f1 is not None and full is not None) else None
            cell.append({'log': str(p), 'seed': seed, 'ablated_f1': f1,
                         'matched_full_f1': full, 'delta': delta})
        deltas = [c['delta'] for c in cell if c['delta'] is not None]
        if len(deltas) < 3:
            print(f"{component:<22} {ds:<5} INCOMPLETE: {cell}")
            continue
        dir_count = sum(1 for d in deltas if d < 0)
        mean = float(np.mean(deltas))
        std = float(np.std(deltas, ddof=1))
        delta_str = ', '.join(f"{c['seed']}:{c['delta']:+.2f}" for c in cell)
        print(f"{component:<22} {ds:<5} {delta_str:<60} "
              f"{mean:<+8.2f} {std:<7.2f} {dir_count}/3")
        rows.append({'component': component, 'dataset': ds,
                     'runs': cell, 'mean_delta': mean, 'std_delta': std,
                     'direction_count': dir_count})

    out = Path('/root/GLiREL/paper/figures/ablation_audit.json')
    out.write_text(json.dumps(rows, indent=2))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
