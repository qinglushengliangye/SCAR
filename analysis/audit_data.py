"""
Data authenticity audit.

Walks every train.log under /root/autodl-tmp/logs-m={5,10,15}/ and
extracts the maximum Macro F1 reported during training (line of the
form `Macro P: ...\tMacro R: ...\tMacro F1: XX.XX%`).

Outputs:
  - paper/figures/verified_per_seed.json : {dataset, m, method, exp_k -> best_f1}
  - paper/figures/audit_report.md        : human-readable summary table and discrepancy list
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

LOG_ROOT = Path('/root/autodl-tmp')
OUT_DIR = Path('/root/GLiREL/paper/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MACRO_F1_RE = re.compile(r"Macro F1:\s*([0-9]+\.[0-9]+)%")


def best_f1_from_log(path: Path) -> float | None:
    if not path.is_file():
        return None
    best = None
    with path.open('r', errors='ignore') as f:
        for line in f:
            m = MACRO_F1_RE.search(line)
            if m:
                v = float(m.group(1))
                if best is None or v > best:
                    best = v
    return best


# Mapping: (dataset, m, method) -> list of (exp_label, log_path)
DATASETS = ['wiki', 'few']
METHODS = ['baseline', 'cca', 'cca_iscl']


def m5_m10_paths(m: int, dataset: str, method: str) -> list[tuple[str, Path]]:
    """m=5, m=10 layout."""
    root = LOG_ROOT / f'logs-m={m}'
    method_dir = {
        'baseline': 'repro',
        'cca': 'cascade',
        'cca_iscl': 'innovation2',
    }[method]
    suffix = 'wikizsl' if dataset == 'wiki' else 'fewrel'
    base = root / f'{method_dir}_{suffix}'
    return [(f'exp{i}', base / f'exp{i}' / 'train.log') for i in range(1, 6)]


def m15_paths(dataset: str, method: str) -> list[tuple[str, Path]]:
    """m=15 layout (longer prefixes; CCA+ISCL Wiki-ZSL uses split{1,2,4,5,6})."""
    root = LOG_ROOT / 'logs-m=15'
    suffix = 'wikizsl' if dataset == 'wiki' else 'fewrel'
    if method == 'baseline':
        base = root / f'logs_repro_{suffix}'
        return [(f'exp{i}', base / f'logs_repro_exp{i}' / 'train.log') for i in range(1, 6)]
    if method == 'cca':
        base = root / f'logs_innovation1_{suffix}'
        return [(f'exp{i}', base / f'logs_innovation1_exp{i}' / 'train.log') for i in range(1, 6)]
    # cca_iscl
    base = root / f'logs_innovation2_{suffix}'
    if dataset == 'wiki':
        splits = [1, 2, 4, 5, 6]  # exp1..exp5 -> split1, split2, split4, split5, split6
        return [(f'exp{i+1}', base / f'logs_innovation2_split{s}' / 'train.log') for i, s in enumerate(splits)]
    return [(f'exp{i}', base / f'logs_innovation2_exp{i}' / 'train.log') for i in range(1, 6)]


def build_verified() -> dict:
    out: dict = {}
    for m in (5, 10, 15):
        for dataset in DATASETS:
            for method in METHODS:
                paths = m15_paths(dataset, method) if m == 15 else m5_m10_paths(m, dataset, method)
                cell = {}
                for exp_label, log_path in paths:
                    v = best_f1_from_log(log_path)
                    cell[exp_label] = {
                        'best_f1': v,
                        'log': str(log_path),
                        'exists': log_path.is_file(),
                    }
                out.setdefault(f'm{m}', {}).setdefault(dataset, {})[method] = cell
    return out


# Paper-claimed numbers (from /root/GLiREL/paper/appendix.tex Tables 6/7/8 and main Table 1)
PAPER_NUMBERS = {
    'm5': {
        'wiki': {
            'baseline':  {'exp1': 88.44, 'exp2': 95.81, 'exp3': 77.10, 'exp4': 81.28, 'exp5': 85.36},
            'cca':       {'exp1': 97.46, 'exp2': 96.84, 'exp3': 90.10, 'exp4': 93.42, 'exp5': 94.24},
            'cca_iscl':  {'exp1': 96.84, 'exp2': 94.45, 'exp3': 93.56, 'exp4': 92.59, 'exp5': 92.33},
        },
        'few': {
            'baseline':  {'exp1': 83.36, 'exp2': 81.24, 'exp3': 97.77, 'exp4': 90.88, 'exp5': 81.19},
            'cca':       {'exp1': 92.14, 'exp2': 94.27, 'exp3': 90.12, 'exp4': 96.94, 'exp5': 98.94},
            'cca_iscl':  {'exp1': 98.16, 'exp2': 94.42, 'exp3': 94.41, 'exp4': 98.57, 'exp5': 94.55},
        },
    },
    'm10': {
        'wiki': {
            'baseline':  {'exp1': 77.76, 'exp2': 77.91, 'exp3': 80.36, 'exp4': 82.35, 'exp5': 82.30},
            'cca':       {'exp1': 93.46, 'exp2': 90.05, 'exp3': 88.56, 'exp4': 89.04, 'exp5': 83.28},
            'cca_iscl':  {'exp1': 91.03, 'exp2': 90.27, 'exp3': 90.40, 'exp4': 86.19, 'exp5': 91.44},
        },
        'few': {
            'baseline':  {'exp1': 86.08, 'exp2': 89.37, 'exp3': 83.53, 'exp4': 84.90, 'exp5': 86.63},
            'cca':       {'exp1': 93.46, 'exp2': 91.09, 'exp3': 86.03, 'exp4': 90.75, 'exp5': 86.26},
            'cca_iscl':  {'exp1': 87.75, 'exp2': 88.67, 'exp3': 90.27, 'exp4': 93.84, 'exp5': 88.14},
        },
    },
    'm15': {
        'wiki': {
            'baseline':  {'exp1': 79.84, 'exp2': 81.53, 'exp3': 76.43, 'exp4': 69.24, 'exp5': 72.37},
            'cca':       {'exp1': 75.03, 'exp2': 82.86, 'exp3': 82.80, 'exp4': 84.92, 'exp5': 86.40},
            'cca_iscl':  {'exp1': 84.06, 'exp2': 88.65, 'exp3': 84.17, 'exp4': 80.45, 'exp5': 77.88},
        },
        'few': {
            'baseline':  {'exp1': 89.30, 'exp2': 80.11, 'exp3': 75.44, 'exp4': 83.87, 'exp5': 73.79},
            'cca':       {'exp1': 90.28, 'exp2': 89.58, 'exp3': 90.08, 'exp4': 83.60, 'exp5': 83.43},
            'cca_iscl':  {'exp1': 83.59, 'exp2': 88.96, 'exp3': 85.27, 'exp4': 89.47, 'exp5': 85.57},
        },
    },
}

# Numbers currently inside analysis/statistical_tests.py DATA dict
SCRIPT_NUMBERS = {
    'm5':  {'wiki': {'baseline': [88.44, 95.81, 77.10, 81.28, 85.36],
                     'cca':      [97.46, 96.84, 90.10, 93.42, 94.24],
                     'cca_iscl': [96.84, 94.45, 93.56, 92.59, 92.33]},
            'few':  {'baseline': [83.36, 81.24, 97.77, 90.88, 81.19],
                     'cca':      [92.14, 94.27, 90.12, 96.94, 98.94],
                     'cca_iscl': [98.16, 94.42, 94.41, 98.57, 94.55]}},
    'm10': {'wiki': {'baseline': [77.76, 77.91, 80.36, 82.35, 82.30],   # was 54.00, corrected to log value 82.35
                     'cca':      [93.46, 90.05, 88.56, 89.04, 83.28],
                     'cca_iscl': [91.03, 90.27, 90.40, 86.19, 91.44]},
            'few':  {'baseline': [86.08, 89.37, 83.53, 84.90, 86.63],
                     'cca':      [93.46, 91.09, 86.03, 90.75, 86.26],
                     'cca_iscl': [87.75, 88.67, 90.27, 93.84, 88.14]}},
    'm15': {'wiki': {'baseline': [79.84, 81.53, 76.43, 69.24, 72.37],
                     'cca':      [75.03, 82.86, 82.80, 84.92, 86.40],
                     'cca_iscl': [84.06, 88.65, 84.17, 80.45, 77.88]},
            'few':  {'baseline': [89.30, 80.11, 75.44, 83.87, 73.79],
                     'cca':      [90.28, 89.58, 90.08, 83.60, 83.43],
                     'cca_iscl': [83.59, 88.96, 85.27, 89.47, 85.57]}},
}


def write_report(verified: dict, path: Path) -> int:
    lines: list[str] = []
    lines.append('# Data Authenticity Audit Report\n')
    lines.append('Source: greps `Macro F1: XX.XX%` from every `train.log` under\n'
                 '`/root/autodl-tmp/logs-m={5,10,15}/` and takes the maximum.\n\n')
    lines.append('Columns: exp1 .. exp5 = the five independent splits as named in the paper.\n')
    lines.append('"PAPER" = value in paper/appendix.tex Tables 6/7/8.\n')
    lines.append('"SCRIPT" = value in analysis/statistical_tests.py DATA dict.\n')
    lines.append('"LOG" = best `Macro F1` observed in the corresponding train.log.\n')
    lines.append('\n---\n')

    discrepancies: list[str] = []
    missing: list[str] = []

    for m_key in ('m5', 'm10', 'm15'):
        for dataset in DATASETS:
            for method in METHODS:
                lines.append(f'\n## {m_key} / {dataset} / {method}\n')
                lines.append('| exp | PAPER | SCRIPT | LOG | match (paper) | match (script) |\n')
                lines.append('|-----|-------|--------|-----|---------------|----------------|\n')
                paper_cell = PAPER_NUMBERS[m_key][dataset][method]
                script_list = SCRIPT_NUMBERS[m_key][dataset][method]
                ver_cell = verified[m_key][dataset][method]
                for i, exp in enumerate(['exp1', 'exp2', 'exp3', 'exp4', 'exp5']):
                    p = paper_cell[exp]
                    s = script_list[i]
                    lg = ver_cell[exp]['best_f1']
                    p_match = '-' if lg is None else ('OK' if abs(lg - p) <= 0.05 else f'DIFF {abs(lg - p):.2f}')
                    s_match = '-' if lg is None else ('OK' if abs(lg - s) <= 0.05 else f'DIFF {abs(lg - s):.2f}')
                    log_str = '(missing)' if lg is None else f'{lg:.2f}'
                    lines.append(f'| {exp} | {p:.2f} | {s:.2f} | {log_str} | {p_match} | {s_match} |\n')
                    if lg is None:
                        missing.append(f'{m_key}/{dataset}/{method}/{exp}: log not found at {ver_cell[exp]["log"]}')
                    else:
                        if abs(lg - p) > 0.05:
                            discrepancies.append(
                                f'{m_key}/{dataset}/{method}/{exp}: paper={p:.2f} vs log={lg:.2f} (delta={lg - p:+.2f})')
                        if abs(lg - s) > 0.05:
                            discrepancies.append(
                                f'{m_key}/{dataset}/{method}/{exp}: script={s:.2f} vs log={lg:.2f} (delta={lg - s:+.2f})')

    lines.append('\n---\n\n## Summary\n\n')
    lines.append(f'- Total cells audited: 90 (6 settings x 3 methods x 5 exps)\n')
    lines.append(f'- Missing logs: {len(missing)}\n')
    lines.append(f'- Cells where PAPER or SCRIPT disagrees with LOG (>0.05 pp): {len(discrepancies)}\n\n')
    if missing:
        lines.append('### Missing logs\n')
        for x in missing:
            lines.append(f'- {x}\n')
        lines.append('\n')
    if discrepancies:
        lines.append('### Discrepancies\n')
        for x in discrepancies:
            lines.append(f'- {x}\n')

    path.write_text(''.join(lines))
    return len(discrepancies) + len(missing)


def main() -> None:
    verified = build_verified()
    (OUT_DIR / 'verified_per_seed.json').write_text(json.dumps(verified, indent=2))
    n = write_report(verified, OUT_DIR / 'audit_report.md')
    print(f'verified_per_seed.json + audit_report.md written to {OUT_DIR}')
    print(f'issues (missing + discrepancies): {n}')


if __name__ == '__main__':
    main()
