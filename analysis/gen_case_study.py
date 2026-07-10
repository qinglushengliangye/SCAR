"""Extract representative case study examples from cached predictions."""
import sys
sys.path.insert(0, '/root/GLiREL')

import json
import os
import pickle
import argparse
import random
import numpy as np

from glirel import GLiREL
from glirel.model import load_config_as_namespace
import torch

OUTPUT_DIR = '/root/GLiREL/paper/figures'

cache_path = f'{OUTPUT_DIR}/analysis_cache.pkl'
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

results_dict = cache['results_dict']

baseline_golds = results_dict['baseline']['gold_triplets']
cca_golds = results_dict['cca']['gold_triplets']
iscl_golds = results_dict['iscl']['gold_triplets']

# Case Type 1: Baseline FN → CCA TP (CCA recall recovery)
cca_recoveries = []
for i, (bg, cg) in enumerate(zip(baseline_golds, cca_golds)):
    if bg['gold_score'] < 0.2 and cg['gold_score'] > 0.5:
        cca_recoveries.append({
            'idx': i,
            'label': bg['label'],
            'baseline_score': bg['gold_score'],
            'cca_score': cg['gold_score'],
            'iscl_score': iscl_golds[i]['gold_score'],
        })

# Case Type 2: CCA FP → ISCL correct (ISCL precision fix)
baseline_preds = results_dict['baseline']['predictions']
cca_preds = results_dict['cca']['predictions']
iscl_preds = results_dict['iscl']['predictions']

cca_fps_fixed = []
for p_cca, p_iscl in zip(cca_preds, iscl_preds):
    if (not p_cca['is_tp'] and p_cca['prob'] > 0.5 and
        p_cca['gold_label_id'] > 0 and
        (p_iscl['is_tp'] or p_iscl['prob'] < 0.3)):
        cca_fps_fixed.append({
            'pred_label': p_cca['pred_label'],
            'gold_label': p_cca['gold_label'],
            'cca_prob': p_cca['prob'],
            'iscl_prob': p_iscl['prob'],
        })

print(f"=== CCA Recall Recoveries: {len(cca_recoveries)} examples ===")
random.seed(42)
selected_recoveries = sorted(cca_recoveries, key=lambda x: x['cca_score'] - x['baseline_score'], reverse=True)[:10]
for r in selected_recoveries[:5]:
    print(f"  Rel: {r['label']}, Baseline: {r['baseline_score']:.3f}, CCA: {r['cca_score']:.3f}, CCA+ISCL: {r['iscl_score']:.3f}")

print(f"\n=== CCA FP Fixed by ISCL: {len(cca_fps_fixed)} examples ===")
selected_fixes = sorted(cca_fps_fixed, key=lambda x: x['cca_prob'], reverse=True)[:10]
for f in selected_fixes[:5]:
    print(f"  Pred: {f['pred_label']}, Gold: {f['gold_label']}, CCA prob: {f['cca_prob']:.3f}, ISCL prob: {f['iscl_prob']:.3f}")

# Save for LaTeX
case_data = {
    'recoveries': selected_recoveries[:5],
    'fixes': selected_fixes[:5],
}
with open(os.path.join(OUTPUT_DIR, 'case_study.json'), 'w') as f:
    json.dump(case_data, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/case_study.json")
