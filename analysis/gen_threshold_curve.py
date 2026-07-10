"""Generate threshold vs F1 sensitivity curve from cached predictions."""
import sys
sys.path.insert(0, '/root/GLiREL')

import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = '/root/GLiREL/paper/figures'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

cache_path = f'{OUTPUT_DIR}/analysis_cache.pkl'
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

results_dict = cache['results_dict']
method_names = ['Baseline', 'CCA', 'CCA+ISCL']
colors = ['#e74c3c', '#3498db', '#2ecc71']

thresholds = np.arange(0.01, 0.95, 0.02)

fig, ax = plt.subplots(figsize=(4.5, 3))

for idx, (method, results) in enumerate(results_dict.items()):
    preds = results['predictions']
    golds = results['gold_triplets']
    total_gold = len(golds)

    f1_values = []
    for thresh in thresholds:
        tp = sum(1 for p in preds if p['is_tp'] and p['prob'] >= thresh)
        fp = sum(1 for p in preds if not p['is_tp'] and p['prob'] >= thresh and p['gold_label_id'] > 0)
        fn = total_gold - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_values.append(f1 * 100)

    ax.plot(thresholds, f1_values, color=colors[idx], linewidth=1.5, label=method_names[idx])

    best_idx = np.argmax(f1_values)
    ax.scatter(thresholds[best_idx], f1_values[best_idx], color=colors[idx], s=30, zorder=5)

ax.set_xlabel('Decision Threshold')
ax.set_ylabel('F1 (%)')
ax.set_title('Threshold Sensitivity', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'threshold_sensitivity.pdf')
plt.savefig(path)
plt.savefig(path.replace('.pdf', '.png'))
plt.close()
print(f"Saved to {path}")
