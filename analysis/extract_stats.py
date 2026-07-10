"""
Extract lightweight statistics from full-dataset inference caches.
Produces a small JSON file that the figure generator reads.
"""
import json
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, '/root/GLiREL')

OUTPUT_DIR = '/root/GLiREL/paper/figures'

def process_method(method):
    path = os.path.join(OUTPUT_DIR, f'full_analysis_{method}.pkl')
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    preds = data['predictions']
    golds = data['gold_triplets']

    tp_scores = [p['prob'] for p in preds if p['is_tp'] and p['prob'] > 0.01]
    fp_scores = [p['prob'] for p in preds if not p['is_tp'] and p['gold_label'] != 'NEG' and p['prob'] > 0.01]

    # Threshold-F1 curve
    thresholds = [round(t, 2) for t in np.arange(0.01, 0.96, 0.02).tolist()]
    thresh_f1 = {}
    for thresh in thresholds:
        tp = sum(1 for p in preds if p['prob'] >= thresh and p['is_tp'])
        fp = sum(1 for p in preds if p['prob'] >= thresh and not p['is_tp'])
        fn = sum(1 for g in golds if g['gold_score'] < thresh)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        thresh_f1[str(thresh)] = round(f1 * 100, 2)

    # Error decomposition at threshold=0.5
    tp_05 = sum(1 for p in preds if p['prob'] >= 0.5 and p['is_tp'])
    fp_05 = sum(1 for p in preds if p['prob'] >= 0.5 and not p['is_tp'])
    fn_05 = sum(1 for g in golds if g['gold_score'] < 0.5)

    # Score margins for correct predictions
    correct_margins = [g['gold_score'] - 0.5 for g in golds if g['is_correct'] and g['gold_score'] > 0.5]

    # Prediction correctness for agreement analysis
    correct_indices = [i for i, g in enumerate(golds) if g['is_correct'] and g['gold_score'] >= 0.5]

    # t-SNE data (subsample representations)
    reps = data.get('rel_reps', np.array([]))
    labels = data.get('labels', [])
    tsne_reps = None
    tsne_labels = None
    if len(reps) > 0:
        max_tsne = 2000
        if len(reps) > max_tsne:
            idx = np.random.RandomState(42).choice(len(reps), max_tsne, replace=False)
            reps_sub = reps[idx]
            labels_sub = [labels[i] for i in idx]
        else:
            reps_sub = reps
            labels_sub = labels
        tsne_reps = reps_sub.tolist()
        tsne_labels = labels_sub

    stats = {
        'tp_scores_sample': sorted(np.random.RandomState(42).choice(tp_scores, min(5000, len(tp_scores)), replace=False).tolist()) if tp_scores else [],
        'fp_scores_sample': sorted(np.random.RandomState(42).choice(fp_scores, min(5000, len(fp_scores)), replace=False).tolist()) if fp_scores else [],
        'tp_mean': float(np.mean(tp_scores)) if tp_scores else 0,
        'fp_mean': float(np.mean(fp_scores)) if fp_scores else 0,
        'tp_count': len(tp_scores),
        'fp_count': len(fp_scores),
        'thresh_f1': thresh_f1,
        'error_tp': tp_05,
        'error_fp': fp_05,
        'error_fn': fn_05,
        'correct_margins': sorted(np.random.RandomState(42).choice(correct_margins, min(2000, len(correct_margins)), replace=False).tolist()) if correct_margins else [],
        'margin_median': float(np.median(correct_margins)) if correct_margins else 0,
        'margin_q25': float(np.percentile(correct_margins, 25)) if correct_margins else 0,
        'margin_q75': float(np.percentile(correct_margins, 75)) if correct_margins else 0,
        'n_gold': len(golds),
        'correct_indices': correct_indices,
    }

    if tsne_reps is not None:
        tsne_path = os.path.join(OUTPUT_DIR, f'tsne_data_{method}.pkl')
        with open(tsne_path, 'wb') as f:
            pickle.dump({'reps': tsne_reps, 'labels': tsne_labels}, f)
        print(f"  Saved t-SNE data to {tsne_path}")

    del data
    return stats


def main():
    all_stats = {}
    for method in ['baseline', 'cca', 'cca_iscl']:
        print(f"\n--- Processing {method} ---")
        all_stats[method] = process_method(method)

    out_path = os.path.join(OUTPUT_DIR, 'analysis_stats.json')
    with open(out_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved stats to {out_path}")

    # Print summary
    for method in all_stats:
        s = all_stats[method]
        gap = s['tp_mean'] - s['fp_mean']
        print(f"\n{method}:")
        print(f"  TP mean={s['tp_mean']:.3f}, FP mean={s['fp_mean']:.3f}, gap={gap:.3f}")
        print(f"  Error: TP={s['error_tp']}, FP={s['error_fp']}, FN={s['error_fn']}")
        print(f"  Margin: median={s['margin_median']:.3f}, IQR=[{s['margin_q25']:.3f}, {s['margin_q75']:.3f}]")


if __name__ == '__main__':
    main()
