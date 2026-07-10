"""
Full-dataset analysis for EMNLP paper (v3).
Runs inference on the ENTIRE Wiki-ZSL corpus (93K samples).
Uses the validated TP/FP definition (per pair×class level, consistent with 20K analysis).
Accumulates statistics in-place to avoid storing millions of prediction dicts.
Stratified t-SNE sampling ensures all 113 relation types are represented.
Extracts qualitative case studies for the paper.
"""
import sys
import os

if '--gpu' in sys.argv:
    idx = sys.argv.index('--gpu')
    if idx + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[idx + 1]

sys.path.insert(0, '/root/GLiREL')

import argparse
import json
import logging
import pickle
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from glirel import GLiREL
from glirel.model import load_config_as_namespace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = '/root/GLiREL/paper/figures'
os.makedirs(CACHE_DIR, exist_ok=True)


def load_model_and_config(ckpt_dir, device='cuda'):
    config_path = os.path.join(ckpt_dir, 'glirel_config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = argparse.Namespace(**config_dict)
    model = GLiREL.from_pretrained(ckpt_dir, map_location=device)
    model.config = config
    model = model.to(device)
    model.eval()
    return model, config


def load_data(data_path):
    with open(data_path, 'r') as f:
        return [json.loads(line) for line in f]


def get_unique_relations(data):
    rels = set()
    for item in data:
        for r in item.get('relations', []):
            rels.add(r['relation_text'])
    return sorted(rels)


def extract_full_dataset(model, config, data, device='cuda', batch_size=192):
    """Run forward pass, accumulating all statistics in-place."""
    all_rel_types = get_unique_relations(data)
    logger.info(f"Total unique relations: {len(all_rel_types)}")

    # --- Accumulators (no giant lists) ---
    # 1) Gold triplet records (lightweight, ~134K entries)
    all_gold_triplets = []
    # 2) Representations for t-SNE (cap 50K, stratified-sample later)
    all_rel_reps = []
    all_rep_labels = []
    MAX_REPS = 80000
    # 3) TP/FP score accumulators (old-style: per pair×class, prob > 0.01)
    #    Only store sampled scores for KDE plots, plus running sums for means
    tp_score_sum = 0.0
    tp_score_count = 0
    fp_score_sum = 0.0
    fp_score_count = 0
    tp_scores_reservoir = []
    fp_scores_reservoir = []
    RESERVOIR_SIZE = 10000
    # 4) Threshold evaluation accumulators
    thresholds = np.array([round(t, 2) for t in np.arange(0.01, 0.96, 0.01)])
    n_thresh = len(thresholds)
    global_tp_thresh = np.zeros(n_thresh, dtype=np.int64)
    global_fp_thresh = np.zeros(n_thresh, dtype=np.int64)
    global_fn_thresh = np.zeros(n_thresh, dtype=np.int64)
    # 5) Case study candidates (top confident correct/incorrect predictions)
    case_studies = []

    sample_count = 0
    rng = np.random.RandomState(42)

    model.eval()
    data_loader = model.create_dataloader(
        data, batch_size=batch_size,
        relation_types=all_rel_types, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Full-dataset inference")):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            captured = {}
            original_scorer_forward = model.scorer.forward

            def hooked_scorer(candidate_pair_rep, rel_type_rep):
                captured['rel_rep'] = candidate_pair_rep.detach().cpu()
                return original_scorer_forward(candidate_pair_rep, rel_type_rep)

            model.scorer.forward = hooked_scorer
            try:
                scores, num_classes, rel_type_mask, _, _, _ = model.compute_score(batch)
            finally:
                model.scorer.forward = original_scorer_forward

            scores_cpu = scores.detach().cpu()
            probabilities = torch.sigmoid(scores_cpu)
            rel_labels = batch['rel_label'].detach().cpu()
            rel_rep = captured['rel_rep']

            B, P, C = scores_cpu.shape
            nc = min(C, num_classes)
            sample_count += B

            for b in range(B):
                classes_to_id = batch['classes_to_id'][b]
                id_to_class = {v: k for k, v in classes_to_id.items()}

                # --- Gold triplet processing (positive pairs only) ---
                for p in range(P):
                    label_id = rel_labels[b, p].item()
                    if label_id <= 0:
                        continue

                    label_name = id_to_class.get(label_id, f'UNK_{label_id}')

                    # Representations (capped)
                    if len(all_rel_reps) < MAX_REPS:
                        all_rel_reps.append(rel_rep[b, p].numpy())
                        all_rep_labels.append(label_name)

                    gold_class_idx = label_id - 1
                    if gold_class_idx < C:
                        gold_score = probabilities[b, p, gold_class_idx].item()
                        max_score = probabilities[b, p, :nc].max().item()
                        pred_idx = probabilities[b, p, :nc].argmax().item()
                        pred_label = id_to_class.get(pred_idx + 1, 'UNK')
                        is_correct = (pred_idx == gold_class_idx)
                        all_gold_triplets.append({
                            'label': label_name,
                            'gold_score': gold_score,
                            'max_score': max_score,
                            'pred_label': pred_label,
                            'is_correct': is_correct,
                        })

                        # FN counting per threshold
                        gold_above = (gold_score >= thresholds)
                        global_fn_thresh += (~gold_above).astype(np.int64)

                        # Case study candidates
                        if len(case_studies) < 200:
                            tokens = batch.get('tokens', [[]])[b] if 'tokens' in batch else []
                            case_studies.append({
                                'text': ' '.join(tokens) if tokens else '',
                                'gold': label_name,
                                'pred': pred_label,
                                'gold_score': round(gold_score, 4),
                                'max_score': round(max_score, 4),
                                'correct': is_correct,
                            })

                # --- Per pair×class TP/FP accumulation (old-style definition) ---
                probs_np = probabilities[b, :, :nc].numpy()  # [P, nc]
                labels_np = rel_labels[b].numpy()  # [P]

                for p in range(P):
                    label_id = labels_np[p]
                    for c_idx in range(nc):
                        prob = probs_np[p, c_idx]
                        if prob <= 0.005:
                            continue
                        pred_class = c_idx + 1
                        is_tp = (label_id == pred_class)

                        # TP/FP at each threshold
                        if is_tp:
                            global_tp_thresh += (prob >= thresholds).astype(np.int64)
                        else:
                            global_fp_thresh += (prob >= thresholds).astype(np.int64)

                        # Score distribution (old-style: all pair×class with prob > 0.01)
                        if prob > 0.01:
                            if is_tp:
                                tp_score_sum += prob
                                tp_score_count += 1
                                if len(tp_scores_reservoir) < RESERVOIR_SIZE:
                                    tp_scores_reservoir.append(float(prob))
                                else:
                                    j = rng.randint(0, tp_score_count)
                                    if j < RESERVOIR_SIZE:
                                        tp_scores_reservoir[j] = float(prob)
                            elif label_id > 0:
                                fp_score_sum += prob
                                fp_score_count += 1
                                if len(fp_scores_reservoir) < RESERVOIR_SIZE:
                                    fp_scores_reservoir.append(float(prob))
                                else:
                                    j = rng.randint(0, fp_score_count)
                                    if j < RESERVOIR_SIZE:
                                        fp_scores_reservoir[j] = float(prob)

    tp_mean = tp_score_sum / tp_score_count if tp_score_count > 0 else 0
    fp_mean = fp_score_sum / fp_score_count if fp_score_count > 0 else 0

    logger.info(f"Processed {sample_count} samples, {len(all_gold_triplets)} gold triplets, "
                f"{len(all_rel_reps)} reps, TP scores: {tp_score_count:,}, FP scores: {fp_score_count:,}")

    # Threshold-F1 curve
    thresh_f1 = {}
    thresh_prf = {}
    for i in range(n_thresh):
        t = round(thresholds[i], 2)
        tp = int(global_tp_thresh[i])
        fp = int(global_fp_thresh[i])
        fn = int(global_fn_thresh[i])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        t_str = str(t)
        thresh_f1[t_str] = round(f1 * 100, 2)
        thresh_prf[t_str] = {'p': round(prec * 100, 2), 'r': round(rec * 100, 2),
                              'f1': round(f1 * 100, 2), 'tp': tp, 'fp': fp, 'fn': fn}

    best_thresh = max(thresh_f1, key=thresh_f1.get)

    # Per-relation recall at threshold=0.5
    rel_counts = Counter(g['label'] for g in all_gold_triplets)
    rel_tp = Counter(g['label'] for g in all_gold_triplets if g['is_correct'] and g['gold_score'] >= 0.5)
    per_rel_f1 = {}
    for rel in all_rel_types:
        total = rel_counts.get(rel, 0)
        tp = rel_tp.get(rel, 0)
        per_rel_f1[rel] = {'recall': round(tp / total * 100, 2) if total > 0 else 0, 'total': total}

    correct_margins = [g['gold_score'] - 0.5 for g in all_gold_triplets
                       if g['is_correct'] and g['gold_score'] > 0.5]
    correct_indices = [i for i, g in enumerate(all_gold_triplets) if g['is_correct'] and g['gold_score'] >= 0.5]
    prf_05 = thresh_prf.get('0.5', {'tp': 0, 'fp': 0, 'fn': 0})

    return {
        'rel_reps': np.array(all_rel_reps) if all_rel_reps else np.array([]),
        'rep_labels': all_rep_labels,
        'gold_triplets': all_gold_triplets,
        'all_rel_types': all_rel_types,
        'tp_mean': tp_mean,
        'fp_mean': fp_mean,
        'tp_count': tp_score_count,
        'fp_count': fp_score_count,
        'tp_scores_reservoir': sorted(tp_scores_reservoir),
        'fp_scores_reservoir': sorted(fp_scores_reservoir),
        'thresh_f1': thresh_f1,
        'thresh_prf': thresh_prf,
        'best_threshold': best_thresh,
        'per_rel_f1': per_rel_f1,
        'correct_margins': correct_margins,
        'correct_indices': correct_indices,
        'error_tp': prf_05.get('tp', 0),
        'error_fp': prf_05.get('fp', 0),
        'error_fn': prf_05.get('fn', 0),
        'n_samples': sample_count,
        'n_gold': len(all_gold_triplets),
        'case_studies': case_studies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['baseline', 'cca', 'cca_iscl'])
    parser.add_argument('--ckpt-dir', type=str, required=True)
    parser.add_argument('--data', type=str, default='/root/GLiREL/data/wiki_zsl_all.jsonl')
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = args.device
    logger.info(f"Loading model from {args.ckpt_dir}")
    model, config = load_model_and_config(args.ckpt_dir, device)

    logger.info(f"Loading data from {args.data}")
    data = load_data(args.data)
    logger.info(f"Loaded {len(data)} samples")

    if args.max_samples > 0 and len(data) > args.max_samples:
        import random
        random.seed(42)
        random.shuffle(data)
        data = data[:args.max_samples]
        logger.info(f"Subsampled to {len(data)} samples")

    t0 = time.time()
    results = extract_full_dataset(model, config, data, device, args.batch_size)
    elapsed = time.time() - t0
    logger.info(f"Inference completed in {elapsed/60:.1f} minutes")

    rng = np.random.RandomState(42)

    # --- Save stats JSON ---
    stats = {
        'tp_scores_sample': results['tp_scores_reservoir'],
        'fp_scores_sample': results['fp_scores_reservoir'],
        'tp_mean': results['tp_mean'],
        'fp_mean': results['fp_mean'],
        'tp_count': results['tp_count'],
        'fp_count': results['fp_count'],
        'thresh_f1': results['thresh_f1'],
        'thresh_prf': results['thresh_prf'],
        'per_rel_f1': results['per_rel_f1'],
        'best_threshold': results['best_threshold'],
        'error_tp': results['error_tp'],
        'error_fp': results['error_fp'],
        'error_fn': results['error_fn'],
        'n_samples': results['n_samples'],
        'n_gold': results['n_gold'],
        'correct_margins': sorted(rng.choice(
            results['correct_margins'], min(2000, len(results['correct_margins'])), replace=False
        ).tolist()) if results['correct_margins'] else [],
        'margin_median': float(np.median(results['correct_margins'])) if results['correct_margins'] else 0,
        'margin_q25': float(np.percentile(results['correct_margins'], 25)) if results['correct_margins'] else 0,
        'margin_q75': float(np.percentile(results['correct_margins'], 75)) if results['correct_margins'] else 0,
        'correct_indices': results['correct_indices'],
    }
    stats_path = os.path.join(CACHE_DIR, f'stats_{args.method}.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    logger.info(f"Saved stats to {stats_path}")

    # --- Save t-SNE data (stratified sampling, all types covered) ---
    reps = results['rel_reps']
    labels_list = results['rep_labels']
    TARGET = 6000
    if len(reps) > TARGET:
        by_class = defaultdict(list)
        for i, lab in enumerate(labels_list):
            by_class[lab].append(i)
        n_classes = len(by_class)
        per_class = max(3, TARGET // n_classes)
        selected = []
        for lab, indices in by_class.items():
            k = min(per_class, len(indices))
            selected.extend(rng.choice(indices, k, replace=False).tolist())
        remaining = TARGET - len(selected)
        if remaining > 0:
            pool = list(set(range(len(reps))) - set(selected))
            selected.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())
        selected = sorted(set(selected))[:TARGET]
        reps = reps[selected]
        labels_list = [labels_list[i] for i in selected]
    tsne_path = os.path.join(CACHE_DIR, f'tsne_data_{args.method}.pkl')
    with open(tsne_path, 'wb') as f:
        pickle.dump({'reps': reps.tolist(), 'labels': labels_list}, f)
    logger.info(f"Saved t-SNE ({len(reps)} reps, {len(set(labels_list))} types) to {tsne_path}")

    # --- Save case studies ---
    cases_path = os.path.join(CACHE_DIR, f'cases_{args.method}.json')
    with open(cases_path, 'w') as f:
        json.dump(results['case_studies'], f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results['case_studies'])} case studies to {cases_path}")

    # --- Summary ---
    gap = results['tp_mean'] - results['fp_mean']
    logger.info(f"=== {args.method} Summary ===")
    logger.info(f"  Samples: {results['n_samples']}, Gold: {results['n_gold']}")
    logger.info(f"  TP mean: {results['tp_mean']:.4f} ({results['tp_count']:,}), "
                f"FP mean: {results['fp_mean']:.4f} ({results['fp_count']:,}), Gap: {gap:.4f}")
    logger.info(f"  Best thresh: {results['best_threshold']} "
                f"(F1={results['thresh_f1'][results['best_threshold']]}%)")
    logger.info(f"  @0.5: TP={results['error_tp']}, FP={results['error_fp']}, FN={results['error_fn']}")
    logger.info(f"  Margin: med={stats['margin_median']:.3f} "
                f"IQR=[{stats['margin_q25']:.3f}, {stats['margin_q75']:.3f}]")


if __name__ == '__main__':
    main()
