"""
Deep analysis script for EMNLP paper.
Tasks: t-SNE, confidence distributions, semantic bucketing, error analysis, ablation curves.
"""
import sys
sys.path.insert(0, '/root/GLiREL')

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm

from glirel import GLiREL
from glirel.model import load_config_as_namespace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

OUTPUT_DIR = '/root/GLiREL/paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            return [json.loads(line) for line in f]
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            return json.load(f)


def get_unique_relations(data):
    rels = set()
    for item in data:
        for r in item.get('relations', []):
            rels.add(r['relation_text'])
    return sorted(rels)


def split_data_by_relation_type(data, num_unseen):
    """Simplified split matching eval.py logic."""
    from eval import split_data_by_relation_type as _split
    return _split(data, num_unseen)


def extract_representations_and_predictions(model, config, eval_data, eval_rel_types, device='cuda', batch_size=8):
    """Run forward on eval data and extract rel_rep, scores, and predictions."""
    all_rel_reps = []
    all_labels = []
    all_scores = []
    all_rel_type_names = []
    all_predictions = []
    all_gold_triplets = []

    model.eval()
    data_loader = model.create_dataloader(
        eval_data, batch_size=batch_size,
        relation_types=eval_rel_types, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting representations")):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # --- Hook into compute_score to capture rel_rep ---
            captured = {}
            original_scorer_forward = model.scorer.forward

            def hooked_scorer(candidate_pair_rep, rel_type_rep):
                captured['rel_rep'] = candidate_pair_rep.detach().cpu()
                captured['rel_type_rep'] = rel_type_rep.detach().cpu()
                return original_scorer_forward(candidate_pair_rep, rel_type_rep)

            model.scorer.forward = hooked_scorer
            try:
                scores, num_classes, rel_type_mask, _, _, _ = model.compute_score(batch)
            finally:
                model.scorer.forward = original_scorer_forward

            scores = scores.detach().cpu()
            probabilities = torch.sigmoid(scores)
            rel_labels = batch['rel_label'].detach().cpu()  # [B, num_pairs]
            rel_rep = captured['rel_rep']  # [B, num_pairs, D]
            rel_type_rep_batch = captured['rel_type_rep']  # [B, num_types, D]

            B, P, C = scores.shape

            for b in range(B):
                classes_to_id = batch['classes_to_id'][b]
                id_to_class = {v: k for k, v in classes_to_id.items()}

                for p in range(P):
                    label_id = rel_labels[b, p].item()
                    if label_id <= 0:
                        continue
                    label_name = id_to_class.get(label_id, f'UNK_{label_id}')
                    all_rel_reps.append(rel_rep[b, p].numpy())
                    all_labels.append(label_name)
                    all_scores.append(probabilities[b, p, :C].numpy())
                    all_rel_type_names.append([id_to_class.get(c+1, f'UNK_{c+1}') for c in range(C)])

                    gold_class_idx = label_id - 1
                    if gold_class_idx < C:
                        gold_score = probabilities[b, p, gold_class_idx].item()
                        all_gold_triplets.append({
                            'label': label_name,
                            'gold_score': gold_score,
                            'max_score': probabilities[b, p].max().item(),
                            'pred_class_idx': probabilities[b, p].argmax().item(),
                        })

            # Also collect TP/FP from model predictions
            for b in range(B):
                classes_to_id = batch['classes_to_id'][b]
                id_to_class = {v: k for k, v in classes_to_id.items()}

                for p in range(P):
                    for c in range(min(scores.shape[2], num_classes)):
                        prob = probabilities[b, p, c].item()
                        if prob > 0.01:
                            label_id = rel_labels[b, p].item()
                            pred_class = c + 1
                            is_tp = (label_id == pred_class)
                            all_predictions.append({
                                'prob': prob,
                                'is_tp': is_tp,
                                'gold_label_id': label_id,
                                'pred_label_id': pred_class,
                                'pred_label': id_to_class.get(pred_class, f'UNK'),
                                'gold_label': id_to_class.get(label_id, 'NEG'),
                            })

    return {
        'rel_reps': np.array(all_rel_reps) if all_rel_reps else np.array([]),
        'labels': all_labels,
        'scores': all_scores,
        'predictions': all_predictions,
        'gold_triplets': all_gold_triplets,
    }


# ============================================================
# Task 1: t-SNE Visualization
# ============================================================
def task1_tsne(results_dict, method_names):
    """Generate t-SNE visualization for 3 models side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (method, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        reps = results['rel_reps']
        labels = results['labels']

        if len(reps) == 0:
            ax.set_title(f'{method_names[idx]}\n(no data)')
            continue

        unique_labels = sorted(set(labels))
        label_to_id = {l: i for i, l in enumerate(unique_labels)}
        label_ids = [label_to_id[l] for l in labels]

        tsne = TSNE(n_components=2, perplexity=min(30, len(reps)-1), random_state=42)
        coords = tsne.fit_transform(reps)

        cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(unique_labels))
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                           c=label_ids, cmap=cmap, s=8, alpha=0.6)
        ax.set_title(method_names[idx], fontweight='bold', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tsne_comparison.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    logger.info(f"Task 1: Saved t-SNE figure to {path}")


# ============================================================
# Task 2: Confidence Distribution (TP vs FP)
# ============================================================
def task2_confidence(results_dict, method_names, thresholds=None):
    """Generate TP/FP confidence distribution histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    stats_rows = []

    if thresholds is None:
        thresholds = [0.3, 0.3, 0.3]

    for idx, (method, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        preds = results['predictions']

        thresh = thresholds[idx]
        tp_scores = [p['prob'] for p in preds if p['is_tp'] and p['prob'] >= thresh]
        fp_scores = [p['prob'] for p in preds if not p['is_tp'] and p['prob'] >= thresh and p['gold_label_id'] > 0]

        if not tp_scores and not fp_scores:
            ax.set_title(f'{method_names[idx]}\n(no data above threshold)')
            continue

        bins = np.linspace(0, 1, 50)
        ax.hist(tp_scores, bins=bins, alpha=0.6, color='#2ecc71', label=f'TP (n={len(tp_scores)})', density=True)
        ax.hist(fp_scores, bins=bins, alpha=0.6, color='#e74c3c', label=f'FP (n={len(fp_scores)})', density=True)

        tp_mean = np.mean(tp_scores) if tp_scores else 0
        fp_mean = np.mean(fp_scores) if fp_scores else 0
        gap = tp_mean - fp_mean

        ax.axvline(tp_mean, color='#27ae60', linestyle='--', linewidth=1.5, label=f'TP $\\mu$={tp_mean:.3f}')
        ax.axvline(fp_mean, color='#c0392b', linestyle='--', linewidth=1.5, label=f'FP $\\mu$={fp_mean:.3f}')

        ax.set_title(f'{method_names[idx]} (gap={gap:.3f})', fontweight='bold')
        ax.set_xlabel('Sigmoid Score')
        ax.set_ylabel('Density' if idx == 0 else '')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(0, 1)

        stats_rows.append({
            'method': method_names[idx],
            'tp_count': len(tp_scores),
            'fp_count': len(fp_scores),
            'tp_mean': tp_mean,
            'fp_mean': fp_mean,
            'gap': gap,
        })

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confidence_distribution.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    logger.info(f"Task 2: Saved confidence distribution to {path}")

    with open(os.path.join(OUTPUT_DIR, 'confidence_stats.json'), 'w') as f:
        json.dump(stats_rows, f, indent=2)
    logger.info(f"Task 2: Stats: {stats_rows}")
    return stats_rows


# ============================================================
# Task 3: Semantic Similarity Bucketing
# ============================================================
def task3_semantic_bucketing(results_dict, method_names, eval_rel_types, model, device='cuda'):
    """Bucket unseen relations by semantic similarity and compare per-bucket F1."""
    from sklearn.metrics.pairwise import cosine_similarity

    tokenizer = model.token_rep_layer.bert_layer.tokenizer
    backbone = model.token_rep_layer.bert_layer.model

    logger.info("Computing relation embeddings for semantic similarity...")
    rel_embeddings = {}
    backbone.eval()
    with torch.no_grad():
        for rel in eval_rel_types:
            inputs = tokenizer(rel, return_tensors='pt', truncation=True, max_length=64, padding=True).to(device)
            outputs = backbone(**inputs)
            rel_embeddings[rel] = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    rel_names = sorted(rel_embeddings.keys())
    emb_matrix = np.array([rel_embeddings[r] for r in rel_names])
    sim_matrix = cosine_similarity(emb_matrix)

    max_sim = {}
    for i, r in enumerate(rel_names):
        sims = [(sim_matrix[i, j], rel_names[j]) for j in range(len(rel_names)) if j != i]
        max_sim[r] = max(sims, key=lambda x: x[0])[0]

    median_sim = np.median(list(max_sim.values()))
    easy_rels = {r for r, s in max_sim.items() if s < median_sim}
    hard_rels = {r for r, s in max_sim.items() if s >= median_sim}

    logger.info(f"Easy relations ({len(easy_rels)}): {easy_rels}")
    logger.info(f"Hard relations ({len(hard_rels)}): {hard_rels}")

    bucket_results = []
    for method, results in results_dict.items():
        gold_by_rel = defaultdict(int)
        tp_by_rel = defaultdict(int)

        for gt in results['gold_triplets']:
            gold_by_rel[gt['label']] += 1
            if gt['gold_score'] > 0.5:
                tp_by_rel[gt['label']] += 1

        easy_f1_rels = {}
        hard_f1_rels = {}
        for rel in rel_names:
            g = gold_by_rel.get(rel, 0)
            t = tp_by_rel.get(rel, 0)
            recall = t / g if g > 0 else 0
            if rel in easy_rels:
                easy_f1_rels[rel] = recall
            else:
                hard_f1_rels[rel] = recall

        easy_avg = np.mean(list(easy_f1_rels.values())) if easy_f1_rels else 0
        hard_avg = np.mean(list(hard_f1_rels.values())) if hard_f1_rels else 0

        bucket_results.append({
            'method': method,
            'easy_recall': f'{easy_avg:.4f}',
            'hard_recall': f'{hard_avg:.4f}',
            'delta': f'{hard_avg - easy_avg:.4f}',
        })

    logger.info(f"Task 3: Bucket results: {json.dumps(bucket_results, indent=2)}")
    serializable_max_sim = {k: float(v) for k, v in max_sim.items()}
    with open(os.path.join(OUTPUT_DIR, 'semantic_bucket_results.json'), 'w') as f:
        json.dump({'median_sim': float(median_sim), 'easy_rels': sorted(easy_rels), 'hard_rels': sorted(hard_rels),
                   'results': bucket_results, 'max_sim_per_rel': serializable_max_sim}, f, indent=2)
    return bucket_results


# ============================================================
# Task 4: Error Analysis (FN/FP complementarity)
# ============================================================
def task4_error_analysis(results_dict, method_names):
    """Analyze FN/FP patterns across Baseline/CCA/ISCL."""
    fig, ax = plt.subplots(figsize=(5, 3))

    method_keys = list(results_dict.keys())
    baseline_key = method_keys[0]
    cca_key = method_keys[1]
    iscl_key = method_keys[2]

    baseline_golds = results_dict[baseline_key]['gold_triplets']
    cca_golds = results_dict[cca_key]['gold_triplets']
    iscl_golds = results_dict[iscl_key]['gold_triplets']

    baseline_fn = sum(1 for g in baseline_golds if g['gold_score'] < 0.3)
    baseline_tp = sum(1 for g in baseline_golds if g['gold_score'] >= 0.3)
    cca_fn = sum(1 for g in cca_golds if g['gold_score'] < 0.3)
    cca_tp = sum(1 for g in cca_golds if g['gold_score'] >= 0.3)
    iscl_fn = sum(1 for g in iscl_golds if g['gold_score'] < 0.3)
    iscl_tp = sum(1 for g in iscl_golds if g['gold_score'] >= 0.3)

    baseline_preds = results_dict[baseline_key]['predictions']
    cca_preds = results_dict[cca_key]['predictions']
    iscl_preds = results_dict[iscl_key]['predictions']

    baseline_fp = sum(1 for p in baseline_preds if not p['is_tp'] and p['prob'] >= 0.3 and p['gold_label_id'] > 0)
    cca_fp = sum(1 for p in cca_preds if not p['is_tp'] and p['prob'] >= 0.3 and p['gold_label_id'] > 0)
    iscl_fp = sum(1 for p in iscl_preds if not p['is_tp'] and p['prob'] >= 0.3 and p['gold_label_id'] > 0)

    x = np.arange(3)
    w = 0.25
    methods = method_names

    fn_counts = [baseline_fn, cca_fn, iscl_fn]
    fp_counts = [baseline_fp, cca_fp, iscl_fp]
    tp_counts = [baseline_tp, cca_tp, iscl_tp]

    ax.bar(x - w, fn_counts, w, color='#e74c3c', label='FN (missed)')
    ax.bar(x, fp_counts, w, color='#f39c12', label='FP (false alarm)')
    ax.bar(x + w, tp_counts, w, color='#2ecc71', label='TP (correct)')

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Count')
    ax.set_title('Error Analysis: FN / FP / TP', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'error_analysis.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()

    error_stats = {
        'methods': methods,
        'fn': fn_counts,
        'fp': fp_counts,
        'tp': tp_counts,
        'fn_fixed_by_cca': baseline_fn - cca_fn,
        'fp_change_cca': cca_fp - baseline_fp,
        'fn_fixed_by_iscl_over_cca': cca_fn - iscl_fn,
        'fp_fixed_by_iscl_over_cca': cca_fp - iscl_fp,
    }
    logger.info(f"Task 4: Error analysis: {json.dumps(error_stats, indent=2)}")
    with open(os.path.join(OUTPUT_DIR, 'error_analysis.json'), 'w') as f:
        json.dump(error_stats, f, indent=2)
    return error_stats


# ============================================================
# Task 5: Ablation Training Curves
# ============================================================
def task5_ablation_curves():
    """Extract and plot training curves from ablation logs."""
    import re

    log_dirs = {
        'Full CCA': '/root/autodl-tmp/logs-m=15/logs_innovation1_wikizsl/logs_innovation1_exp3',
        'w/o warmup': '/root/autodl-tmp/logs-m=15/logs_ablation_cca/ablation_no_cascade_warmup',
        'w/o grad iso': '/root/autodl-tmp/logs-m=15/logs_ablation_cca/ablation_no_grad_iso',
        'w/o Z-score': '/root/autodl-tmp/logs-m=15/logs_ablation_cca/ablation_no_zscore',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    colors = {'Full CCA': '#2ecc71', 'w/o warmup': '#3498db', 'w/o grad iso': '#e74c3c', 'w/o Z-score': '#f39c12'}

    for label, log_dir in log_dirs.items():
        log_file = os.path.join(log_dir, 'train.log')
        if not os.path.exists(log_file):
            log_file = os.path.join(log_dir, 'nohup.log')
        if not os.path.exists(log_file):
            logger.warning(f"No log file found for {label} at {log_dir}")
            continue

        with open(log_file, 'r') as f:
            content = f.read()

        steps = []
        macro_f1s = []
        losses = []

        for match in re.finditer(r'step (\d+)', content):
            step = int(match.group(1))
            steps.append(step)

        eval_steps = []
        for match in re.finditer(r'Macro F1: ([\d.]+)%', content):
            macro_f1s.append(float(match.group(1)))

        step_pattern = re.findall(r'step (\d+).*?Macro F1: ([\d.]+)%', content, re.DOTALL)

        if not step_pattern:
            step_numbers = re.findall(r'step[= ]?(\d+)', content)
            macro_vals = re.findall(r'Macro F1: ([\d.]+)%', content)
            if macro_vals:
                n = min(len(macro_vals), 50)
                x_vals = list(range(n))
                ax1.plot(x_vals, [float(v) for v in macro_vals[:n]], label=label, color=colors.get(label, 'gray'), linewidth=1.2)
        else:
            step_vals = [int(s) for s, _ in step_pattern]
            f1_vals = [float(f) for _, f in step_pattern]
            ax1.plot(step_vals, f1_vals, label=label, color=colors.get(label, 'gray'), linewidth=1.2)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Macro F1 (%)')
    ax1.set_title('CCA Ablation: Macro F1 Curves', fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # ISCL ablation
    iscl_log_dirs = {
        'Full ISCL': '/root/autodl-tmp/logs-m=15/logs_innovation2_wikizsl/logs_innovation2_split2',
        'w/o cross-attn': '/root/autodl-tmp/logs-m=15/logs_ablation_iscl/ablation_no_cross_attn',
        'w/o global align': '/root/autodl-tmp/logs-m=15/logs_ablation_iscl/ablation_no_global_align',
        'w/o SC warmup': '/root/autodl-tmp/logs-m=15/logs_ablation_iscl/ablation_no_supcon_warmup',
    }
    iscl_colors = {'Full ISCL': '#2ecc71', 'w/o cross-attn': '#3498db', 'w/o global align': '#e74c3c', 'w/o SC warmup': '#f39c12'}

    for label, log_dir in iscl_log_dirs.items():
        log_file = os.path.join(log_dir, 'train.log')
        if not os.path.exists(log_file):
            log_file = os.path.join(log_dir, 'nohup.log')
        if not os.path.exists(log_file):
            continue

        with open(log_file, 'r') as f:
            content = f.read()

        macro_vals = re.findall(r'Macro F1: ([\d.]+)%', content)
        if macro_vals:
            n = min(len(macro_vals), 50)
            ax2.plot(range(n), [float(v) for v in macro_vals[:n]], label=label, color=iscl_colors.get(label, 'gray'), linewidth=1.2)

    ax2.set_xlabel('Evaluation Index')
    ax2.set_ylabel('Macro F1 (%)')
    ax2.set_title('ISCL Ablation: Macro F1 Curves', fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'ablation_curves.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    logger.info(f"Task 5: Saved ablation curves to {path}")


# ============================================================
# Main
# ============================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    checkpoints = {
        'baseline': '/root/autodl-tmp/logs-m=15/logs_repro_wikizsl/logs_repro_exp3/model_3900',
        'cca': '/root/autodl-tmp/logs-m=15/logs_innovation1_wikizsl/logs_innovation1_exp3/model_6600',
        'iscl': '/root/autodl-tmp/logs-m=15/logs_innovation2_wikizsl/logs_innovation2_split2/model_6000',
    }
    method_names = ['Baseline', 'CCA', 'CCA+ISCL']

    data_path = '/root/GLiREL/data/wiki_zsl_all.jsonl'
    logger.info(f"Loading data from {data_path}...")
    all_data = load_data(data_path)
    logger.info(f"Loaded {len(all_data)} samples")

    results_dict = {}
    eval_rel_types_saved = None
    last_model = None

    for method, ckpt_dir in checkpoints.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {method}: {ckpt_dir}")
        logger.info(f"{'='*60}")

        model, config = load_model_and_config(ckpt_dir, device)
        config.num_unseen_rel_types = 15

        # Use the model's config seed for splitting
        np.random.seed(42)
        import random as rnd
        rnd.seed(42)
        torch.manual_seed(42)

        train_data, eval_data = split_data_by_relation_type(all_data, config.num_unseen_rel_types)
        eval_rel_types = get_unique_relations(eval_data)
        eval_rel_types_saved = eval_rel_types
        last_model = model

        logger.info(f"Eval data: {len(eval_data)} samples, {len(eval_rel_types)} relation types")

        results = extract_representations_and_predictions(
            model, config, eval_data, eval_rel_types, device, batch_size=4
        )
        results_dict[method] = results
        logger.info(f"Extracted {len(results['rel_reps'])} representations, {len(results['predictions'])} predictions")

        del model
        torch.cuda.empty_cache()

    # Save intermediate results
    import pickle
    cache_path = os.path.join(OUTPUT_DIR, 'analysis_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({'results_dict': results_dict, 'eval_rel_types': eval_rel_types_saved}, f)
    logger.info(f"Saved intermediate results to {cache_path}")

    # --- Task 1: t-SNE ---
    logger.info("\n" + "="*60 + "\nTask 1: t-SNE Visualization\n" + "="*60)
    task1_tsne(results_dict, method_names)

    # --- Task 2: Confidence Distribution ---
    logger.info("\n" + "="*60 + "\nTask 2: Confidence Distribution\n" + "="*60)
    conf_stats = task2_confidence(results_dict, method_names)

    # --- Task 3: Semantic Bucketing ---
    logger.info("\n" + "="*60 + "\nTask 3: Semantic Similarity Bucketing\n" + "="*60)
    if last_model is not None and eval_rel_types_saved is not None:
        last_model_reload, _ = load_model_and_config(checkpoints['baseline'], device)
        bucket_results = task3_semantic_bucketing(results_dict, method_names, eval_rel_types_saved, last_model_reload, device)
        del last_model_reload
        torch.cuda.empty_cache()

    # --- Task 4: Error Analysis ---
    logger.info("\n" + "="*60 + "\nTask 4: Error Analysis\n" + "="*60)
    error_stats = task4_error_analysis(results_dict, method_names)

    # --- Task 5: Ablation Curves ---
    logger.info("\n" + "="*60 + "\nTask 5: Ablation Training Curves\n" + "="*60)
    task5_ablation_curves()

    logger.info("\n" + "="*60 + "\nAll tasks complete!" + "\n" + "="*60)
    logger.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
