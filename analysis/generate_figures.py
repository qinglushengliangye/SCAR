"""
Generate all paper figures in Nature style.
Reads lightweight stats from analysis_stats.json (produced by extract_stats.py).
"""
import json
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

PALETTE = {
    "baseline": "#484878",
    "cca": "#3775BA",
    "cca_iscl": "#B64342",
}

METHOD_NAMES = {
    "baseline": "Baseline",
    "cca": "CCA",
    "cca_iscl": "CCA + ISCL",
}

OUTPUT_DIR = '/root/GLiREL/paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_stats():
    combined_path = os.path.join(OUTPUT_DIR, 'analysis_stats.json')
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            return json.load(f)
    stats = {}
    for method in ['baseline', 'cca', 'cca_iscl']:
        path = os.path.join(OUTPUT_DIR, f'stats_{method}.json')
        if os.path.exists(path):
            with open(path) as f:
                stats[method] = json.load(f)
    return stats


def save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.svg"), bbox_inches="tight")
    print(f"  Saved: {name}.pdf / .svg")
    plt.close(fig)


def fig_main_results():
    """Grouped bar chart for all 6 dataset-m combinations."""
    data = {
        'Wiki-ZSL\nm=5': {'baseline': 85.60, 'cca': 94.41, 'cca_iscl': 93.95},
        'Wiki-ZSL\nm=10': {'baseline': 74.47, 'cca': 88.88, 'cca_iscl': 89.87},
        'Wiki-ZSL\nm=15': {'baseline': 75.88, 'cca': 82.40, 'cca_iscl': 83.04},
        'FewRel\nm=5': {'baseline': 86.89, 'cca': 94.48, 'cca_iscl': 96.02},
        'FewRel\nm=10': {'baseline': 86.10, 'cca': 89.52, 'cca_iscl': 89.73},
        'FewRel\nm=15': {'baseline': 80.50, 'cca': 87.39, 'cca_iscl': 86.57},
    }
    stds = {
        'Wiki-ZSL\nm=5': {'baseline': 7.1, 'cca': 2.9, 'cca_iscl': 1.8},
        'Wiki-ZSL\nm=10': {'baseline': 11.6, 'cca': 3.7, 'cca_iscl': 2.1},
        'Wiki-ZSL\nm=15': {'baseline': 5.1, 'cca': 4.4, 'cca_iscl': 4.1},
        'FewRel\nm=5': {'baseline': 7.3, 'cca': 3.6, 'cca_iscl': 2.1},
        'FewRel\nm=10': {'baseline': 2.2, 'cca': 3.3, 'cca_iscl': 2.5},
        'FewRel\nm=15': {'baseline': 6.3, 'cca': 3.6, 'cca_iscl': 2.5},
    }

    settings = list(data.keys())
    methods = ['baseline', 'cca', 'cca_iscl']
    x = np.arange(len(settings))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for i, method in enumerate(methods):
        vals = [data[s][method] for s in settings]
        errs = [stds[s][method] for s in settings]
        ax.bar(x + (i - 1) * width, vals, width, label=METHOD_NAMES[method],
               color=PALETTE[method], edgecolor='white', linewidth=0.3,
               yerr=errs, capsize=2, error_kw={'linewidth': 0.6})

    ax.set_ylabel('Macro F1 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=6.5)
    ax.set_ylim(60, 100)
    ax.legend(loc='upper right', ncol=3, fontsize=6)
    ax.axhline(y=80, color='#CCCCCC', linewidth=0.4, linestyle='--', zorder=0)
    ax.axhline(y=90, color='#CCCCCC', linewidth=0.4, linestyle='--', zorder=0)
    save_fig(fig, 'fig_main_results')


def fig_tsne():
    """3-panel t-SNE: 8 hand-picked relations with largest silhouette improvement
    (BL->ISCL) for maximum visual contrast. High-contrast color palette.
    Other relations as small grey dots. All-label silhouette in subtitle."""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    # 8 relations selected for largest monotonic silhouette gain (BL→CCA→ISCL)
    HIGHLIGHT_RELS = [
        'position held',
        'award received',
        'member of political party',
        'sport',
        'production company',
        'heritage designation',
        'manufacturer',
        'occupation',
    ]
    # Maximally distinct, high-saturation palette (8 colors)
    COLORS = [
        '#E41A1C',  # red
        '#377EB8',  # blue
        '#4DAF4A',  # green
        '#FF7F00',  # orange
        '#984EA3',  # purple
        '#A65628',  # brown
        '#F781BF',  # pink
        '#FFFF33',  # yellow
    ]

    rel_to_color = {r: COLORS[i] for i, r in enumerate(HIGHLIGHT_RELS)}
    highlight_set = set(HIGHLIGHT_RELS)

    methods = ['baseline', 'cca', 'cca_iscl']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        tsne_path = os.path.join(OUTPUT_DIR, f'tsne_data_{method}.pkl')
        if not os.path.exists(tsne_path):
            ax.set_title(METHOD_NAMES[method], fontsize=8, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        with open(tsne_path, 'rb') as f:
            td = pickle.load(f)
        reps = np.array(td['reps'])
        labels = td['labels']

        all_unique = sorted(set(labels))
        n_classes = len(all_unique)
        all_label_to_id = {l: i for i, l in enumerate(all_unique)}
        all_label_ids = np.array([all_label_to_id[l] for l in labels])
        sil_all = silhouette_score(reps, all_label_ids, metric='cosine',
                                   sample_size=min(2000, len(reps)), random_state=42)

        tsne = TSNE(n_components=2, perplexity=min(30, len(reps) - 1), random_state=42)
        coords = tsne.fit_transform(reps)

        # Grey background for non-highlighted relations
        other_mask = np.array([l not in highlight_set for l in labels])
        if other_mask.any():
            ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                       c='#DCDCDC', s=3, alpha=0.25, linewidths=0,
                       rasterized=True, zorder=1)

        # Highlighted relations with high-contrast colors
        for rel in HIGHLIGHT_RELS:
            mask = np.array([l == rel for l in labels])
            if mask.any():
                short = rel[:22] + ('..' if len(rel) > 22 else '')
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=rel_to_color[rel], s=16, alpha=0.85,
                           linewidths=0.15, edgecolors='black',
                           rasterized=True, zorder=2,
                           label=short if idx == 0 else None)

        ax.set_title(METHOD_NAMES[method], fontsize=8, fontweight='bold', pad=4)
        ax.text(0.5, -0.02, f"silhouette = {sil_all:.2f}  ({n_classes} types)",
                transform=ax.transAxes, ha='center', va='top', fontsize=5.5,
                color='#444444')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color('#333333')
        ax.set_aspect('equal', adjustable='datalim')

    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc='lower center', ncol=4,
                   fontsize=5.5, markerscale=1.0, handletextpad=0.3,
                   columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.10, 1, 1], w_pad=0.8)
    save_fig(fig, 'fig_tsne')


def fig_confidence(stats):
    """TP vs FP confidence distribution per method."""
    methods = ['baseline', 'cca', 'cca_iscl']
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    panel_labels = ['a', 'b', 'c']

    for idx, method in enumerate(methods):
        ax = axes[idx]
        s = stats[method]
        tp_scores = s['tp_scores_sample']
        fp_scores = s['fp_scores_sample']
        x_range = np.linspace(0, 1, 200)

        if tp_scores:
            kde_tp = gaussian_kde(tp_scores, bw_method=0.05)
            ax.fill_between(x_range, kde_tp(x_range), alpha=0.3, color='#2E9E44')
            ax.plot(x_range, kde_tp(x_range), color='#2E9E44', linewidth=1.0, label='TP')
            ax.axvline(s['tp_mean'], color='#2E9E44', linestyle='--', linewidth=0.7)
        if fp_scores:
            kde_fp = gaussian_kde(fp_scores, bw_method=0.05)
            ax.fill_between(x_range, kde_fp(x_range), alpha=0.3, color='#E53935')
            ax.plot(x_range, kde_fp(x_range), color='#E53935', linewidth=1.0, label='FP')
            ax.axvline(s['fp_mean'], color='#E53935', linestyle='--', linewidth=0.7)

        gap = s['tp_mean'] - s['fp_mean']
        ax.set_xlabel('Score')
        if idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(f"{METHOD_NAMES[method]} (gap={gap:.2f})", fontsize=7, fontweight='bold')
        ax.legend(fontsize=5.5)
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    plt.tight_layout(w_pad=0.8)
    save_fig(fig, 'fig_confidence')


def fig_analysis_composite(stats):
    """Multi-panel: (a) confidence KDE overlay, (b) threshold-F1, (c) score margin boxplot."""
    methods = ['baseline', 'cca', 'cca_iscl']
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    panel_labels = ['a', 'b', 'c']

    # Panel (a): TP confidence overlay
    ax = axes[0]
    for method in methods:
        tp = stats[method]['tp_scores_sample']
        if tp:
            x_range = np.linspace(0, 1, 200)
            kde = gaussian_kde(tp, bw_method=0.05)
            ax.plot(x_range, kde(x_range), color=PALETTE[method], linewidth=1.2,
                    label=f"{METHOD_NAMES[method]}")
    ax.set_xlabel('TP confidence')
    ax.set_ylabel('Density')
    ax.set_title('TP score distribution', fontsize=7, fontweight='bold')
    ax.legend(fontsize=5, loc='upper left')
    ax.text(-0.15, 1.05, panel_labels[0], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    # Panel (b): Threshold-F1
    ax = axes[1]
    for method in methods:
        tf = stats[method]['thresh_f1']
        thresholds = sorted([float(k) for k in tf.keys()])
        f1s = [tf[str(round(t, 2))] for t in thresholds]
        ax.plot(thresholds, f1s, color=PALETTE[method], linewidth=1.2,
                label=METHOD_NAMES[method])
    ax.axvline(x=0.5, color='#999999', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 (%)')
    ax.set_title('Threshold robustness', fontsize=7, fontweight='bold')
    ax.legend(fontsize=5.5)
    ax.text(-0.15, 1.05, panel_labels[1], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    # Panel (c): Score margin boxplot
    ax = axes[2]
    margins_data = []
    colors = []
    for method in methods:
        margins_data.append(stats[method]['correct_margins'])
        colors.append(PALETTE[method])

    bp = ax.boxplot(margins_data, positions=range(len(methods)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 0.8})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=6)
    ax.set_ylabel('Margin above 0.5')
    ax.set_title('Decision confidence', fontsize=7, fontweight='bold')
    ax.text(-0.15, 1.05, panel_labels[2], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    plt.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig_analysis_composite')


def fig_error_analysis(stats):
    """Stacked bar showing TP/FP/FN decomposition."""
    methods = ['baseline', 'cca', 'cca_iscl']
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    tp_counts = [stats[m]['error_tp'] for m in methods]
    fp_counts = [stats[m]['error_fp'] for m in methods]
    fn_counts = [stats[m]['error_fn'] for m in methods]

    x = np.arange(len(methods))
    width = 0.6
    ax.bar(x, tp_counts, width, label='TP', color='#2E9E44', alpha=0.85)
    ax.bar(x, fp_counts, width, bottom=tp_counts, label='FP', color='#E53935', alpha=0.85)
    ax.bar(x, fn_counts, width, bottom=[t + f for t, f in zip(tp_counts, fp_counts)],
           label='FN', color='#767676', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=7)
    ax.set_ylabel('Count')
    ax.set_title('Error decomposition (threshold=0.5)', fontsize=7, fontweight='bold')
    ax.legend(fontsize=6, loc='upper right')
    save_fig(fig, 'fig_error_analysis')


def fig_agreement(stats):
    """Prediction agreement between methods."""
    methods = ['baseline', 'cca', 'cca_iscl']
    correct_sets = {}
    for method in methods:
        correct_sets[method] = set(stats[method]['correct_indices'])

    n_total = stats['baseline']['n_gold']
    base_unique = correct_sets['baseline'] - correct_sets['cca'] - correct_sets['cca_iscl']
    cca_unique = correct_sets['cca'] - correct_sets['baseline'] - correct_sets['cca_iscl']
    full_unique = correct_sets['cca_iscl'] - correct_sets['baseline'] - correct_sets['cca']
    all_three = correct_sets['baseline'] & correct_sets['cca'] & correct_sets['cca_iscl']
    cca_and_full = (correct_sets['cca'] & correct_sets['cca_iscl']) - correct_sets['baseline']
    neither = n_total - len(correct_sets['baseline'] | correct_sets['cca'] | correct_sets['cca_iscl'])

    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    categories = [
        'All three\ncorrect',
        'CCA & CCA+ISCL\n(not Baseline)',
        'Only CCA\ncorrect',
        'Only CCA+ISCL\ncorrect',
        'Only Baseline\ncorrect',
        'None correct',
    ]
    values = [len(all_three), len(cca_and_full), len(cca_unique), len(full_unique), len(base_unique), neither]
    colors_bar = ['#2E9E44', PALETTE['cca_iscl'], PALETTE['cca'], '#D47A7A', PALETTE['baseline'], '#999999']

    bars = ax.barh(range(len(categories)), values, color=colors_bar, height=0.6)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=6)
    ax.set_xlabel('Gold triplets')
    ax.set_title('Prediction agreement', fontsize=7, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=5.5)
    save_fig(fig, 'fig_agreement')


def fig_per_relation(stats):
    """Per-relation recall comparison: top-20 most common relations."""
    methods = ['baseline', 'cca', 'cca_iscl']
    base_prf = stats['baseline'].get('per_rel_f1', {})
    if not base_prf:
        print("  Skipping per-relation figure (no data)")
        return

    top_rels = sorted(base_prf.keys(), key=lambda r: base_prf[r]['total'], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    y = np.arange(len(top_rels))
    height = 0.25

    for i, method in enumerate(methods):
        prf = stats[method].get('per_rel_f1', {})
        recalls = [prf.get(r, {}).get('recall', 0) for r in top_rels]
        ax.barh(y + (i - 1) * height, recalls, height, label=METHOD_NAMES[method],
                color=PALETTE[method], alpha=0.85, edgecolor='white', linewidth=0.3)

    short_names = [r[:25] + ('...' if len(r) > 25 else '') for r in top_rels]
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=5)
    ax.set_xlabel('Recall (%)')
    ax.set_title('Per-relation recall (top-20 by frequency)', fontsize=7, fontweight='bold')
    ax.legend(fontsize=5.5, loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, 'fig_per_relation')


def fig_prf_breakdown(stats):
    """Precision / Recall / F1 at threshold=0.5 for each method."""
    methods = ['baseline', 'cca', 'cca_iscl']
    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    x = np.arange(3)
    width = 0.22
    metrics = ['p', 'r', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1']
    metric_colors = ['#3775BA', '#2E9E44', '#B64342']

    for mi, metric in enumerate(metrics):
        vals = []
        for method in methods:
            prf = stats[method].get('thresh_prf', {}).get('0.5', {})
            vals.append(prf.get(metric, 0))
        ax.bar(x + (mi - 1) * width, vals, width, label=metric_labels[mi],
               color=metric_colors[mi], alpha=0.85, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=7)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 100)
    ax.set_title('P / R / F1 at threshold 0.5', fontsize=7, fontweight='bold')
    ax.legend(fontsize=5.5)
    plt.tight_layout()
    save_fig(fig, 'fig_prf_breakdown')


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Nature-style figures")
    print("=" * 60)

    stats = load_stats()

    print("\n[1/8] Main results bar chart")
    fig_main_results()
    print("[2/8] Confidence distributions")
    fig_confidence(stats)
    print("[3/8] Analysis composite")
    fig_analysis_composite(stats)
    print("[4/8] Error analysis")
    fig_error_analysis(stats)
    print("[5/8] Prediction agreement")
    fig_agreement(stats)
    print("[6/8] Per-relation recall")
    fig_per_relation(stats)
    print("[7/8] P/R/F1 breakdown")
    fig_prf_breakdown(stats)
    print("[8/8] t-SNE (may take a few minutes)...")
    fig_tsne()

    print("\nAll figures generated!")
