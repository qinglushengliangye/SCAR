"""Regenerate fig_per_rel_delta with a clean layout (no y-axis label overlap).

Layout: x = relation rank (sorted by Delta recall, descending),
        y = Delta Recall (pp, CCA+ISCL - Baseline),
       color = green for positive, red for negative,
   annotations = top-5 gainers and bottom-5 losers (rest unlabeled).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = '/root/GLiREL/paper/figures'

GAIN = '#2E9E44'
LOSS = '#B64342'
ZERO = '#888888'


def main():
    b = json.load(open(os.path.join(FIG_DIR, 'stats_baseline.json')))['per_rel_f1']
    i = json.load(open(os.path.join(FIG_DIR, 'stats_cca_iscl.json')))['per_rel_f1']

    rels = list(b.keys())
    deltas = [(r, i[r]['recall'] - b[r]['recall']) for r in rels]
    deltas.sort(key=lambda kv: kv[1], reverse=True)

    n = len(deltas)
    x = np.arange(n)
    y = np.array([d for _, d in deltas])
    colors = [GAIN if d > 0 else (LOSS if d < 0 else ZERO) for _, d in deltas]

    n_pos = sum(1 for _, d in deltas if d > 0)
    n_neg = sum(1 for _, d in deltas if d < 0)
    n_zero = n - n_pos - n_neg

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, y, color=colors, width=1.0, edgecolor='none')
    ax.axhline(0, color='black', linewidth=0.6)

    # Select top-4 gainers and bottom-4 losers.
    top_gain = [(k, deltas[k][0], deltas[k][1]) for k in range(min(4, n))]
    bot_loss = [(n - 1 - k, deltas[n - 1 - k][0], deltas[n - 1 - k][1]) for k in range(min(4, n))]
    # Keep loser labels in left-to-right bar order so connectors do not cross.
    bot_loss.sort(key=lambda t: t[0])

    def clean_label(name):
        if ' / ' in name:
            name = name.split(' / ')[0]
        if name == 'position played on team / speciality':
            name = 'position played on team'
        return name if len(name) <= 34 else name[:31] + '...'

    # Gainer annotations: place labels in a stack to the right of the bars
    base_x = 22
    base_y_top = 102
    line_h = 12
    for j, (idx, name, d) in enumerate(top_gain):
        ax.annotate(
                clean_label(name),
            xy=(idx, d),
            xytext=(base_x, base_y_top - j * line_h),
            fontsize=6.4,
            color='black',
            arrowprops=dict(arrowstyle='-', lw=0.4, color='gray'),
        )

    # Loser annotations: place labels in a stack to the left of the bars
    base_x_loss = n - 35
    base_y_loss = -32
    for j, (idx, name, d) in enumerate(bot_loss):
        ax.annotate(
            clean_label(name),
            xy=(idx, d),
            xytext=(base_x_loss, base_y_loss - j * line_h),
            fontsize=6.4,
            color='black',
            ha='right',
            arrowprops=dict(arrowstyle='-', lw=0.4, color='gray'),
        )

    ax.set_xlabel('Relation rank (sorted by $\\Delta$Recall, descending)', fontsize=8)
    ax.set_ylabel('$\\Delta$ Recall (pp)', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.set_xlim(-1.5, n + 1)
    ax.set_ylim(min(y) - 20, max(y) + 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    summary = (
        f'$\\mathbf{{{n_pos}}}$ improve  '
        f'($\\Delta$Recall $>$ 0)\u2003·\u2003'
        f'$\\mathbf{{{n_neg}}}$ regress'
    )
    ax.text(
        0.99,
        0.96,
        summary,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=7.5,
    )

    plt.tight_layout()
    pdf_path = os.path.join(FIG_DIR, 'fig_per_rel_delta.pdf')
    svg_path = os.path.join(FIG_DIR, 'fig_per_rel_delta.svg')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.04)
    print(f'wrote {pdf_path}  ({n_pos} improve, {n_neg} regress, {n_zero} unchanged)')


if __name__ == '__main__':
    main()
