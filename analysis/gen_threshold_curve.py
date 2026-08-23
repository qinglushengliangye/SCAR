"""Regenerate the threshold-vs-F1 sensitivity figure.

Reads the cached full-vocabulary threshold sweep stored in stats_*.json, so the
figure can be rebuilt without re-running inference.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHED = os.path.join(_HERE, "cached")
FIG_DIR = _CACHED if os.path.isdir(_CACHED) else os.path.join(_HERE, "..", "paper", "figures")
OUT_DIR = os.path.join(_HERE, "..", "paper", "figures")

METHODS = [
    ("Baseline", "stats_baseline.json", "#e74c3c"),
    ("CCA", "stats_cca.json", "#3498db"),
    ("SCAR", "stats_cca_iscl.json", "#2ecc71"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for label, fname, color in METHODS:
        curve = json.load(open(os.path.join(FIG_DIR, fname)))["thresh_f1"]
        pts = sorted((float(t), f1) for t, f1 in curve.items())
        xs = [t for t, _ in pts]
        ys = [f1 for _, f1 in pts]
        ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
        best_t, best_f1 = max(pts, key=lambda tf: tf[1])
        ax.plot([best_t], [best_f1], marker="o", ms=3.5, color=color)
        print("%-9s best F1 %5.2f at t=%.2f" % (label, best_f1, best_t))

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("F1 (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)

    out = os.path.join(OUT_DIR, "threshold_sensitivity.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
