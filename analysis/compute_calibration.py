"""E3: Calibration diagnostics (ECE / Brier / NLL) for the rebuttal.

Reads the cached full-vocabulary score samples in paper/figures/stats_*.json and
computes calibration metrics treating each (pair, label) candidate as a binary
decision: is this candidate the gold relation? The predicted probability is the
model score sigma(S_final) in [0, 1].

Because stats_*.json stores a random SAMPLE of gold ("tp") and non-gold ("fp")
candidate scores (10k each) plus the true population counts (tp_count / fp_count),
we reweight each sampled score by (true_count / n_sample) so the reported metrics
reflect the real gold/negative prevalence rather than the 1:1 sample.

Usage:
    python analysis/compute_calibration.py
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHED = os.path.join(_HERE, "cached")
FIG_DIR = _CACHED if os.path.isdir(_CACHED) else os.path.join(_HERE, "..", "paper", "figures")
METHODS = {
    "Baseline": "stats_baseline.json",
    "CCA": "stats_cca.json",
    "SCAR": "stats_cca_iscl.json",
}
EPS = 1e-7
N_BINS = 15


def load(method_file):
    d = json.load(open(os.path.join(FIG_DIR, method_file)))
    tp = np.asarray(d["tp_scores_sample"], dtype=np.float64)  # gold candidates, y=1
    fp = np.asarray(d["fp_scores_sample"], dtype=np.float64)  # non-gold candidates, y=0
    tp_w = d["tp_count"] / len(tp)  # up-weight each gold sample to population
    fp_w = d["fp_count"] / len(fp)  # up-weight each negative sample to population
    return tp, fp, tp_w, fp_w


def weighted_metrics(tp, fp, tp_w, fp_w):
    """Return ECE, Brier, NLL under population-reweighted prevalence."""
    p = np.concatenate([tp, fp])
    y = np.concatenate([np.ones_like(tp), np.zeros_like(fp)])
    w = np.concatenate([np.full_like(tp, tp_w), np.full_like(fp, fp_w)])
    w = w / w.sum()

    pc = np.clip(p, EPS, 1 - EPS)
    brier = np.sum(w * (p - y) ** 2)
    nll = np.sum(w * -(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

    # Expected Calibration Error (equal-width bins on predicted prob).
    ece = 0.0
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        wb = w[m].sum()
        if wb <= 0:
            continue
        conf = np.sum(w[m] * p[m]) / wb
        acc = np.sum(w[m] * y[m]) / wb  # weighted fraction of gold in bin
        ece += wb * abs(acc - conf)
    return ece, brier, nll


def main():
    print(f"{'Method':10s} {'ECE':>8s} {'Brier':>8s} {'NLL':>8s} {'Gold+mu':>8s} {'Neg mu':>8s}")
    for name, f in METHODS.items():
        tp, fp, tp_w, fp_w = load(f)
        ece, brier, nll = weighted_metrics(tp, fp, tp_w, fp_w)
        print(f"{name:10s} {ece:8.4f} {brier:8.4f} {nll:8.4f} {tp.mean():8.4f} {fp.mean():8.4f}")
    print(
        "\nNote: population-reweighted (gold prevalence ~7%). Scores are sigmoid(S_final);"
        "\nlower ECE/Brier/NLL = better-calibrated probabilities."
    )


if __name__ == "__main__":
    main()
