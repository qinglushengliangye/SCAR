"""Characterize when and why SCAR regresses (appendix / Section 5.4 analysis).

For every relation we locate its nearest neighbour in the learned representation
space (the relation whose instances have the closest mean cosine direction), then
test whether the regressing relations are the ones that dominated that pair under
the Baseline. Also emits the sibling-recovery table used in the paper.
"""
import json
import os
import pickle
import statistics as st

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHED = os.path.join(_HERE, "cached")
FIG_DIR = _CACHED if os.path.isdir(_CACHED) else os.path.join(_HERE, "..", "paper", "figures")


CACHE = os.path.join(os.path.dirname(__file__), "cached", "rel_centroids_scar.npz")


def relation_centroids(method):
    """L2-normalized mean direction of each relation's instance representations.

    Uses the shipped 113x768 cache when present; falls back to the full
    representation dump produced by extract_tsne_reps.py. Both give identical
    nearest neighbours, since the mean similarity of a relation's instances to
    every centroid is proportional to its own centroid's similarity.
    """
    if method == "cca_iscl" and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return list(z["names"]), z["centroids"]
    d = pickle.load(open(os.path.join(FIG_DIR, "tsne_data_%s.pkl" % method), "rb"))
    X = np.asarray(d["reps"], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.asarray(d["labels"])
    names = sorted(set(d["labels"]))
    C = np.stack([X[labels == u].mean(0) for u in names])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return names, C


def nearest_neighbours(method):
    names, C = relation_centroids(method)
    sims = C @ C.T
    np.fill_diagonal(sims, -np.inf)
    return {u: names[int(np.argmax(sims[k]))] for k, u in enumerate(names)}


def main():
    base = json.load(open(os.path.join(FIG_DIR, "stats_baseline.json")))["per_rel_f1"]
    scar = json.load(open(os.path.join(FIG_DIR, "stats_cca_iscl.json")))["per_rel_f1"]
    delta = {r: scar[r]["recall"] - base[r]["recall"] for r in base}
    nn = nearest_neighbours("cca_iscl")

    reg = [r for r in delta if delta[r] < 0]
    imp = [r for r in delta if delta[r] > 0]

    def dominant(group):
        return [r for r in group if base[nn[r]]["recall"] < base[r]["recall"]]

    print("=== Is the regressing relation the dominant member of its nearest-neighbour pair? ===")
    for group, tag in ((reg, "regressing"), (imp, "improving")):
        dom = dominant(group)
        gaps = [base[r]["recall"] - base[nn[r]]["recall"] for r in group]
        print("  %-11s %3d/%3d (%.0f%%), median Baseline recall gap %+.1f pp"
              % (tag, len(dom), len(group), 100 * len(dom) / len(group), st.median(gaps)))

    print("\n=== Regressing relations and their nearest neighbour ===")
    print("  %-30s %7s %7s | %-30s %7s %7s" % ("relation", "base", "delta", "neighbour", "base", "delta"))
    lost = gained = 0.0
    for r in sorted(reg, key=lambda r: delta[r]):
        s = nn[r]
        lost += -delta[r] / 100 * base[r]["total"]
        gained += delta[s] / 100 * base[s]["total"]
        print("  %-30s %7.1f %+7.1f | %-30s %7.1f %+7.1f"
              % (r[:30], base[r]["recall"], delta[r], s[:30], base[s]["recall"], delta[s]))
    print("\n  instances given up by the %d regressing relations : %.0f" % (len(reg), lost))
    print("  instances gained by their nearest neighbours       : %.0f" % gained)


if __name__ == "__main__":
    main()
