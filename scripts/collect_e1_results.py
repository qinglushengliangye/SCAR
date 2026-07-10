"""Collect E1 dev-split results into a rebuttal-ready paired table.

For each paired split (same seed => identical train/dev/test relations across
methods) reports the dev-selected TEST Macro-F1, the per-split paired delta
(CCA/SCAR - Baseline), and a paired t-test across splits.
"""
import glob
import json
import os
import statistics as st

BASE = "/root/autodl-tmp/logs-e1"
METHODS = [("repro", "Baseline"), ("cascade", "CCA"), ("innovation2", "SCAR")]
EXPS = ["exp1", "exp2", "exp3"]
PAPER = {"Baseline": 75.88, "CCA": 82.40, "SCAR": 83.04}  # standard (dev==test) protocol


def load(mk, e):
    f = os.path.join(BASE, f"{mk}_wikizsl", e, "dev_test_results.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return j["test_macro_f1"] * 100, j["dev_macro_f1"] * 100


def main():
    tab = {(mn, e): load(mk, e) for mk, mn in METHODS for e in EXPS}

    print("E1: independent dev-split (Wiki-ZSL m=15) | dev-selected TEST Macro-F1")
    print(f"{'split':7s} {'Baseline':>16s} {'CCA':>16s} {'SCAR':>16s}")
    for e in EXPS:
        row = f"{e:7s}"
        for _, mn in METHODS:
            v = tab[(mn, e)]
            row += f" {v[0]:5.2f}(dev{v[1]:4.1f})".rjust(17) if v else f"{'--':>17s}"
        print(row)

    print()
    for _, mn in METHODS:
        vals = [tab[(mn, e)][0] for e in EXPS if tab[(mn, e)]]
        if vals:
            s = st.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"  {mn:9s} mean test F1 = {sum(vals)/len(vals):.2f} +/- {s:.2f}   (paper dev==test: {PAPER[mn]})")

    base = [tab[("Baseline", e)][0] for e in EXPS]
    print()
    for mn in ["CCA", "SCAR"]:
        x = [tab[(mn, e)][0] for e in EXPS]
        d = [x[i] - base[i] for i in range(len(EXPS))]
        md = sum(d) / len(d)
        if len(d) > 1:
            se = st.stdev(d) / (len(d) ** 0.5)
            t = md / se if se else float("inf")
            print(f"  delta({mn}-Baseline) = {[round(z,2) for z in d]}  mean=+{md:.2f}  paired t={t:.2f} (df={len(d)-1})")


if __name__ == "__main__":
    main()
