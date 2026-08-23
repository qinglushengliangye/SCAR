"""Emit the full 113-relation recall breakdown as a LaTeX table for the appendix.

Reads the cached full-vocabulary per-relation statistics and writes
paper/tables/per_rel_table.tex. Relations are sorted by Delta Recall (descending)
and laid out in two side-by-side blocks so the table fits a single page.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHED = os.path.join(_HERE, "cached")
FIG_DIR = _CACHED if os.path.isdir(_CACHED) else os.path.join(_HERE, "..", "paper", "figures")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "tables", "per_rel_table.tex")
MAXLEN = 32


def esc(name):
    if len(name) > MAXLEN:
        name = name[: MAXLEN - 3] + "..."
    return name.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def main():
    base = json.load(open(os.path.join(FIG_DIR, "stats_baseline.json")))["per_rel_f1"]
    scar = json.load(open(os.path.join(FIG_DIR, "stats_cca_iscl.json")))["per_rel_f1"]

    rows = sorted(base, key=lambda r: -(scar[r]["recall"] - base[r]["recall"]))
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    right += [None] * (len(left) - len(right))

    def cells(r):
        if r is None:
            return " & & & & "
        d = scar[r]["recall"] - base[r]["recall"]
        mark = r"\textcolor{red}{%+.1f}" % d if d < 0 else "%+.1f" % d
        return "%s & %d & %.1f & %.1f & %s" % (
            esc(r), base[r]["total"], base[r]["recall"], scar[r]["recall"], mark)

    body = "\n".join(r"%s & %s \\" % (cells(a), cells(b)) for a, b in zip(left, right))

    head = (r"Relation & Gold & Base & SCAR & $\Delta$ & "
            r"Relation & Gold & Base & SCAR & $\Delta$ \\")

    tex = r"""\begin{table*}[p]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Full per-relation recall (\%%) on the complete Wiki-ZSL corpus (all $113$ types, $134$K gold instances,
threshold $0.5$), sorted by $\Delta$Recall $=$ SCAR $-$ Baseline (descending). ``Gold'' is the number of gold
instances of that relation. Regressions are shown in \textcolor{red}{red}. Recall improves on $\mathbf{101}$ of
$113$ types; the $12$ regressing types carry $8.8\%%$ of gold instances and cost $-2.8$~pp against a
$+35.0$~pp gold-weighted net gain (\Cref{sec:whenfail}).}
\label{tab:per_rel_full}
\begin{tabular}{l r r r r @{\hspace{10pt}} l r r r r}
\toprule
%s
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (head, body)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(tex)
    print("wrote %s (%d relations, %d table rows)" % (OUT, len(rows), len(left)))


if __name__ == "__main__":
    main()
