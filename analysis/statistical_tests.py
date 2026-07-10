"""
Statistical significance analysis for the EMNLP paper.

Splits are drawn independently for each method (see Limitations), so the
correct reference framework is unpaired: bootstrap CI on the mean difference
between two independent samples and an unpaired permutation test. The
Wilcoxon signed-rank statistic is also reported for compatibility with prior
versions of this script; it is **not** used as the primary p-value because
its paired assumption does not hold for our data.

Writes
  paper/figures/significance.json
with bootstrap CIs (unpaired), Welch t-tests, and unpaired permutation
p-values per setting and on the aggregate of 30 (setting, split) runs.
"""
import json
import os

import numpy as np
from scipy import stats

# Per-seed Macro F1 values (5 seeds each)
DATA = {
    'wiki_m5': {
        'baseline': [88.44, 95.81, 77.10, 81.28, 85.36],
        'cca': [97.46, 96.84, 90.10, 93.42, 94.24],
        'cca_iscl': [96.84, 94.45, 93.56, 92.59, 92.33],
    },
    'wiki_m10': {
        'baseline': [77.76, 77.91, 80.36, 82.35, 82.30],
        'cca': [93.46, 90.05, 88.56, 89.04, 83.28],
        'cca_iscl': [91.03, 90.27, 90.40, 86.19, 91.44],
    },
    'wiki_m15': {
        'baseline': [79.84, 81.53, 76.43, 69.24, 72.37],
        'cca': [75.03, 82.86, 82.80, 84.92, 86.40],
        'cca_iscl': [84.06, 88.65, 84.17, 80.45, 77.88],
    },
    'few_m5': {
        'baseline': [83.36, 81.24, 97.77, 90.88, 81.19],
        'cca': [92.14, 94.27, 90.12, 96.94, 98.94],
        'cca_iscl': [98.16, 94.42, 94.41, 98.57, 94.55],
    },
    'few_m10': {
        'baseline': [86.08, 89.37, 83.53, 84.90, 86.63],
        'cca': [93.46, 91.09, 86.03, 90.75, 86.26],
        'cca_iscl': [87.75, 88.67, 90.27, 93.84, 88.14],
    },
    'few_m15': {
        'baseline': [89.30, 80.11, 75.44, 83.87, 73.79],
        'cca': [90.28, 89.58, 90.08, 83.60, 83.43],
        'cca_iscl': [83.59, 88.96, 85.27, 89.47, 85.57],
    },
}


def welch_ttest(x, y):
    """Two-sample Welch (unequal-variance) t-test, returning (t, p)."""
    return stats.ttest_ind(x, y, equal_var=False)


def paired_ttest(x, y):
    """Paired t-test (kept for reference only; do not use as primary)."""
    return stats.ttest_rel(x, y)


def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test (paired; kept for reference only)."""
    d = np.array(x) - np.array(y)
    if np.all(d == 0):
        return None, 1.0
    try:
        return stats.wilcoxon(d)
    except ValueError:
        return None, 1.0


def permutation_test(x, y, n_resamples=20000, seed=42):
    """Unpaired two-sample permutation test on the difference of means.

    Returns (observed_mean_diff, p_value).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_resamples):
        rng.shuffle(pooled)
        diff = pooled[:n_x].mean() - pooled[n_x:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    return float(observed), (count + 1) / (n_resamples + 1)


def bootstrap_ci_unpaired(x, y, n_bootstrap=10000, ci=0.95, seed=42):
    """Unpaired bootstrap CI on the difference of means.

    Each bootstrap iteration resamples x and y independently (with
    replacement) and records the resulting mean difference.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        xs = rng.choice(x, len(x), replace=True)
        ys = rng.choice(y, len(y), replace=True)
        diffs[i] = xs.mean() - ys.mean()
    alpha = (1 - ci) / 2
    return np.percentile(diffs, [alpha * 100, (1 - alpha) * 100])


def bootstrap_ci(x, y, n_bootstrap=10000, ci=0.95):
    """Paired bootstrap (kept for reference only; do not use as primary).

    Treats x and y as paired samples and resamples the indices jointly.
    """
    rng = np.random.RandomState(42)
    diffs = np.array(x) - np.array(y)
    n = len(diffs)
    boot_means = np.array([rng.choice(diffs, n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])


def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS (unpaired)")
    print("=" * 80)

    out = {'per_setting': {}, 'aggregate': {}, 'variance': {}}

    all_baseline = []
    all_cca = []
    all_cca_iscl = []

    for setting in DATA:
        all_baseline.extend(DATA[setting]['baseline'])
        all_cca.extend(DATA[setting]['cca'])
        all_cca_iscl.extend(DATA[setting]['cca_iscl'])

    print(
        f"\n{'Setting':<10} {'CCA vs Base perm p':<22} "
        f"{'ISCL vs Base perm p':<22} {'CCA-Base CI':<22} {'ISCL-Base CI':<22}"
    )
    print("-" * 100)

    for setting in DATA:
        b = DATA[setting]['baseline']
        c = DATA[setting]['cca']
        ci = DATA[setting]['cca_iscl']

        obs_cb, p_cb = permutation_test(c, b)
        obs_ib, p_ib = permutation_test(ci, b)
        ci_cb = bootstrap_ci_unpaired(c, b)
        ci_ib = bootstrap_ci_unpaired(ci, b)
        _, p_welch_cb = welch_ttest(c, b)
        _, p_welch_ib = welch_ttest(ci, b)

        out['per_setting'][setting] = {
            'mean_delta_cca_vs_base': obs_cb,
            'mean_delta_iscl_vs_base': obs_ib,
            'perm_p_cca_vs_base': p_cb,
            'perm_p_iscl_vs_base': p_ib,
            'welch_p_cca_vs_base': float(p_welch_cb),
            'welch_p_iscl_vs_base': float(p_welch_ib),
            'bootstrap_ci_cca_vs_base': [float(ci_cb[0]), float(ci_cb[1])],
            'bootstrap_ci_iscl_vs_base': [float(ci_ib[0]), float(ci_ib[1])],
        }

        print(
            f"{setting:<10} p={p_cb:<19.4f} p={p_ib:<19.4f} "
            f"[{ci_cb[0]:+.2f},{ci_cb[1]:+.2f}]   [{ci_ib[0]:+.2f},{ci_ib[1]:+.2f}]"
        )

    print("\n--- Aggregate unpaired permutation test (30 runs per arm) ---")
    obs_agg_c, p_agg_c = permutation_test(all_cca, all_baseline)
    obs_agg_i, p_agg_i = permutation_test(all_cca_iscl, all_baseline)
    obs_agg_ic, p_agg_ic = permutation_test(all_cca_iscl, all_cca)
    ci_agg_c = bootstrap_ci_unpaired(all_cca, all_baseline)
    ci_agg_i = bootstrap_ci_unpaired(all_cca_iscl, all_baseline)
    ci_agg_ic = bootstrap_ci_unpaired(all_cca_iscl, all_cca)
    _, welch_c = welch_ttest(all_cca, all_baseline)
    _, welch_i = welch_ttest(all_cca_iscl, all_baseline)
    _, welch_ic = welch_ttest(all_cca_iscl, all_cca)

    print(f"CCA      vs Baseline: mean Delta = {obs_agg_c:+.2f} pp, perm p = {p_agg_c:.4f}, "
          f"Welch p = {welch_c:.4f}, bootstrap 95% CI [{ci_agg_c[0]:+.2f}, {ci_agg_c[1]:+.2f}] pp")
    print(f"CCA+ISCL vs Baseline: mean Delta = {obs_agg_i:+.2f} pp, perm p = {p_agg_i:.4f}, "
          f"Welch p = {welch_i:.4f}, bootstrap 95% CI [{ci_agg_i[0]:+.2f}, {ci_agg_i[1]:+.2f}] pp")
    print(f"CCA+ISCL vs CCA     : mean Delta = {obs_agg_ic:+.2f} pp, perm p = {p_agg_ic:.4f}, "
          f"Welch p = {welch_ic:.4f}, bootstrap 95% CI [{ci_agg_ic[0]:+.2f}, {ci_agg_ic[1]:+.2f}] pp")

    out['aggregate'] = {
        'cca_vs_base':      {'mean_delta': obs_agg_c,  'perm_p': p_agg_c,  'welch_p': float(welch_c),
                             'ci_95': [float(ci_agg_c[0]),  float(ci_agg_c[1])]},
        'iscl_vs_base':     {'mean_delta': obs_agg_i,  'perm_p': p_agg_i,  'welch_p': float(welch_i),
                             'ci_95': [float(ci_agg_i[0]),  float(ci_agg_i[1])]},
        'iscl_vs_cca':      {'mean_delta': obs_agg_ic, 'perm_p': p_agg_ic, 'welch_p': float(welch_ic),
                             'ci_95': [float(ci_agg_ic[0]), float(ci_agg_ic[1])]},
    }

    print("\n--- Variance reduction (average std across 6 settings) ---")
    stds = {'baseline': [], 'cca': [], 'cca_iscl': []}
    for setting in DATA:
        for method in stds:
            stds[method].append(np.std(DATA[setting][method], ddof=1))
    for method in stds:
        print(f"  {method:<10}: mean std = {np.mean(stds[method]):.2f}")
    reduction_cca = (1 - np.mean(stds['cca']) / np.mean(stds['baseline'])) * 100
    reduction_iscl = (1 - np.mean(stds['cca_iscl']) / np.mean(stds['baseline'])) * 100
    print(f"  CCA reduces mean std by {reduction_cca:.1f}%")
    print(f"  CCA+ISCL reduces mean std by {reduction_iscl:.1f}%")

    out['variance'] = {
        'mean_std_baseline':  float(np.mean(stds['baseline'])),
        'mean_std_cca':       float(np.mean(stds['cca'])),
        'mean_std_cca_iscl':  float(np.mean(stds['cca_iscl'])),
        'cca_reduction_pct':  float(reduction_cca),
        'iscl_reduction_pct': float(reduction_iscl),
    }

    out_path = os.path.join('/root/GLiREL/paper/figures', 'significance.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
