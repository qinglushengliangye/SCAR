# SCAR: Distributional Stabilization for Zero-Shot Relation Extraction

Code and data for **"Distributional Stabilization for Zero-Shot Relation
Extraction"**, EMNLP 2026 (Main Conference).

---

## What the paper is about

Joint-encoding zero-shot relation extraction (ZSRE) scores every candidate
relation label in a single Transformer forward pass, which makes it both
accurate and fast (156–176 sent/s, versus 1.4–27.8 for a representative
pairwise method). Its robustness, however, has only ever been tested on the
standard protocol, which shows the model just 5–15 candidate relations at a
time.

Under a **controlled full-vocabulary diagnostic** — scoring against all 113
Wiki-ZSL relations at once — two failure modes appear:

| Failure mode | Symptom | Root cause |
| --- | --- | --- |
| **Recall collapse** | 35 of 113 relation types receive *zero* recall (24% of gold instances) | Unnormalized dot-product scores are not scale-invariant: magnitudes learned against ≤25 candidates do not transfer to 113 |
| **Confidence ambiguity** | Gold and negative scores overlap (gap 0.30); the best single threshold yields only 18.2% Macro-F1 | Per-instance BCE never constrains the global geometry of the representation space |

The two are **independent**: fixing the scores alone recovers recall but leaves
best-threshold F1 unchanged at 18.2%. Each needs its own mechanism.

**SCAR** (Stabilized Cascade and Alignment for Robust joint-encoding ZSRE)
keeps the baseline pathway intact and attaches two lightweight branches:

- **CCA — Coarse-to-Fine Cascade Augmentation** (`glirel/modules/dual_encoder_retriever.py`)
  A gradient-isolated dual-encoder stream whose per-sample Z-score normalization
  turns absolute scores into within-candidate-set rankings, restoring
  cross-relation comparability.
- **ISCL — Interactive Supervised Contrastive Learning** (`glirel/modules/supcon.py`)
  A training-only contrastive branch with label-aware cross-attention anchors and
  label-embedding hard negatives, which reshapes the representation geometry so
  gold and negative scores separate. Removed at inference.

Naive integration of either branch is unstable (up to −23.3 pp), so both are
stabilized with gradient isolation, Z-score normalization and curriculum warmup.

### Headline results

| | Baseline | CCA | SCAR |
| --- | --- | --- | --- |
| Wiki-ZSL m=15 Macro-F1 | 75.88 | 82.40 | **83.04** |
| Zero-recall types (of 113) | 35 | 22 | **1** |
| Gold–negative gap | 0.296 | 0.370 | **0.560** |
| Best-threshold Macro-F1 (full vocab) | 18.2 | 18.2 | **42.9** |
| Leakage-free split, test Macro-F1 | 53.5 | 64.0 | **64.5** |

Across 30 runs SCAR improves Macro-F1 by +3.6 to +9.7 pp, cuts cross-run
variance by 49.8%, and adds only ~0.1% parameters and ~2% inference overhead.

---

## Setup

```bash
git clone https://github.com/qinglushengliangye/SCAR.git && cd SCAR
pip install -r requirements.txt
```

Datasets are already preprocessed and included:

| File | Content |
| --- | --- |
| `data/wiki_zsl_all.jsonl` | Wiki-ZSL, 93,399 sentences, 113 relation types |
| `data/few_rel_all.jsonl` | FewRel, 56,000 sentences, 80 relation types |

Configs point at a local DeBERTa-v3-large checkpoint via `model_name` /
`prev_path` / `root_dir`; edit those paths for your environment before running.

---

## Reproducing the paper

### Protocol A — standard ZSRE (main results table)

The protocol of Chen and Li (2021), shared by every compared method: the
held-out relation split serves as both dev and test set, and the best
**Macro-F1** checkpoint over the threshold grid is saved.

```bash
python3 train.py --config configs/config_wiki_zsl_repro.yaml       --log_dir logs/baseline   # Baseline
python3 train.py --config configs/config_wiki_zsl_cascade.yaml     --log_dir logs/cca       # CCA
python3 train.py --config configs/config_wiki_zsl_innovation2.yaml --log_dir logs/scar      # SCAR
```

Swap in `config_few_rel_*.yaml` for FewRel. Each number in the main table is the
mean over five independent random relation splits.

### Protocol B — leakage-free three-way split

Train / dev / test relation sets are mutually disjoint (83 / 15 / 15 relations).
Threshold **and** checkpoint are selected on dev only; the test relations are
scored exactly once at the dev-selected threshold and never influence selection.
Results are written to `<log_dir>/dev_test_results.json`.

```bash
bash scripts/launch_e1_devsplit.sh "0 1 2 3 4"   # GPU ids; runs train_leakage_free.py
python3 scripts/collect_e1_results.py            # paired table, per-split deltas, paired t-test
python3 analysis/e1_oracle_threshold.py          # dev-selected vs. test-oracle threshold
```

### Controlled learning rate

Each configuration run at the other's learning rate, to rule out a
hyperparameter confound.

```bash
bash scripts/launch_e2_controlled_lr.sh "0 1 2 3 4"
```

### Ablations

```bash
bash scripts/run_ablation_cca_no_zscore.sh
bash scripts/run_ablation_cca_no_grad_iso.sh
bash scripts/run_ablation_iscl_no_cross_attn.sh
bash scripts/run_ablation_iscl_no_global_align.sh
bash scripts/run_ablation_iscl_no_supcon_warmup.sh
```

### Multi-GPU scheduler

One job per GPU, auto-retry, skips runs that already produced `_SUCCESS`.

```bash
python3 scripts/run_rebuttal_queue.py 0 1 2 3 4
```

### Analysis and figures (no GPU required)

These read the cached full-vocabulary statistics in `analysis/cached/`, so every
number and figure in the paper can be regenerated without re-running inference.

```bash
python3 analysis/compute_calibration.py    # Table 4: ECE / Brier / NLL
python3 analysis/gen_failure_analysis.py   # Section 4.6: when and why SCAR regresses
python3 analysis/gen_per_rel_table.py      # Table 11: full 113-relation breakdown
python3 analysis/gen_threshold_curve.py    # threshold-vs-F1 sensitivity figure
python3 analysis/gen_per_rel_delta.py      # per-relation Delta-recall figure
```

---

## Repository layout

```
glirel/                          model and modules (CCA, ISCL, evaluator, ...)
train.py                         PROTOCOL A entry point (standard ZSRE)
train_leakage_free.py            PROTOCOL B entry point (three-way disjoint split)
eval.py, infer_and_eval.py       evaluation and inference helpers
configs/                         main experiment + ablation configs
configs/e1_devsplit/             PROTOCOL B configs
configs/e2_controlled_lr/        controlled learning-rate configs
scripts/                         config generators, multi-GPU launchers, collectors
analysis/                        diagnostics that produce the paper's tables/figures
analysis/cached/                 cached full-vocabulary statistics + relation centroids
data/                            preprocessed Wiki-ZSL and FewRel
```

---

## Citation

```bibtex
@inproceedings{liu2026scar,
  title     = {Distributional Stabilization for Zero-Shot Relation Extraction},
  author    = {Liu, Yongxin and Zhang, Huaping and Li, Qiuchi and Li, Lei and
               Gao, Chunxiao and Lv, Haocheng and Yan, Ruohao and Zhang, Baohua},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

## License

Apache-2.0, following the upstream GLiREL framework.
