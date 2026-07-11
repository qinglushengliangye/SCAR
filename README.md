# SCAR: Stabilized Cascade and Alignment for Robust Joint-Encoding ZSRE

Anonymized code and data for the submission *"Distributional Recalibration for
Zero-Shot Relation Extraction"*. SCAR augments a joint-encoding ZSRE baseline
(GLiREL-style, DeBERTa-v3-large) with two lightweight, stabilized auxiliary
branches:

- **CCA (Coarse-to-Fine Cascade Augmentation)** — a gradient-isolated dual-encoder
  scoring stream with per-sample Z-score normalization, added to the fine-grained
  score. See `glirel/modules/dual_encoder_retriever.py`.
- **ISCL (Interactive Supervised Contrastive Learning)** — a training-only,
  label-aware supervised contrastive branch (removed at inference). See
  `glirel/modules/supcon.py`.

## Repository layout

```
glirel/                         model + modules (CCA, ISCL, evaluator, ...)
train_original.py               PROTOCOL A entry point (standard ZSRE, paper main table)
train.py                        PROTOCOL A/B entry point (adds leakage-free dev-split)
configs/                        main experiment configs (Baseline / CCA / SCAR, m=15)
configs/e1_devsplit/            PROTOCOL B: independent dev-split configs
configs/e2_controlled_lr/       controlled learning-rate configs
scripts/                        config generators + multi-GPU launchers + collectors
analysis/compute_calibration.py ECE / Brier / NLL diagnostics
data/wiki_zsl_all.jsonl         preprocessed Wiki-ZSL (113 relation types)
data/few_rel_all.jsonl          preprocessed FewRel (80 relation types)
```

## Two evaluation protocols

**Protocol A — standard ZSRE (paper main table).** The relation-disjoint held-out
set is used as *both* dev and test: the decision threshold and the best checkpoint
are selected on the same held-out relation set that is then reported. This is the
field-standard protocol (ZS-BERT / TMC-BERT / GLiREL). Reproduced by
`train_original.py` (unmodified), or by `train.py` when `num_dev_rel_types` is
absent from the config.

**Protocol B — leakage-free dev-split (rebuttal experiment E1).** The held-out
relations are further partitioned into a *disjoint* dev and test set (train / dev /
test relation sets are mutually disjoint). Threshold and checkpoint are selected
*only* on dev; the test relations are scored once, at the dev-selected threshold,
and never influence selection. Enabled by setting `num_dev_rel_types` (see
`configs/e1_devsplit/`); the per-run dev-selected test score is written to
`<log_dir>/dev_test_results.json`.

Protocol B is a stricter, self-contained comparison used to verify that the SCAR /
CCA gains are not an artifact of selecting the threshold on the reported test
relations; its absolute numbers are lower than Protocol A (independent selection +
fewer training relations) and are not directly comparable to the main table.

`train.py` additionally re-binds the selection metric to the config's
`threshold_search_metric` (macro_f1) and makes relation splits deterministic per
seed; `train_original.py` is kept verbatim for exact reproduction of the main table.

## Setup

```bash
pip install -r requirements.txt
# DeBERTa-v3-large backbone + a GLiREL-style pretrained checkpoint are loaded via
# the paths in each config (model_name / prev_path); set them to your local paths.
```

## Data

`data/wiki_zsl_all.jsonl` and `data/few_rel_all.jsonl` are the preprocessed,
span-level datasets used in all experiments (each line is one sentence with entity
spans and relation annotations). Preprocessing scripts: `data/process_wiki_zsl.py`,
`data/process_few_rel.py`.

## Reproducing the experiments

Relation splits are seeded and deterministic (`get_unique_relations` sorts before
shuffling), so a given seed reproduces the same train/dev/test relation partition,
and the three methods on the same split are paired.

Single run (Protocol A, standard):

```bash
# exact main-table reproduction (unmodified script):
python3 train_original.py --config configs/config_wiki_zsl_innovation2.yaml --log_dir logs/scar_wikizsl
# equivalent via the dual-protocol script (num_dev_rel_types absent => Protocol A):
python3 train.py          --config configs/config_wiki_zsl_innovation2.yaml --log_dir logs/scar_wikizsl
```

- **Baseline** = `*_repro*` configs, **CCA** = `*_cascade*`, **SCAR** = `*_innovation2*`.
- Threshold search + checkpoint selection use Macro-F1 (`threshold_search_metric`).

### E1 — independent dev-split (leakage-free selection)

Three-way, relation-disjoint split (`num_dev_rel_types`): threshold + checkpoint
are selected on an independent **dev** relation set; the **test** relations are
scored once at the dev-selected threshold and never used for selection
(`<log_dir>/dev_test_results.json`).

```bash
bash scripts/launch_e1_devsplit.sh "0 1 2 3 4"   # GPU ids
python3 scripts/collect_e1_results.py            # paired table + per-split deltas + paired t-test
```

### E2 — controlled learning-rate

```bash
bash scripts/launch_e2_controlled_lr.sh "0 1 2 3 4"
```

### E3 — calibration diagnostics (no GPU)

```bash
python3 analysis/compute_calibration.py          # ECE / Brier / NLL from cached full-vocabulary scores
```

### Multi-GPU scheduler (E1 + E2, resilient)

```bash
python3 scripts/run_rebuttal_queue.py 0 1 2 3 4  # 1 job/GPU, auto-retry, skips completed (_SUCCESS)
```

## Notes

- Configs use absolute paths (`model_name`, `prev_path`, `root_dir`) reflecting the
  original run environment; adjust them to your setup.
- ISCL is training-only and removed at inference; only CCA's additive fusion
  persists (~0.1% params, ~2% throughput overhead).
- This work builds on the public GLiREL joint-encoding framework (cited in the paper).
