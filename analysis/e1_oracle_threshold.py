"""Rebuttal E1 decomposition: capacity vs leakage.

For each leakage-free (83-relation) Baseline checkpoint, evaluate the untouched
TEST relations at:
  (a) the threshold selected on the disjoint DEV relations  -> clean protocol
  (b) the threshold that maximises TEST macro-F1 (oracle)    -> upper bound on
      the "shared-set threshold tuning" that the standard protocol allows.
The gap (b)-(a) is the net magnitude of threshold-selection leakage on the SAME
model; everything else in the 75.9 -> 53.5 drop is the capacity effect of
training on 83 instead of ~98 relations.
"""
import json
import random
import time
import logging

import torch

from glirel import GLiREL
from glirel.model import load_config_as_namespace

logging.basicConfig(level=logging.WARNING)

GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6]
DATA_PATH = "data/wiki_zsl_all.jsonl"


def get_unique_relations(data):
    rels = []
    for item in data:
        for r in item["relations"]:
            rels.append(r["relation_text"])
    return sorted(set(rels))


def split_3way(data, num_unseen, num_dev, seed):
    unique = get_unique_relations(data)
    while True:
        random.seed(seed)
        random.shuffle(unique)
        test_rt = set(unique[:num_unseen])
        dev_rt = set(unique[num_unseen:num_unseen + num_dev])
        train_rt = set(unique[num_unseen + num_dev:])
        tr, dv, te, sk = [], [], [], []
        for item in data:
            rt = {r["relation_text"] for r in item["relations"]}
            if rt.issubset(test_rt):
                te.append(item)
            elif rt.issubset(dev_rt):
                dv.append(item)
            elif rt.issubset(train_rt):
                tr.append(item)
            else:
                sk.append(item)
        if len(get_unique_relations(te)) == num_unseen and len(get_unique_relations(dv)) == num_dev:
            return tr, dv, te
        seed = random.randint(0, 1000)


def macro_at(model, data, rel_types, thresholds):
    _, md = model.evaluate(
        data, flat_ner=True, threshold=list(thresholds), batch_size=48,
        relation_types=rel_types, top_k=1, dataset_name="wiki_zsl",
    )
    return md


def main():
    with open(DATA_PATH) as f:
        data = [json.loads(l) for l in f]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exps = [
        (1, "/root/autodl-tmp/logs-e1/repro_wikizsl/exp1/model_900"),
        (2, "/root/autodl-tmp/logs-e1/repro_wikizsl/exp2/model_900"),
        (3, "/root/autodl-tmp/logs-e1/repro_wikizsl/exp3/model_900"),
    ]
    rows = []
    for seed, ckpt in exps:
        print(f"\n===== repro exp seed={seed} ckpt={ckpt} =====", flush=True)
        _, dev_data, test_data = split_3way(data, 15, 15, seed)
        dev_rt = get_unique_relations(dev_data)
        test_rt = get_unique_relations(test_data)
        print(f"dev rels={len(dev_rt)} test rels={len(test_rt)} "
              f"test items={len(test_data)}", flush=True)

        model = GLiREL.from_pretrained(ckpt)
        model.threshold_search_metric = "macro_f1"
        model = model.to(device)
        model.device = device
        model.eval()

        # (a) select threshold on DEV, apply single threshold to TEST
        dev_md = macro_at(model, dev_data, dev_rt, GRID)
        dev_thr = dev_md["best_threshold"]
        test_at_dev = macro_at(model, test_data, test_rt, [dev_thr])
        # (b) oracle: best TEST macro-F1 over the grid
        test_oracle = macro_at(model, test_data, test_rt, GRID)

        clean = test_at_dev["macro_f1"] * 100
        oracle = test_oracle["macro_f1"] * 100
        rows.append((seed, dev_thr, clean, test_oracle["best_threshold"], oracle))
        print(f"[seed {seed}] dev_thr={dev_thr} | test@dev_thr(clean)={clean:.2f} "
              f"| oracle_thr={test_oracle['best_threshold']} oracle={oracle:.2f} "
              f"| leakage=+{oracle-clean:.2f}", flush=True)

    print("\n\n================ SUMMARY (Baseline, 83-relation model) ================")
    print(f"{'seed':>4} {'dev_thr':>8} {'clean(test@dev)':>16} {'oracle_thr':>10} {'oracle(test-tuned)':>18} {'leakage':>8}")
    for seed, dt, clean, ot, oracle in rows:
        print(f"{seed:>4} {dt:>8} {clean:>16.2f} {ot:>10} {oracle:>18.2f} {oracle-clean:>+8.2f}")
    cm = sum(r[2] for r in rows) / len(rows)
    om = sum(r[4] for r in rows) / len(rows)
    print(f"\nmean clean (test@dev)   = {cm:.2f}")
    print(f"mean oracle (test-tuned) = {om:.2f}")
    print(f"mean leakage effect      = +{om-cm:.2f}")
    print(f"main-table Baseline m=15 = 75.88  -> capacity effect ~= {75.88-om:.2f} (75.88 - oracle)")


if __name__ == "__main__":
    main()
