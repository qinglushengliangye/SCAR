"""
Extract entity-pair representations for t-SNE, with stratified sampling
to ensure ALL 113 relation types are represented.
Much faster than full_dataset_analysis.py (no threshold/prediction tracking).
"""
import sys, os

if '--gpu' in sys.argv:
    idx = sys.argv.index('--gpu')
    if idx + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[idx + 1]

sys.path.insert(0, '/root/GLiREL')

import argparse, json, pickle, logging
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from glirel import GLiREL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = '/root/GLiREL/paper/figures'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['baseline', 'cca', 'cca_iscl'])
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--data', default='/root/GLiREL/data/wiki_zsl_all.jsonl')
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--target-reps', type=int, default=6000,
                        help='Total reps to collect (stratified across all types)')
    args = parser.parse_args()

    config_path = os.path.join(args.ckpt_dir, 'glirel_config.json')
    with open(config_path) as f:
        cfg = json.load(f)
    config = argparse.Namespace(**cfg)
    model = GLiREL.from_pretrained(args.ckpt_dir, map_location='cuda').to('cuda').eval()

    with open(args.data) as f:
        data = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(data)} samples")

    all_rels = sorted(set(r['relation_text'] for item in data for r in item.get('relations', [])))
    logger.info(f"Total relation types: {len(all_rels)}")

    by_class = defaultdict(list)
    all_reps_list = []
    all_labels_list = []

    data_loader = model.create_dataloader(
        data, batch_size=args.batch_size,
        relation_types=all_rels, shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting reps"):
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            captured = {}
            orig = model.scorer.forward
            def hook(cp, rt):
                captured['rel_rep'] = cp.detach().cpu()
                return orig(cp, rt)
            model.scorer.forward = hook
            try:
                model.compute_score(batch)
            finally:
                model.scorer.forward = orig

            rel_labels = batch['rel_label'].cpu()
            rel_rep = captured['rel_rep']
            B, P, _ = rel_rep.shape

            for b in range(B):
                classes_to_id = batch['classes_to_id'][b]
                id_to_class = {v: k for k, v in classes_to_id.items()}
                for p in range(P):
                    label_id = rel_labels[b, p].item()
                    if label_id <= 0:
                        continue
                    label_name = id_to_class.get(label_id, f'UNK_{label_id}')
                    idx = len(all_reps_list)
                    all_reps_list.append(rel_rep[b, p].numpy())
                    all_labels_list.append(label_name)
                    by_class[label_name].append(idx)

            if len(all_reps_list) >= 80000:
                break

    logger.info(f"Collected {len(all_reps_list)} reps, {len(by_class)} types")

    # Stratified sampling
    rng = np.random.RandomState(42)
    n_classes = len(by_class)
    per_class = max(3, args.target_reps // n_classes)
    selected = []
    for lab, indices in by_class.items():
        k = min(per_class, len(indices))
        selected.extend(rng.choice(indices, k, replace=False).tolist())

    remaining = args.target_reps - len(selected)
    if remaining > 0:
        pool = list(set(range(len(all_reps_list))) - set(selected))
        selected.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())
    selected = sorted(set(selected))[:args.target_reps]

    reps_out = np.array([all_reps_list[i] for i in selected])
    labels_out = [all_labels_list[i] for i in selected]

    logger.info(f"Stratified sample: {len(reps_out)} reps, {len(set(labels_out))} types")

    tsne_path = os.path.join(OUTPUT_DIR, f'tsne_data_{args.method}.pkl')
    with open(tsne_path, 'wb') as f:
        pickle.dump({'reps': reps_out.tolist(), 'labels': labels_out}, f)
    logger.info(f"Saved to {tsne_path}")


if __name__ == '__main__':
    main()
