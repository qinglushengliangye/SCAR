"""Run analysis tasks from cached extraction results."""
import sys
sys.path.insert(0, '/root/GLiREL')

import pickle
import logging
from analysis.deep_analysis import (
    task1_tsne, task2_confidence, task4_error_analysis, task5_ablation_curves,
    OUTPUT_DIR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cache_path = f'{OUTPUT_DIR}/analysis_cache.pkl'
logger.info(f"Loading cached results from {cache_path}")
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

results_dict = cache['results_dict']
eval_rel_types = cache['eval_rel_types']
method_names = ['Baseline', 'CCA', 'CCA+ISCL']

for method, results in results_dict.items():
    logger.info(f"{method}: {len(results['rel_reps'])} reps, {len(results['predictions'])} preds, {len(results['gold_triplets'])} golds")

logger.info("\n=== Task 1: t-SNE ===")
task1_tsne(results_dict, method_names)

logger.info("\n=== Task 2: Confidence Distribution ===")
task2_confidence(results_dict, method_names, thresholds=[0.1, 0.1, 0.1])

logger.info("\n=== Task 4: Error Analysis ===")
task4_error_analysis(results_dict, method_names)

logger.info("\n=== Task 5: Ablation Curves ===")
task5_ablation_curves()

logger.info("\nAll tasks from cache complete!")
