"""Task 3: Semantic similarity bucketing (needs model for embeddings)."""
import sys
sys.path.insert(0, '/root/GLiREL')

import pickle
import logging
import torch
from analysis.deep_analysis import (
    task3_semantic_bucketing, load_model_and_config, OUTPUT_DIR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cache_path = f'{OUTPUT_DIR}/analysis_cache.pkl'
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

results_dict = cache['results_dict']
eval_rel_types = cache['eval_rel_types']
method_names = ['Baseline', 'CCA', 'CCA+ISCL']

ckpt = '/root/autodl-tmp/logs-m=15/logs_repro_wikizsl/logs_repro_exp3/model_3900'
model, _ = load_model_and_config(ckpt, device)

bucket_results = task3_semantic_bucketing(results_dict, method_names, eval_rel_types, model, device)
logger.info(f"Bucket results: {bucket_results}")
del model
torch.cuda.empty_cache()
