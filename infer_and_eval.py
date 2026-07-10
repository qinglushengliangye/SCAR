#!/usr/bin/env python3
"""
GLiREL 完整推理评估脚本
支持 FewRel 和 Wiki-ZSL 全数据集推理
核心指标：Micro/Macro F1、文本处理速度
结果输出到日志文件和控制台

用法:
    # 用预训练模型直接评估（零样本）
    python infer_and_eval.py \
        --model /root/autodl-tmp/models/glirel-large-v0 --datasets both

    # 用微调后的模型分别评估
    python infer_and_eval.py \
        --fewrel-model logs_repro/fewrel/model_XXXX \
        --wiki-model   logs_repro/wiki_zsl/model_XXXX \
        --datasets both

    # 搜索最优 threshold
    python infer_and_eval.py \
        --model /root/autodl-tmp/models/glirel-large-v0 \
        --datasets both --search-threshold
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import torch

# ---------------------------------------------------------------------------
# 日志设置
# ---------------------------------------------------------------------------
os.makedirs("logs_repro", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"logs_repro/eval_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_unique_relations(data):
    rels = set()
    for item in data:
        for r in item["relations"]:
            rels.add(r["relation_text"])
    return sorted(rels)


def split_data_by_relation_type(data, num_unseen_rel_types, seed=42):
    """与 train.py 相同的 zero-shot 拆分逻辑。"""
    unique_relations = get_unique_relations(data)
    original_num = num_unseen_rel_types
    current_seed = seed

    logger.info(
        f"数据集拆分：共 {len(unique_relations)} 种关系，"
        f"目标 eval 关系数={original_num}"
    )
    count = 0
    while True:
        random.seed(current_seed)
        shuffled = list(unique_relations)
        random.shuffle(shuffled)
        test_rels = set(shuffled[:num_unseen_rel_types])
        train_rels = set(shuffled[num_unseen_rel_types:])

        train_data, test_data, skipped = [], [], []
        for item in data:
            item_rels = {r["relation_text"] for r in item["relations"]}
            if item_rels.issubset(test_rels):
                test_data.append(item)
            elif item_rels.issubset(train_rels):
                train_data.append(item)
            else:
                skipped.append(item)

        actual_eval_rels = {
            r["relation_text"]
            for item in test_data
            for r in item["relations"]
        }
        if len(actual_eval_rels) == original_num:
            logger.info(
                f"  拆分完成: train={len(train_data)}, "
                f"eval={len(test_data)}, skipped={len(skipped)}"
            )
            return train_data, test_data

        num_unseen_rel_types = (
            num_unseen_rel_types + 1
            if num_unseen_rel_types < original_num * 2
            else original_num
        )
        current_seed += 1
        count += 1
        if count % 100 == 0:
            logger.info(f"  拆分尝试 {count} 次...")


# ---------------------------------------------------------------------------
# 核心评估函数
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model,
    data,
    dataset_name,
    eval_batch_size=32,
    threshold=0.5,
    top_k=1,
    fixed_relation_types=True,
    num_unseen_rel_types=15,
):
    """在完整数据集上按 zero-shot 协议推理，返回评估指标和速度统计。"""
    logger.info("\n" + "=" * 60)
    logger.info(f"评估数据集: {dataset_name}")
    logger.info(f"全数据集样本数: {len(data)}")

    _, eval_data = split_data_by_relation_type(data, num_unseen_rel_types)
    eval_rel_types = get_unique_relations(eval_data)

    logger.info(f"Eval 分片样本数: {len(eval_data)}")
    logger.info(f"Eval 关系类型数: {len(eval_rel_types)}")
    logger.info(f"Eval 关系类型 (前20): {eval_rel_types[:20]}")

    t_start = time.perf_counter()

    results_str, metric_dict = model.evaluate(
        eval_data,
        flat_ner=True,
        threshold=threshold,
        batch_size=eval_batch_size,
        relation_types=eval_rel_types if fixed_relation_types else [],
        top_k=top_k,
        dataset_name=dataset_name,
    )

    elapsed = time.perf_counter() - t_start

    num_sentences = len(eval_data)
    total_tokens = sum(len(item["tokenized_text"]) for item in eval_data)
    total_relations = sum(len(item["relations"]) for item in eval_data)
    sentences_per_sec = num_sentences / elapsed
    tokens_per_sec = total_tokens / elapsed
    relations_per_sec = total_relations / elapsed

    logger.info(f"\n--- [{dataset_name}] 评估结果 ---")
    logger.info(results_str.strip())
    logger.info(f"Best threshold: {metric_dict.get('best_threshold', threshold)}")
    logger.info(f"\n--- [{dataset_name}] 速度统计 ---")
    logger.info(f"总推理时间      : {elapsed:.2f} 秒")
    logger.info(f"样本数          : {num_sentences}")
    logger.info(f"文本处理速度    : {sentences_per_sec:.2f} sentences/s")
    logger.info(f"Token 处理速度  : {tokens_per_sec:.2f} tokens/s")
    logger.info(f"关系实例数      : {total_relations}")
    logger.info(f"关系处理速度    : {relations_per_sec:.2f} relations/s")

    speed = {
        "elapsed_sec": elapsed,
        "num_sentences": num_sentences,
        "total_tokens": total_tokens,
        "total_relations": total_relations,
        "sentences_per_sec": sentences_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "relations_per_sec": relations_per_sec,
    }
    return metric_dict, speed


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLiREL 完整推理评估")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径（同时用于 FewRel 和 Wiki-ZSL）")
    parser.add_argument("--fewrel-model", type=str, default=None,
                        help="FewRel 专用微调模型路径")
    parser.add_argument("--wiki-model", type=str, default=None,
                        help="Wiki-ZSL 专用微调模型路径")
    parser.add_argument("--datasets", type=str, default="both",
                        choices=["fewrel", "wiki_zsl", "both"])
    parser.add_argument("--fewrel-data", type=str,
                        default="data/few_rel_all.jsonl")
    parser.add_argument("--wiki-data", type=str,
                        default="data/wiki_zsl_all.jsonl")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--search-threshold", action="store_true",
                        help="启用多 threshold 搜索（更准确但更慢）")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--num-unseen-rel-types", type=int, default=15,
                        help="zero-shot 拆分 eval 关系类型数")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GLiREL 完整复现推理评估")
    logger.info(f"日志文件: {os.path.abspath(LOG_FILE)}")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu.name} | {gpu.total_memory // 1024**3} GB")
    else:
        logger.info("设备: CPU")
    logger.info("=" * 60)

    threshold = (
        [0.01, 0.1, 0.2, 0.3, 0.5, 0.6]
        if args.search_threshold
        else args.threshold
    )
    logger.info(f"使用 threshold: {threshold}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}

    # ------------------------------------------------------------------
    # FewRel
    # ------------------------------------------------------------------
    if args.datasets in ("fewrel", "both"):
        model_path = args.fewrel_model or args.model
        if not model_path:
            logger.error("请通过 --model 或 --fewrel-model 指定模型路径")
            sys.exit(1)

        logger.info(f"\n加载 FewRel 模型: {model_path}")
        from glirel import GLiREL
        model = GLiREL.from_pretrained(model_path, map_location=device)
        model = model.to(device)
        model.eval()

        fewrel_data = load_jsonl(args.fewrel_data)
        logger.info(f"FewRel 全数据集: {len(fewrel_data)} 条")

        metrics, speed = evaluate_dataset(
            model=model,
            data=fewrel_data,
            dataset_name="few_rel",
            eval_batch_size=args.eval_batch_size,
            threshold=threshold,
            top_k=args.top_k,
            fixed_relation_types=True,
            num_unseen_rel_types=args.num_unseen_rel_types,
        )
        summary["fewrel"] = {"metrics": metrics, "speed": speed}
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Wiki-ZSL
    # ------------------------------------------------------------------
    if args.datasets in ("wiki_zsl", "both"):
        model_path = args.wiki_model or args.model
        if not model_path:
            logger.error("请通过 --model 或 --wiki-model 指定模型路径")
            sys.exit(1)

        logger.info(f"\n加载 Wiki-ZSL 模型: {model_path}")
        from glirel import GLiREL
        model = GLiREL.from_pretrained(model_path, map_location=device)
        model = model.to(device)
        model.eval()

        wiki_data = load_jsonl(args.wiki_data)
        logger.info(f"Wiki-ZSL 全数据集: {len(wiki_data)} 条")

        metrics, speed = evaluate_dataset(
            model=model,
            data=wiki_data,
            dataset_name="wiki_zsl",
            eval_batch_size=args.eval_batch_size,
            threshold=threshold,
            top_k=args.top_k,
            fixed_relation_types=True,
            num_unseen_rel_types=args.num_unseen_rel_types,
        )
        summary["wiki_zsl"] = {"metrics": metrics, "speed": speed}
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 汇总报告
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("最终汇总报告")
    logger.info("=" * 60)

    for ds_key, label in [("fewrel", "FewRel"), ("wiki_zsl", "Wiki-ZSL")]:
        if ds_key not in summary:
            continue
        m = summary[ds_key]["metrics"]
        s = summary[ds_key]["speed"]
        target_met = m["micro_f1"] >= 0.80
        logger.info(f"\n[{label}]")
        logger.info(f"  Micro F1        : {m['micro_f1']:.4f}  ({m['micro_f1']*100:.2f}%)")
        logger.info(f"  Macro F1        : {m['macro_f1']:.4f}  ({m['macro_f1']*100:.2f}%)")
        logger.info(f"  Micro Precision : {m['micro_precision']:.4f}")
        logger.info(f"  Micro Recall    : {m['micro_recall']:.4f}")
        logger.info(f"  文本处理速度    : {s['sentences_per_sec']:.2f} sentences/s")
        logger.info(f"  Token 处理速度  : {s['tokens_per_sec']:.2f} tokens/s")
        logger.info(f"  关系处理速度    : {s['relations_per_sec']:.2f} relations/s")
        logger.info(f"  推理总时间      : {s['elapsed_sec']:.2f} s")
        logger.info(f"  论文目标(>=0.80): {'[PASS] 达标' if target_met else '[FAIL] 未达标'}")

    if len(summary) == 2:
        avg_micro = (
            summary["fewrel"]["metrics"]["micro_f1"] +
            summary["wiki_zsl"]["metrics"]["micro_f1"]
        ) / 2
        avg_macro = (
            summary["fewrel"]["metrics"]["macro_f1"] +
            summary["wiki_zsl"]["metrics"]["macro_f1"]
        ) / 2
        avg_speed = (
            summary["fewrel"]["speed"]["sentences_per_sec"] +
            summary["wiki_zsl"]["speed"]["sentences_per_sec"]
        ) / 2
        both_pass = (
            summary["fewrel"]["metrics"]["micro_f1"] >= 0.80 and
            summary["wiki_zsl"]["metrics"]["micro_f1"] >= 0.80
        )
        logger.info("\n[两数据集平均]")
        logger.info(f"  平均 Micro F1   : {avg_micro:.4f}  ({avg_micro*100:.2f}%)")
        logger.info(f"  平均 Macro F1   : {avg_macro:.4f}  ({avg_macro*100:.2f}%)")
        logger.info(f"  平均文本处理速度: {avg_speed:.2f} sentences/s")
        logger.info(f"  两数据集均>=0.80: {'[PASS]' if both_pass else '[FAIL]'}")

    logger.info(f"\n日志已保存至: {os.path.abspath(LOG_FILE)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
