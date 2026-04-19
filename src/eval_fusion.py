import json
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--predictions_path_qwen", type=str, required=True)
    parser.add_argument("--golds_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        required=False,
        default=[1, 2, 3, 5, 10, 20, 50, 100, 200],
    )
    return parser.parse_args()


def read_json(file_path: str):
    preds = []
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            preds = json.load(f)
        return preds
    else:
        pred_data = []
        with open(file_path, "r") as f:
            for line in f:
                pred_data.append(json.loads(line))

        preds = {"default": {"test": {}}}
        for pred in pred_data:
            preds["default"]["test"][pred["qid"]] = pred["retrieved"]

        return preds


def _minmax(vals):
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def _linear_fusion(retrieved: dict, rerank_scores: dict, alpha: float) -> dict:
    """alpha * normalised_rerank + (1-alpha) * normalised_retrieval."""
    cand_ids = list(rerank_scores.keys())
    ret_vals = [retrieved.get(d, 0.0) for d in cand_ids]
    rer_vals = [rerank_scores[d] for d in cand_ids]
    norm_ret = _minmax(ret_vals)
    norm_rer = _minmax(rer_vals)
    return {
        did: alpha * nr + (1.0 - alpha) * nret
        for did, nr, nret in zip(cand_ids, norm_rer, norm_ret)
    }


def fuse_predictions(preds_retrieval: dict, preds_qwen: dict, alpha: float) -> dict:
    """
    Kết hợp điểm retrieval và qwen theo linear fusion cho từng query.
    Trả về dict cùng format với preds_retrieval.
    """
    fused = {"default": {"test": {}}}
    test_retrieval = preds_retrieval["default"]["test"]
    test_qwen = preds_qwen["default"]["test"]

    for qid in test_retrieval:
        retrieved_scores = test_retrieval[qid]   # {doc_id: score}

        if qid in test_qwen:
            rerank_scores = test_qwen[qid]       # {doc_id: score}
        else:
            # Không có qwen score → giữ nguyên retrieval
            fused["default"]["test"][qid] = retrieved_scores
            continue

        fused["default"]["test"][qid] = _linear_fusion(retrieved_scores, rerank_scores, alpha)

    return fused


def save_json(data: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def ndcg_at_k(relevances, k, num_relevant):
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))
    return dcg / idcg if idcg > 0 else 0


def evaluate_metrics(preds, golds, k):
    tp, fp, fn = 0, 0, 0
    matched_queries = 0
    correct_queries = 0
    ndcg_scores = []

    for qid in preds["default"]["test"]:
        if qid in golds:
            matched_queries += 1
            gold_docs = set(golds[qid])

            pred_dict = preds["default"]["test"][qid]
            pred_docs = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
            pred_docs = [doc_id for doc_id, score in pred_docs[:k]]
            pred_docs_set = set(pred_docs)

            relevances = [1 if doc_id in gold_docs else 0 for doc_id in pred_docs]
            ndcg_scores.append(ndcg_at_k(relevances, k, len(gold_docs)))

            if gold_docs & pred_docs_set:
                correct_queries += 1

            for doc_id in pred_docs:
                if doc_id in gold_docs:
                    tp += 1
                else:
                    fp += 1

            for gold_doc in gold_docs:
                if gold_doc not in pred_docs_set:
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = correct_queries / matched_queries if matched_queries > 0 else 0
    ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

    return {
        "k": k,
        "matched_queries": matched_queries,
        "prec": precision,
        "recall": recall,
        "f1": f1,
        "acc": accuracy,
        "ndcg": ndcg,
    }


def print_metrics(metrics: dict) -> None:
    header = f"{'k':>6} │ {'Prec':>8} │ {'Recall':>8} │ {'F1':>8} │ {'NDCG':>8} │ {'Acc':>8}"
    separator = "─" * len(header)

    print("\n" + "═" * len(header))
    print("  RETRIEVAL EVALUATION RESULTS")
    print("═" * len(header))
    print(header)
    print(separator)

    for scores in metrics:
        print(
            f"{scores['k']:>6} │ {scores['prec']:>8.4f} │ {scores['recall']:>8.4f} │ "
            f"{scores['f1']:>8.4f} │ {scores['ndcg']:>8.4f} │ {scores['acc']:>8.4f}"
        )

    print(separator)
    print()


if __name__ == "__main__":
    args = parse_args()

    preds_retrieval = read_json(args.predictions_path)
    preds_qwen = read_json(args.predictions_path_qwen)
    golds = read_json(args.golds_path)

    # Fuse
    fused = fuse_predictions(preds_retrieval, preds_qwen, args.alpha)

    # Lưu file
    save_json(fused, args.output_path)
    print(f"Saved fused predictions to {args.output_path}")

    # Evaluate
    metrics = [evaluate_metrics(fused, golds, k) for k in args.topk]
    print_metrics(metrics)