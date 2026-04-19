from utils import utils
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--golds_path", type=str, required=True)
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        required=False,
        default=[1, 2, 3, 5, 10, 20, 50, 100, 200],
    )
    return parser.parse_args()


def ndcg_at_k(relevances, k, num_relevant):
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))
    return dcg / idcg if idcg > 0 else 0


def evaluate_metrics(preds, golds, k):
    tp, fp, fn = 0, 0, 0
    matched_queries = 0
    correct_queries = 0
    ndcg_scores = []

    for qid in preds['default']["test"]:
        if qid in golds:
            matched_queries += 1
            gold_docs = set(golds[qid])

            # Sort by score descending and take top-k
            pred_dict = preds['default']['test'][qid]
            pred_docs = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
            pred_docs = [doc_id for doc_id, score in pred_docs[:k]]
            pred_docs_set = set(pred_docs)

            # Relevance list for NDCG
            relevances = [1 if doc_id in gold_docs else 0 for doc_id in pred_docs]
            ndcg_scores.append(ndcg_at_k(relevances, k, len(gold_docs)))

            # Accuracy
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
    """Pretty print evaluation metrics in a formatted table."""
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
    preds = utils.read_json(args.predictions_path)
    golds = utils.read_json(args.golds_path)
    metrics = [evaluate_metrics(preds, golds, k) for k in args.topk]
    print_metrics(metrics)
