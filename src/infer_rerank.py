"""
Inference & Rerank script

# Dùng Lightning checkpoint (.ckpt):
    python infer_rerank.py \
        --model_path checkpoints/best-epoch=0-val_loss=0.1866.ckpt \
        --model_type ckpt \
        --queries_path data/queries.json \
        --corpus_path data/corpus.json \
        --retrieval_path data/retrieval_results.jsonl \
        --output_path data/reranked.jsonl

# Dùng HuggingFace model (folder hoặc model name):
    python infer_rerank.py \
        --model_path allenai/longformer-base-4096 \
        --model_type hf \
        --queries_path data/queries.json \
        --corpus_path data/corpus.json \
        --retrieval_path data/retrieval_results.jsonl \
        --output_path data/reranked.jsonl
"""

import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LongformerTokenizer, LongformerForSequenceClassification

from TransformerColiee import TransformerColiee


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class InferDataset(Dataset):
    def __init__(self, queries, corpus, retrieval_results, tokenizer,
                 max_query_words=2000, max_doc_words=2000, max_seq_len=4096, top_k=100):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.top_k = top_k

        # index corpus by docid
        corpus_index = {d["docid"]: d["text"] for d in corpus}

        # truncate by words
        def truncate(text, n_words):
            return " ".join(text.split()[:n_words])

        self.samples = []
        for item in retrieval_results:
            qid = item["qid"]
            query_text = truncate(queries[qid], max_query_words)

            retrieved_items = list(item["retrieved"].items())[:self.top_k]
            for docid, bm25_score in retrieved_items:
                if docid not in corpus_index:
                    continue
                doc_text = truncate(corpus_index[docid], max_doc_words)
                self.samples.append({
                    "qid": qid,
                    "docid": docid,
                    "query": query_text,
                    "doc": doc_text,
                    "bm25_score": bm25_score,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        encoding = self.tokenizer(
            s["query"],
            s["doc"],
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # global attention on CLS token
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[0] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "qid": s["qid"],
            "docid": s["docid"],
        }


# ─────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────
def collate_fn(batch):
    return {
        "input_ids":             torch.stack([x["input_ids"] for x in batch]),
        "attention_mask":        torch.stack([x["attention_mask"] for x in batch]),
        "global_attention_mask": torch.stack([x["global_attention_mask"] for x in batch]),
        "qid":   [x["qid"] for x in batch],
        "docid": [x["docid"] for x in batch],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ──
    print(f"Loading model ({args.model_type}): {args.model_path}")

    if args.model_type == "ckpt":
        # Lightning checkpoint → dùng TransformerColiee wrapper
        pl_model = TransformerColiee.load_from_checkpoint(
            args.model_path,
            map_location=device,
        )
        pl_model.eval()
        pl_model.to(device)
        hf_model  = pl_model.model      # LongformerForSequenceClassification
        tokenizer = pl_model.tokenizer

        def get_logits(input_ids, attention_mask, global_attention_mask):
            return pl_model(input_ids, attention_mask, global_attention_mask)

    elif args.model_type == "hf":
        # HuggingFace model folder hoặc model hub name
        tokenizer = LongformerTokenizer.from_pretrained(args.model_path)
        hf_model  = LongformerForSequenceClassification.from_pretrained(
            args.model_path, num_labels=1
        )
        hf_model.eval()
        hf_model.to(device)

        def get_logits(input_ids, attention_mask, global_attention_mask):
            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            return outputs.logits.squeeze(-1)

    else:
        raise ValueError("--model_type phải là 'ckpt' hoặc 'hf'")

    # ── Load data ──
    with open(args.queries_path) as f:
        queries_list = json.load(f)
    queries = {q["qid"]: q["text"] for q in queries_list}

    with open(args.corpus_path) as f:
        corpus = json.load(f)

    retrieval_results = []
    with open(args.retrieval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                retrieval_results.append(json.loads(line))

    print(f"Queries: {len(queries)} | Corpus: {len(corpus)} | Retrieved queries: {len(retrieval_results)}")

    # ── Dataset & DataLoader ──
    dataset = InferDataset(
        queries=queries,
        corpus=corpus,
        retrieval_results=retrieval_results,
        tokenizer=tokenizer,
        max_query_words=args.max_query_words,
        max_doc_words=args.max_doc_words,
        max_seq_len=args.max_seq_len,
        top_k=args.top_k,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"Total pairs to score: {len(dataset)}")

    # ── Inference ──
    scores = defaultdict(dict)  # scores[qid][docid] = score

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring"):
            input_ids             = batch["input_ids"].to(device)
            attention_mask        = batch["attention_mask"].to(device)
            global_attention_mask = batch["global_attention_mask"].to(device)

            logits = get_logits(input_ids, attention_mask, global_attention_mask)
            probs  = torch.sigmoid(logits).cpu().tolist()

            for qid, docid, score in zip(batch["qid"], batch["docid"], probs):
                scores[qid][docid] = round(score, 6)

    # ── Sort & Save ──
    print(f"Saving reranked results to: {args.output_path}")
    with open(args.output_path, "w") as f:
        for qid, doc_scores in scores.items():
            # sort descending by score
            sorted_docs = dict(
                sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            )
            f.write(json.dumps({"qid": qid, "retrieved": sorted_docs}) + "\n")

    print("Done!")


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",      required=True,  help="Path to .ckpt hoặc HuggingFace model folder/name")
    parser.add_argument("--model_type",      required=True,  choices=["ckpt", "hf"], help="'ckpt' cho Lightning checkpoint, 'hf' cho HuggingFace model")
    parser.add_argument("--queries_path",    required=True)
    parser.add_argument("--corpus_path",     required=True)
    parser.add_argument("--retrieval_path",  required=True)
    parser.add_argument("--output_path",     required=True)

    parser.add_argument("--top_k",           type=int, default=100, help="Số doc tối đa lấy từ retrieved mỗi query")
    parser.add_argument("--max_doc_words",   type=int, default=2000)
    parser.add_argument("--max_seq_len",     type=int, default=4096)
    parser.add_argument("--batch_size",      type=int, default=8)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--device",          type=str, default="cuda")

    args = parser.parse_args()
    main(args)
