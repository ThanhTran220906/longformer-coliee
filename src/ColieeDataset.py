import json
import random
import torch
from torch.utils.data import Dataset


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class ColieeDataset(Dataset):
    """
    Dataset for COLIEE Task 1 document re-ranking, compatible with Longformer.

    Expected file formats:
        queries : JSON list of {"qid": str, "text": str}
        labels  : JSON dict  of {"qid": [doc_id, ...]}
        corpus  : JSON list of {"docid": str, "text": str}
        retrieval_results (optional): JSONL, one per line:
            {"qid": str, "retrieved": {"doc_id": score, ...}}

    Each __getitem__ returns one (query, document, label) pair — same
    structure as the original MarcoDataset, so it plugs into the same
    training loop unchanged.
    """

    def __init__(
        self,
        queries_path,
        labels_path,
        corpus_path,
        tokenizer,
        mode="train",                   # "train" | "dev" | "test"
        max_seq_len=4096,               # Longformer default
        retrieval_results_path=None,    # JSONL hard negatives
        num_hard_negs=7,
        neg_strategy="gap",             # "gap" | "random" | "topdown"
        lower_bound=0.05,
        upper_bound=0.15,
        seed=42,
    ):
        random.seed(seed)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode

        # ── Load raw data ────────────────────────────────────────────────────
        queries_raw = read_json(queries_path)       # [{qid, text}]
        labels_raw  = read_json(labels_path)        # {qid: [docid, ...]}
        corpus_raw  = read_json(corpus_path)        # [{docid, text}]

        self.doc_lookup: dict[str, str] = {d["docid"]: d["text"] for d in corpus_raw}
        self.query_lookup: dict[str, str] = {q["qid"]: q["text"] for q in queries_raw}
        all_doc_ids = list(self.doc_lookup.keys())

        # ── Load retrieval results (hard negatives) ──────────────────────────
        retrieval_map: dict[str, list[tuple[str, float]]] = {}
        if retrieval_results_path:
            for entry in read_jsonl(retrieval_results_path):
                qid = entry["qid"]
                sorted_docs = sorted(
                    entry["retrieved"].items(), key=lambda x: x[1], reverse=True
                )
                retrieval_map[qid] = sorted_docs

        # ── Build flat (qid, did, label) pairs ───────────────────────────────
        # Mirrors MarcoDataset.top100: one row = one training example
        pairs = []  # list of (qid, did, label)

        # Shuffle so positives are evenly distributed
        if mode == "train":
            for q in queries_raw:
                qid = q["qid"]
                pos_ids = set(labels_raw.get(qid, []))
                pos_ids = {d for d in pos_ids if d in self.doc_lookup}
                if not pos_ids:
                    continue

                # Positive pairs
                for did in pos_ids:
                    pairs.append((qid, did, 1))

                # Negative sampling
                retrieved = retrieval_map.get(qid, [])
                neg_candidates = [
                    (did, sc) for did, sc in retrieved
                    if did not in pos_ids and did in self.doc_lookup
                ]

                neg_ids = self._sample_negatives(
                    pos_ids=pos_ids,
                    pos_ids_list=list(pos_ids),
                    neg_candidates=neg_candidates,
                    retrieved_docs=retrieved,
                    all_doc_ids=all_doc_ids,
                    num_hard_negs=num_hard_negs,
                    neg_strategy=neg_strategy,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

                for did in neg_ids:
                    pairs.append((qid, did, 0))
            random.shuffle(pairs)

        elif mode == "dev":
            for q in queries_raw:
                qid = q["qid"]
                pos_ids = set(labels_raw.get(qid, []))
                retrieved = retrieval_map.get(qid, [])  # đã sort theo score

                for did, _ in retrieved[:100]:  # top 100 là đủ
                    if did not in self.doc_lookup:
                        continue
                    label = 1 if did in pos_ids else 0
                    pairs.append((qid, did, label))

        self.pairs = pairs
        print(f"[ColieeDataset] {mode} — {len(pairs)} pairs "
              f"({sum(l for *_, l in pairs)} pos / "
              f"{sum(1-l for *_, l in pairs)} neg)")

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qid, did, label = self.pairs[idx]
        query    = self.query_lookup[qid]
        document = self.doc_lookup[did]
        return self._encode(query, document, label, qid)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode(self, query, document, label, qid):
        encoded = self.tokenizer(
            query,
            document,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation="longest_first",
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        input_ids = encoded["input_ids"]

        # Global attention on [CLS] — required by Longformer
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1

        return {
            "input_ids":             torch.tensor(input_ids),
            "attention_mask":        torch.tensor(encoded["attention_mask"]),
            "global_attention_mask": torch.tensor(global_attention_mask),
            "label":                 torch.LongTensor([label]),
            "qid":                   qid 
        }

    # ── Negative sampling (ported from create_dataset.py) ────────────────────

    @staticmethod
    def _sample_negatives(
        pos_ids, pos_ids_list, neg_candidates, retrieved_docs,
        all_doc_ids, num_hard_negs, neg_strategy,
        lower_bound, upper_bound,
    ):
        target = num_hard_negs * len(pos_ids_list)

        if neg_strategy == "random":
            pool = [d for d in all_doc_ids if d not in pos_ids]
            return random.sample(pool, min(target, len(pool)))

        elif neg_strategy == "gap":
            pos_scores = {did: sc for did, sc in retrieved_docs if did in pos_ids}
            fallback   = min((sc for _, sc in neg_candidates), default=0) * 0.9
            sorted_pids = sorted(
                pos_ids_list,
                key=lambda p: pos_scores.get(p, fallback),
                reverse=True,
            )
            pool = []
            seen = set()
            for pid in sorted_pids:
                p_score = pos_scores.get(pid, fallback)
                ranked = sorted(
                    [(d, sc) for d, sc in neg_candidates
                     if lower_bound <= p_score - sc < upper_bound],
                    key=lambda x: x[1], reverse=True,
                )
                selected = [d for d, _ in ranked[:num_hard_negs] if d not in seen]
                seen.update(selected)
                # fallback nếu không đủ
                if len(selected) < num_hard_negs:
                    fb = sorted(neg_candidates, key=lambda x: x[1], reverse=True)
                    selected += [d for d, _ in fb if d not in selected][
                        : num_hard_negs - len(selected)
                    ]
                pool.extend(selected)
            return pool[:target]

        elif neg_strategy == "topdown":
            pos_scores = {did: sc for did, sc in retrieved_docs if did in pos_ids}
            fallback   = min((sc for _, sc in neg_candidates), default=0) * 0.9
            sorted_pids = sorted(
                pos_ids_list,
                key=lambda p: pos_scores.get(p, fallback),
                reverse=True,
            )
            pool, seen = [], set()
            for pid in sorted_pids:
                p_score = pos_scores.get(pid, fallback)
                ranked = sorted(
                    [(d, sc) for d, sc in neg_candidates
                     if d not in seen and lower_bound <= p_score - sc < upper_bound],
                    key=lambda x: x[1], reverse=True,
                )
                pool.extend(d for d, _ in ranked[:num_hard_negs])
                seen.update(d for d, _ in ranked[:num_hard_negs])
            return pool[:target]

        return []  # fallback