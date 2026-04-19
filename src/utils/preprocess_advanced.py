import json
import re
from typing import List, Tuple
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ============================================================
# NOISE TOKENS & PATTERNS
# ============================================================

TOKENS_TO_REMOVE = [
    "[translation]",
    "[Translation]",
    "[sic]",
    "[ sic ]",
    "[Emphasis added.]",
    "[emphasis added]",
    "[End of document]",
    "*",
    "[  ]",
    "[]",
    "[ ]",
    "[DATE_SUPPRESSED]",
    "[TRANSLATION]",
    "[English language version follows French language version]",
    "[La version anglaise vient à la suite de la version française]",
    "[Diagram omitted - see printed version]",
    "[French language version follows English language version]",
    "[La version française vient à la suite de la version anglaise]",
    "[Traduction]",
]

STRUCTURAL_PATTERNS = [
    r"\[\d+\]",
    r"(?m)^[A-Z][A-Z\s]{4,}$",
    r"R\.S\.C\.\s+\d{4},\s+c\.\s+[A-Z]-?\d+",
    r"(?:section|subsection|paragraph|s\.)\s*\d+(?:\.\d+)?(?:\(\d+\))?(?:\([a-z]\))?",
]

METADATA_PATTERNS = [
    r"\s*<FRAGMENT_SUPPRESSED>\s*",
    r"Counsel:.*?(?=\n[A-Z]|\nSolicitor|\n\[|\Z)",
    r"Solicitors?\s+of\s+Record:.*?(?=\n[A-Z]|\nSummary|\n\[|\Z)",
    r"Summary:.*?(?=\n\[|\Z)",
    r"Editor:.*?(?=\n|\Z)",
    r"MLB\s+unedited\s+judgment",
    r"This\s+case\s+is\s+unedited.*?summary\.",
    r"\((?:FC|FCA|SCC|ONCA|BCCA|ABCA|ONSC|BCSC)\)",
    r"(?:Docket|File|No\.?)\s*[:.]?\s*[A-Z]{1,4}[\-]?\d+[\-\d]*",
    r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?,\s*J\.?(?:\s*:)?",
]

DATE_CITATION_PATTERNS = [
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    r"\b\d{4}\s+(?:FC|FCA|SCC|ONCA|BCCA|SCR|FCR|OR|DLR)\s+\d+\b",
    r"\[\d{4}\]\s+\d+\s+(?:FC|FCA|SCC|SCR|FCR)\s+\d+",
]

CLEANUP_PATTERNS = [
    (r"[ \t]+", " "),
    (r"\n{3,}", "\n\n"),
    (r"(?m)^[ \t]+|[ \t]+$", ""),
    (r"\s+([.,;:!?])", r"\1"),
]

# ============================================================
# HELPER REGEX CALLBACKS
# ============================================================


def _rep(match):
    return match.group().replace("[", "{").replace("]", "}")


def _rep2(match):
    return match.group().replace("{}", "[").replace("}", "]")


def _remove(match):
    return match.group().replace("[", "").replace("]", "").replace(" ", "")


def _remove2(match):
    return match.group().replace("[", "").replace("]", "")


# ============================================================
# LANGUAGE DETECTOR (GPU-accelerated, batch inference)
# ============================================================

# Model options (ordered by recommendation for H100):
#
# 1. "papluca/xlm-roberta-base-language-detection"  ← your original suggestion
#    - 278M params, 97 langs, solid accuracy
#
# 2. "facebook/fasttext-language-identification"    ← fastest, ~218 langs
#    - Uses fasttext under the hood, not a HF transformer model
#    - Requires: pip install fasttext-wheel
#
# 3. "setu4993/smaller-LaBSE"                       ← multilingual sentence encoder
#
# We default to xlm-roberta for easy HF integration + GPU batching.
# Change MODEL_NAME below to swap models.

MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
BATCH_SIZE = 256      # tune to fill H100 VRAM (80GB); increase for more throughput
MAX_LENGTH = 128      # max tokens per line; longer lines get truncated


class LanguageDetector:
    """
    Batch GPU language detector using a HuggingFace sequence classification model.
    Detects language for a list of text strings in one batched forward pass.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cuda",
        batch_size: int = BATCH_SIZE,
        max_length: int = MAX_LENGTH,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"[LangDetect] Loading model '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

        # id2label: e.g. {0: "ar", 1: "bg", ..., 19: "zh-cn"}
        self.id2label = self.model.config.id2label
        print(f"[LangDetect] Model loaded. Supports {len(self.id2label)} languages.")

    @torch.inference_mode()
    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Predict language for a batch of texts.
        Returns list of ISO 639-1 language codes (e.g. "en", "fr").
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits          # (B, num_labels)
            pred_ids = logits.argmax(dim=-1).tolist()
            results.extend([self.id2label[pid] for pid in pred_ids])

        return results

    def predict(self, text: str) -> str:
        """Predict language for a single string."""
        return self.predict_batch([text])[0]


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter French lines and noise from legal documents."
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--text_field", type=str, required=True, help="Text field name")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"HuggingFace model for language detection (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Number of lines per GPU forward pass (default: 256)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help="Max token length per line (default: 128)",
    )
    parser.add_argument(
        "--remove_structural",
        action="store_true",
        help="Remove paragraph markers and section headers",
    )
    parser.add_argument(
        "--remove_metadata",
        action="store_true",
        help="Remove counsel info, editor notes, etc.",
    )
    parser.add_argument(
        "--remove_dates",
        action="store_true",
        help="Remove dates and case citations",
    )
    return parser.parse_args()


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================


def preprocess_text(text: str) -> str:
    """Clean suppressed tags, brackets, and noisy tokens."""
    text = re.sub(r"\. *(\. *)+", "", text)
    text = re.sub(r"[A-Z]*_SUPPRESSED", "", text)
    text = text.replace("<FRAGMENT_SUPPRESSED>", "")

    for token in TOKENS_TO_REMOVE:
        text = text.replace(token, "")

    text = re.sub(r"\[[A-Z][A-Z]+\]", _rep, text)
    text = re.sub(r"[^a-zA-Z]\[[b-zB-Z]\] ", _remove, text)
    text = re.sub(r"\[[a-zA-Z][a-zA-Z \.']*\]", _remove2, text)
    text = re.sub(r"\{[A-Z][A-Z]+\}", _rep2, text)

    text = re.sub(r"\n\n+", "\n\n", text)
    text = re.sub(r"\.\.+", ".", text)
    text = re.sub(r"\n\.\n", "\n\n", text)

    return text


def remove_structural_noise(text: str) -> str:
    for pattern in STRUCTURAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def remove_metadata(text: str) -> str:
    for pattern in METADATA_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def remove_dates_citations(text: str) -> str:
    for pattern in DATE_CITATION_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def cleanup_whitespace(text: str) -> str:
    for pattern, replacement in CLEANUP_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text.strip()


# ============================================================
# FRENCH-LINE REMOVAL  (now uses GPU model)
# ============================================================


def remove_french_lines(text: str, detector: LanguageDetector) -> str:
    """
    Preprocess text then remove French lines using batch GPU language detection.

    Strategy:
    - Collect all non-empty lines, detect their language in one batched call.
    - Apply the same last_lang carry-forward logic as the original code.
    """
    text = preprocess_text(text)
    lines = text.split("\n")

    # Separate non-empty lines for batch inference
    non_empty_indices = [i for i, l in enumerate(lines) if l.strip()]
    non_empty_texts = [lines[i] for i in non_empty_indices]

    if not non_empty_texts:
        return text.strip()

    # Single batched GPU call for all non-empty lines in this document
    langs = detector.predict_batch(non_empty_texts)
    lang_map = dict(zip(non_empty_indices, langs))  # index -> lang code

    last_lang = "en"
    for i in range(len(lines)):
        if not lines[i].strip():
            continue

        lang = lang_map.get(i, "en")

        if lang == "fr":
            last_lang = "fr"
            lines[i] = ""
        elif lang != "en":
            # Ambiguous / other language: remove if last known was French
            if last_lang == "fr":
                lines[i] = ""
        else:
            last_lang = "en"

    result = "\n".join(lines)
    result = re.sub(r"\n\n+", "\n\n", result)
    return result.strip()


def clean_text(
    text: str,
    detector: LanguageDetector,
    remove_structural: bool = False,
    remove_meta: bool = False,
    remove_dates: bool = False,
) -> str:
    """Full cleaning pipeline."""
    text = remove_french_lines(text, detector)

    if remove_structural:
        text = remove_structural_noise(text)
    if remove_meta:
        text = remove_metadata(text)
    if remove_dates:
        text = remove_dates_citations(text)

    text = cleanup_whitespace(text)
    return text


# ============================================================
# BATCH FILTERING
# ============================================================


def filter_french(
    data: List[dict],
    text_field: str,
    detector: LanguageDetector,
    remove_structural: bool = False,
    remove_meta: bool = False,
    remove_dates: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    Filter French content and optionally apply additional cleaning.

    For maximum GPU utilisation on H100, we could also flatten ALL lines
    from ALL documents into one mega-batch. That approach is left as a
    comment below; enable it if per-document batching is still too slow.

    Returns:
        Tuple of (filtered_data, fully_french documents).
    """
    french_data = []
    filtered_data = []

    for item in tqdm(data, desc="Filtering documents"):
        if text_field not in item or not item[text_field]:
            filtered_data.append(item)
            continue

        cleaned = clean_text(
            item[text_field],
            detector=detector,
            remove_structural=remove_structural,
            remove_meta=remove_meta,
            remove_dates=remove_dates,
        )

        if not cleaned:
            french_data.append(item)
        else:
            filtered_data.append({**item, text_field: cleaned})

    return filtered_data, french_data


# ============================================================
# OPTIONAL: mega-batch variant for maximum H100 throughput
# ============================================================
# If you have millions of short documents, consider collecting ALL lines
# from ALL documents into one list, running detector.predict_batch() once,
# then re-distributing results. Example sketch:
#
#   all_lines = []
#   doc_line_ranges = []
#   for item in data:
#       lines = preprocess_text(item[text_field]).split("\n")
#       doc_line_ranges.append((len(all_lines), len(all_lines) + len(lines)))
#       all_lines.extend(lines)
#
#   all_langs = detector.predict_batch(all_lines)   # one giant GPU call
#
#   for item, (start, end) in zip(data, doc_line_ranges):
#       langs = all_langs[start:end]
#       # apply last_lang logic per document


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    detector = LanguageDetector(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    with open(args.input, "r") as f:
        input_data = json.load(f)

    filtered_data, french_data = filter_french(
        input_data,
        args.text_field,
        detector=detector,
        remove_structural=args.remove_structural,
        remove_meta=args.remove_metadata,
        remove_dates=args.remove_dates,
    )

    print(
        f"Total: {len(input_data)}, "
        f"Fully French: {len(french_data)}, "
        f"Kept: {len(filtered_data)}"
    )

    with open(args.output, "w") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print("Filtered file saved.")
