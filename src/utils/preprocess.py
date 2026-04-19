import json
import re
from typing import List, Tuple, Optional, Set
from langdetect import detect, DetectorFactory
import argparse
from tqdm import tqdm

DetectorFactory.seed = 0

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
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter French lines and noise from legal documents."
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON file path"
    )
    parser.add_argument("--text_field", type=str, required=True, help="Text field name")
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
        "--remove_dates", action="store_true", help="Remove dates and case citations"
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
    """Remove structural markers like paragraph numbers and headers."""
    for pattern in STRUCTURAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def remove_metadata(text: str) -> str:
    """Remove metadata and boilerplate sections."""
    for pattern in METADATA_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def remove_dates_citations(text: str) -> str:
    """Remove dates and case citations."""
    for pattern in DATE_CITATION_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def cleanup_whitespace(text: str) -> str:
    """Clean up extra whitespace and punctuation."""
    for pattern, replacement in CLEANUP_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def remove_french_lines(text: str) -> str:
    """
    Preprocess text then remove French lines using line-by-line detection.
    Uses a last_lang state so that ambiguous lines following French
    lines are also removed.
    """
    text = preprocess_text(text)

    lines = text.split("\n")
    last_lang = "en"
    for i in range(len(lines)):
        if len(lines[i].strip()) == 0:
            continue
        try:
            lang = detect(lines[i])
        except:
            if last_lang == "fr":
                lines[i] = ""
            continue

        if lang == "fr":
            last_lang = "fr"
            lines[i] = ""
        elif lang != "en":
            if last_lang == "fr":
                lines[i] = ""
        else:
            last_lang = "en"

    result = "\n".join(lines)
    result = re.sub(r"\n\n+", "\n\n", result)
    return result.strip()


def clean_text(
    text: str,
    remove_structural: bool = False,
    remove_meta: bool = False,
    remove_dates: bool = False,
) -> str:
    """
    Full cleaning pipeline: preprocess, remove French, then optionally
    remove structural noise, metadata, dates/citations.
    """
    text = remove_french_lines(text)

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
    remove_structural: bool = False,
    remove_meta: bool = False,
    remove_dates: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    Filter French content and optionally apply additional cleaning.

    Returns:
        Tuple of (filtered_data, fully_french documents).
    """
    french_data = []
    filtered_data = []
    for item in tqdm(data):
        if text_field not in item or not item[text_field]:
            filtered_data.append(item)
            continue

        cleaned = clean_text(
            item[text_field],
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
# MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    with open(args.input, "r") as f:
        input_data = json.load(f)

    filtered_data, french_data = filter_french(
        input_data,
        args.text_field,
        remove_structural=args.remove_structural,
        remove_meta=args.remove_metadata,
        remove_dates=args.remove_dates,
    )

    print(
        f"Total: {len(input_data)}, Fully French: {len(french_data)}, Kept: {len(filtered_data)}"
    )

    with open(args.output, "w") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print("Filtered file saved.")
