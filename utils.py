import json
import re
from urllib.parse import urlparse


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tokenize(text: str):
    text = normalize_text(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    bigrams = ["_".join(words[i:i + 2]) for i in range(len(words) - 1)]
    trigrams = ["_".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return words + bigrams + trigrams


def split_sentences(text: str):
    text = normalize_text(text)
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 6]


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name[:180]


def is_generic_noise_text(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True
    lower = t.lower()
    words = re.findall(r"[a-z0-9]+", lower)
    if len(words) < 12:
        return True
    if "................" in lower:
        return True
    bad_page_patterns = [
        "table of contents", "intellectual property rights", "foreword",
        "modal verbs terminology", "change history", "history",
    ]
    if len(words) < 120 and any(p in lower for p in bad_page_patterns):
        return True
    clause_refs = re.findall(r"\b\d+\.\d+(?:\.\d+)*\b", lower)
    if len(clause_refs) > 65:
        return True
    rp_like_ids = re.findall(r"\b(?:rp|r2|r3|r4|r5)-\d+\b", lower)
    if len(rp_like_ids) > 12:
        return True
    if len(words) > 150:
        unique_ratio = len(set(words)) / max(1, len(words))
        if unique_ratio < 0.13:
            return True
    return False


def is_bad_answer(answer: str) -> bool:
    a = normalize_text(answer).lower()
    if not a:
        return True
    bad_patterns = [
        "table of contents", "rapporteur", "agenda item", "change history",
        "revision history", "copyright notification", "intellectual property rights",
        "not enough evidence but",
    ]
    if any(x in a for x in bad_patterns):
        return True
    if len(re.findall(r"\b(?:rp|r2|r3|r4|r5)-\d+\b", a)) > 4:
        return True
    return False


def fmt(x) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)
