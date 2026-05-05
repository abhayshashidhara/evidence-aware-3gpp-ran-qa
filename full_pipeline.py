import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi


# =========================================================
# Embedded allowed sources
# =========================================================

DEFAULT_PRIMARY_PDF_URL = (
    "https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/"
    "18.01.00_60/ts_138331v180100p.pdf"
)

DEFAULT_ADAPTIVE_URLS = [
    "https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/18.01.00_60/ts_138331v180100p.pdf",
    "https://en.wikipedia.org/wiki/Radio_Resource_Control",
    "https://www.3glteinfo.com/5g/protocols/rrc/",
    "https://www.telcomaglobal.com/p/5g-nr-rrc",
    "https://www.ericsson.com/en/blog/2019/5/lte-nr-interworking-in-5g",
    "https://www.eventhelix.com/5G/",
    "https://www.qualcomm.com/research/5g/5g-nr",
    "https://www.techplayon.com/5gnr/",
    "https://en.wikipedia.org/wiki/5G_NR",
    "https://www.3gpp.org/technologies/5g-system-overview",
]


# =========================================================
# Basic helpers
# =========================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    bigrams = ["_".join(words[i:i + 2]) for i in range(len(words) - 1)]
    trigrams = ["_".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return words + bigrams + trigrams


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 6]


def fmt(x) -> str:
    if x is None:
        return "N/A"

    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name[:180]


# =========================================================
# Noise filters
# =========================================================

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
        "table of contents",
        "intellectual property rights",
        "foreword",
        "modal verbs terminology",
        "change history",
        "history",
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
        "table of contents",
        "rapporteur",
        "agenda item",
        "change history",
        "revision history",
        "copyright notification",
        "intellectual property rights",
        "not enough evidence but",
    ]

    if any(x in a for x in bad_patterns):
        return True

    if len(re.findall(r"\b(?:rp|r2|r3|r4|r5)-\d+\b", a)) > 4:
        return True

    return False


# =========================================================
# Chunk model
# =========================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    source_type: str
    source_title: str
    source_url: Optional[str]


# =========================================================
# PDF loading
# =========================================================

def extract_pdf_pages_from_bytes(pdf_bytes: bytes, source_title: str) -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = normalize_text(text)

        if text:
            pages.append(
                {
                    "page": i,
                    "text": text,
                    "source_title": source_title,
                }
            )

    if not pages:
        raise ValueError(
            "The PDF was opened, but no extractable text was found. "
            "If this is a scanned PDF, OCR is required."
        )

    return pages


def extract_pdf_pages_from_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        return extract_pdf_pages_from_bytes(f.read(), os.path.basename(path))


def download_pdf_to_cache(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)

    filename = safe_filename_from_url(url)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    out_path = os.path.join(cache_dir, filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_path

    print(f"Downloading primary PDF:\n{url}")

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 Evidence-Aware-RAG/1.0"},
        timeout=60,
    )
    response.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(response.content)

    print(f"Saved PDF to: {out_path}")
    return out_path


def load_primary_pages(args) -> List[Dict[str, Any]]:
    if args.pdf_path:
        if not os.path.exists(args.pdf_path):
            raise FileNotFoundError(f"Could not find local PDF: {args.pdf_path}")
        return extract_pdf_pages_from_file(args.pdf_path)

    pdf_path = download_pdf_to_cache(DEFAULT_PRIMARY_PDF_URL, args.source_cache_dir)
    return extract_pdf_pages_from_file(pdf_path)


# =========================================================
# Chunk building
# =========================================================

def build_chunks_from_pages(
    pages: List[Dict[str, Any]],
    source_title: str,
    source_type: str,
    source_url: Optional[str],
    words_per_chunk: int,
    overlap_words: int,
    min_words: int = 80,
) -> List[Chunk]:

    stream = []

    for page in pages:
        page_num = int(page.get("page", 0))
        text = normalize_text(page.get("text", ""))

        if len(text.split()) < 20:
            continue

        if is_generic_noise_text(text):
            continue

        for word in text.split():
            stream.append((word, page_num))

    chunks = []
    start = 0
    idx = 0
    step = max(1, words_per_chunk - overlap_words)

    while start < len(stream):
        end = min(len(stream), start + words_per_chunk)
        window = stream[start:end]

        if len(window) < min_words:
            break

        words = [w for w, _ in window]
        page_nums = [p for _, p in window]
        text = " ".join(words)

        if not is_generic_noise_text(text):
            chunks.append(
                Chunk(
                    chunk_id=f"{source_type}_{idx:06d}",
                    text=text,
                    page_start=min(page_nums) if page_nums else None,
                    page_end=max(page_nums) if page_nums else None,
                    source_type=source_type,
                    source_title=source_title,
                    source_url=source_url,
                )
            )
            idx += 1

        start += step

    return chunks


# =========================================================
# BM25 retriever
# =========================================================

class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokenized = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if chunks else None

    def search(
        self,
        query: str,
        top_k: int = 8,
        prefetch_k: int = 120,
    ) -> List[Dict[str, Any]]:

        if not self.chunks or self.bm25 is None:
            return []

        q_tokens = tokenize(query)

        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:prefetch_k]

        out = []

        for i in ranked:
            c = self.chunks[i]

            if is_generic_noise_text(c.text):
                continue

            out.append(
                {
                    "rank": len(out) + 1,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "score": float(scores[i]),
                    "source_type": c.source_type,
                    "source_title": c.source_title,
                    "source_url": c.source_url,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                }
            )

            if len(out) >= top_k:
                break

        return out


# =========================================================
# Primary retriever
# =========================================================

def load_or_build_primary_retriever(args) -> BM25Retriever:
    cache_path = args.primary_cache_json

    cache_key = {
        "primary_pdf_url": DEFAULT_PRIMARY_PDF_URL,
        "local_pdf_path": os.path.abspath(args.pdf_path) if args.pdf_path else None,
        "words_per_chunk": args.words_per_chunk,
        "overlap_words": args.overlap_words,
    }

    if args.use_primary_cache and os.path.exists(cache_path) and not args.rebuild_source_cache:
        try:
            cached = read_json(cache_path)

            if cached.get("cache_key") == cache_key and cached.get("chunks"):
                chunks = [
                    Chunk(
                        chunk_id=x["chunk_id"],
                        text=x["text"],
                        page_start=x.get("page_start"),
                        page_end=x.get("page_end"),
                        source_type=x["source_type"],
                        source_title=x["source_title"],
                        source_url=x.get("source_url"),
                    )
                    for x in cached["chunks"]
                ]

                print(f"Loaded primary cache: {cache_path}")
                print(f"Primary chunks: {len(chunks)}")
                return BM25Retriever(chunks)

        except Exception as e:
            print(f"Could not load primary cache. Rebuilding. Reason: {e}")

    pages = load_primary_pages(args)
    source_title = os.path.basename(args.pdf_path) if args.pdf_path else "ETSI TS 138 331 v18.1.0 PDF"

    chunks = build_chunks_from_pages(
        pages=pages,
        source_title=source_title,
        source_type="primary_pdf",
        source_url=args.pdf_path if args.pdf_path else DEFAULT_PRIMARY_PDF_URL,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.overlap_words,
    )

    print(f"Loaded primary PDF pages: {len(pages)}")
    print(f"Built primary PDF chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No primary chunks were built. Check PDF extraction.")

    if args.use_primary_cache:
        write_json(
            cache_path,
            {
                "cache_key": cache_key,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "source_type": c.source_type,
                        "source_title": c.source_title,
                        "source_url": c.source_url,
                    }
                    for c in chunks
                ],
            },
        )
        print(f"Saved primary cache: {cache_path}")

    return BM25Retriever(chunks)


# =========================================================
# Adaptive retrieval from embedded sources only
# =========================================================

def fetch_html_text(url: str, timeout: int = 30) -> Dict[str, str]:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 Evidence-Aware-RAG/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        pages = extract_pdf_pages_from_bytes(response.content, source_title=url)
        text = " ".join(p["text"] for p in pages)
        return {
            "url": url,
            "title": "ETSI TS 138 331 v18.1.0 PDF",
            "text": text,
        }

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "svg", "footer", "nav"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = normalize_text(soup.get_text(" ", strip=True))

    return {
        "url": url,
        "title": title,
        "text": text,
    }


def build_web_chunks(page: Dict[str, str]) -> List[Chunk]:
    words = page["text"].split()

    if len(words) < 80:
        return []

    parsed = urlparse(page["url"])
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", parsed.netloc + parsed.path)

    chunks = []
    words_per_chunk = 360
    overlap = 60
    step = words_per_chunk - overlap
    idx = 0

    for start in range(0, len(words), step):
        part = words[start:start + words_per_chunk]

        if len(part) < 80:
            break

        text = " ".join(part)

        if not is_generic_noise_text(text):
            chunks.append(
                Chunk(
                    chunk_id=f"adaptive_{safe}_{idx:04d}",
                    text=text,
                    page_start=None,
                    page_end=None,
                    source_type="adaptive_url",
                    source_title=page["title"],
                    source_url=page["url"],
                )
            )
            idx += 1

    return chunks


def make_adaptive_retriever(args) -> BM25Retriever:
    urls = DEFAULT_ADAPTIVE_URLS
    cache_path = args.adaptive_cache_json

    cache_key = {
        "urls": urls,
        "adaptive_words_per_chunk": 360,
        "adaptive_overlap_words": 60,
    }

    if (
        not args.no_adaptive_cache
        and os.path.exists(cache_path)
        and not args.rebuild_source_cache
    ):
        try:
            cached = read_json(cache_path)

            if cached.get("cache_key") == cache_key and cached.get("chunks"):
                chunks = [
                    Chunk(
                        chunk_id=x["chunk_id"],
                        text=x["text"],
                        page_start=x.get("page_start"),
                        page_end=x.get("page_end"),
                        source_type=x["source_type"],
                        source_title=x["source_title"],
                        source_url=x.get("source_url"),
                    )
                    for x in cached["chunks"]
                ]

                print(f"Loaded adaptive cache: {cache_path}")
                print(f"Adaptive chunks: {len(chunks)}")

                return BM25Retriever(chunks)

        except Exception as e:
            print(f"Could not load adaptive cache. Rebuilding. Reason: {e}")

    all_chunks = []

    for url in urls:
        try:
            print(f"Fetching allowed adaptive source: {url}")
            page = fetch_html_text(url)
            chunks = build_web_chunks(page)
            print(f"Built {len(chunks)} chunks from: {page['title']}")
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Could not fetch {url}. Reason: {e}")

        time.sleep(0.2)

    write_json(
        cache_path,
        {
            "cache_key": cache_key,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "source_type": c.source_type,
                    "source_title": c.source_title,
                    "source_url": c.source_url,
                }
                for c in all_chunks
            ],
        },
    )

    print(f"Saved adaptive cache: {cache_path}")
    print(f"Adaptive chunks built: {len(all_chunks)}")

    return BM25Retriever(all_chunks)


# =========================================================
# Evidence selection
# =========================================================

def overlap_score(question: str, sentence: str) -> float:
    q = set(tokenize(question))
    s = set(tokenize(sentence))

    if not q or not s:
        return 0.0

    return len(q.intersection(s)) / max(1, len(q))


def extract_evidence(
    question: str,
    results: List[Dict[str, Any]],
    max_evidence: int,
    min_overlap: float,
) -> List[Dict[str, Any]]:

    candidates = []

    for r in results:
        for sent in split_sentences(r.get("text", "")):
            if is_generic_noise_text(sent):
                continue

            score = overlap_score(question, sent)

            if score >= min_overlap:
                candidates.append(
                    {
                        "sentence": sent,
                        "chunk_id": r.get("chunk_id"),
                        "source_type": r.get("source_type"),
                        "source_title": r.get("source_title"),
                        "source_url": r.get("source_url"),
                        "page_start": r.get("page_start"),
                        "page_end": r.get("page_end"),
                        "overlap_score": score,
                        "chunk_score": float(r.get("score", 0.0)),
                    }
                )

    candidates = sorted(
        candidates,
        key=lambda x: (x["overlap_score"], x["chunk_score"]),
        reverse=True,
    )

    selected = []
    seen = set()

    for ev in candidates:
        norm = re.sub(r"\W+", " ", ev["sentence"].lower()).strip()

        if norm in seen:
            continue

        seen.add(norm)
        selected.append(ev)

        if len(selected) >= max_evidence:
            break

    return selected


def evidence_good(evidence: List[Dict[str, Any]], top_score: float, min_top_score: float) -> bool:
    if top_score < min_top_score:
        return False

    if not evidence:
        return False

    avg_overlap = sum(x["overlap_score"] for x in evidence) / len(evidence)
    return avg_overlap >= 0.01


def citation(ev: Dict[str, Any]) -> str:
    if ev.get("source_type") == "adaptive_url":
        return f'{ev.get("source_title")} | {ev.get("source_url")}'

    if ev.get("page_start") is not None:
        return f'{ev.get("source_title")} | pages {ev.get("page_start")}-{ev.get("page_end")}'

    return f'{ev.get("source_title")} | {ev.get("source_url")}'


def evidence_block(evidence: List[Dict[str, Any]]) -> str:
    lines = []

    for i, ev in enumerate(evidence, start=1):
        lines.append(f"[E{i}] {citation(ev)}\n{ev['sentence']}")

    return "\n\n".join(lines)


# =========================================================
# Stronger prompt
# =========================================================

def build_prompt(question: str, evidence: List[Dict[str, Any]], source_name: str) -> str:
    return f"""
You are a senior 5G NR RAN engineer and 3GPP RRC specification analyst.

Your job is to answer technical questions carefully using ONLY the evidence provided below.
You are allowed to explain the concept in engineering language, but every technical claim must be grounded in the evidence.

Very important rules:
1. Use ONLY the evidence below.
2. Do NOT use outside knowledge, memory, assumptions, or general telecom knowledge.
3. If the evidence does not directly support the answer, say exactly: Not enough information.
4. Do NOT invent timers, messages, information elements, states, procedures, causes, conditions, or sequence steps.
5. Do NOT answer from general knowledge if the evidence is weak.
6. Do NOT mention hidden reasoning, judge scores, retrieval route, BM25, adaptive retrieval, or thresholds.
7. Do NOT copy unrelated table of contents, revision history, RP numbers, metadata, references, or webpage navigation text.
8. If the question asks for a procedure, explain the procedure only if the evidence gives enough procedural detail.
9. If the answer has conditions or multiple cases, organize it using bullets.
10. If the evidence only partially supports the answer, clearly limit the answer to what is supported.
11. Prefer a useful technical explanation, not a one-line answer, when evidence is sufficient.

Answer style:
- Start with a direct answer.
- Then provide a clear engineering explanation.
- Use 3 to 8 bullets if helpful.
- Keep it technical but readable.
- Include exact terms from the evidence when useful.
- Do not over-explain beyond the evidence.

Output exactly:

Answer:
<direct answer followed by a useful explanation. Use one or two short paragraphs plus bullets if needed.>

Evidence Used:
- E<number>: <briefly explain what this evidence supports>

Confidence:
High/Medium/Low

Question:
{question}

Evidence source:
{source_name}

Evidence:
{evidence_block(evidence)}
""".strip()


# =========================================================
# Local generator
# =========================================================

class LocalGenerator:
    def __init__(self, model_name: str, allow_cpu: bool = False, load_in_4bit: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu" and not allow_cpu:
            raise RuntimeError(
                "CUDA GPU not detected. Run this:\n"
                "python -c \"import torch; print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')\""
            )

        hf_token = os.environ.get("HF_TOKEN")

        print(f"Loading generator on {self.device}: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "token": hf_token,
        }

        if self.device == "cuda":
            kwargs["device_map"] = "auto"

            if load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        print("Generator loaded.")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior 5G NR RAN engineer. "
                    "Use only the provided evidence. "
                    "If unsupported, say Not enough information."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=6144,
        )

        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.06,
        }

        if temperature > 0:
            kwargs["temperature"] = temperature

        with self.torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        gen = out[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_answer(raw: str) -> str:
    raw = raw.strip()

    m = re.search(
        r"(?is)\bAnswer\s*:\s*(.+?)(\n\s*Evidence Used\s*:|\n\s*Confidence\s*:|$)",
        raw,
    )

    answer = m.group(1).strip() if m else raw

    if not answer:
        return "Not enough information"

    if is_bad_answer(answer):
        return "Not enough information"

    return answer


# =========================================================
# NLI judge
# =========================================================

class NLIJudge:
    def __init__(self, model_name: str, force_cpu: bool = False):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

        hf_token = os.environ.get("HF_TOKEN")

        print(f"Loading NLI judge on {self.device}: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            token=hf_token,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.entail_idx, self.neutral_idx, self.contradiction_idx = self.label_indices()

        print("NLI judge loaded.")

    def label_indices(self):
        entail = None
        neutral = None
        contradiction = None

        for idx, label in self.model.config.id2label.items():
            label = str(label).lower()

            if "entail" in label:
                entail = int(idx)
            elif "neutral" in label:
                neutral = int(idx)
            elif "contrad" in label:
                contradiction = int(idx)

        return (
            entail if entail is not None else 1,
            neutral if neutral is not None else 2,
            contradiction if contradiction is not None else 0,
        )

    def judge(
        self,
        answer: str,
        evidence: List[Dict[str, Any]],
        entailment_threshold: float,
        partial_threshold: float,
        max_contradiction: float,
        min_support_score: float,
    ) -> Dict[str, Any]:

        if not answer or "not enough information" in answer.lower():
            return {
                "verdict": "ABSTAINED",
                "reason": "Answer abstained.",
                "entailment_avg": 0.0,
                "contradiction_max": 0.0,
                "support_score": 0.0,
            }

        premise = normalize_text(" ".join(ev["sentence"] for ev in evidence))[:5000]

        claims = [
            c.strip()
            for c in re.split(r"(?<=[.!?;])\s+|\n+|•|- ", normalize_text(answer))
            if len(c.strip().split()) >= 5
        ]

        if not claims:
            claims = [answer]

        scores = []

        for claim in claims:
            enc = self.tokenizer(
                premise,
                claim,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            enc = {k: v.to(self.device) for k, v in enc.items()}

            with self.torch.inference_mode():
                logits = self.model(**enc).logits

            probs = self.torch.softmax(logits, dim=-1)[0]

            scores.append(
                {
                    "entailment": float(probs[self.entail_idx].detach().cpu()),
                    "neutral": float(probs[self.neutral_idx].detach().cpu()),
                    "contradiction": float(probs[self.contradiction_idx].detach().cpu()),
                    "claim": claim,
                }
            )

        entail_avg = sum(s["entailment"] for s in scores) / len(scores)
        contradiction_max = max(s["contradiction"] for s in scores)
        support_score = sum(
            1 for s in scores if s["entailment"] >= entailment_threshold
        ) / len(scores)

        if contradiction_max > max_contradiction:
            verdict = "UNSUPPORTED"
            reason = "High contradiction against evidence."
        elif entail_avg >= entailment_threshold or support_score >= min_support_score:
            verdict = "SUPPORTED"
            reason = "Answer is supported by evidence."
        elif entail_avg >= partial_threshold:
            verdict = "PARTIALLY_SUPPORTED"
            reason = "Answer is partially supported by evidence."
        else:
            verdict = "UNSUPPORTED"
            reason = "Insufficient evidence support."

        return {
            "verdict": verdict,
            "reason": reason,
            "entailment_avg": entail_avg,
            "contradiction_max": contradiction_max,
            "support_score": support_score,
            "claim_scores": scores,
        }


# =========================================================
# Main bot
# =========================================================

class SpecBot:
    def __init__(self, args):
        self.args = args

        print("\nStarting evidence-aware embedded-source 3GPP bot...")

        self.primary_retriever = load_or_build_primary_retriever(args)
        self.adaptive_retriever = make_adaptive_retriever(args)

        self.generator = LocalGenerator(
            model_name=args.generator_model_name,
            allow_cpu=args.allow_cpu,
            load_in_4bit=not args.no_4bit,
        )

        self.judge = None if args.skip_nli else NLIJudge(
            model_name=args.nli_model_name,
            force_cpu=args.nli_cpu,
        )

        print("\nBot ready.\n")

    def basic_result(self, route: str, answer: str, question: str, start: float) -> Dict[str, Any]:
        return {
            "route": route,
            "answer": answer,
            "question": question,
            "primary_verdict": "not used",
            "adaptive_verdict": "not used",
            "primary_top_score": None,
            "adaptive_top_score": None,
            "primary_entailment": None,
            "adaptive_entailment": None,
            "primary_support": None,
            "adaptive_support": None,
            "primary_contradiction": None,
            "adaptive_contradiction": None,
            "evidence": [],
            "runtime_sec": time.time() - start,
        }

    def run_generation_and_judge(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        source_name: str,
        stage: str,
    ):
        prompt = build_prompt(question, evidence, source_name)

        raw = self.generator.generate(
            prompt=prompt,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
        )

        answer = parse_answer(raw)

        if self.judge is None:
            judge_result = {
                "verdict": "SKIPPED",
                "reason": "NLI judge skipped.",
                "entailment_avg": None,
                "contradiction_max": None,
                "support_score": None,
            }
            return answer, judge_result

        if stage == "primary":
            judge_result = self.judge.judge(
                answer=answer,
                evidence=evidence,
                entailment_threshold=self.args.primary_entailment_threshold,
                partial_threshold=self.args.primary_partial_threshold,
                max_contradiction=self.args.primary_max_contradiction,
                min_support_score=self.args.primary_min_support_score,
            )
        else:
            judge_result = self.judge.judge(
                answer=answer,
                evidence=evidence,
                entailment_threshold=self.args.adaptive_entailment_threshold,
                partial_threshold=self.args.adaptive_partial_threshold,
                max_contradiction=self.args.adaptive_max_contradiction,
                min_support_score=self.args.adaptive_min_support_score,
            )

        return answer, judge_result

    def answer(self, question: str) -> Dict[str, Any]:
        start = time.time()
        question = normalize_text(question)

        if not question:
            return self.basic_result("empty", "Please enter a question.", question, start)

        primary_results = self.primary_retriever.search(
            query=question,
            top_k=self.args.top_k,
            prefetch_k=self.args.prefetch_k,
        )

        primary_top_score = primary_results[0]["score"] if primary_results else 0.0

        primary_evidence = extract_evidence(
            question=question,
            results=primary_results,
            max_evidence=self.args.max_evidence,
            min_overlap=self.args.min_evidence_overlap,
        )

        primary_judge = {
            "verdict": "NO_GOOD_PRIMARY_EVIDENCE",
            "reason": "Primary evidence was weak.",
            "entailment_avg": None,
            "support_score": None,
            "contradiction_max": None,
        }

        if evidence_good(primary_evidence, primary_top_score, self.args.min_top_score):
            primary_answer, primary_judge = self.run_generation_and_judge(
                question=question,
                evidence=primary_evidence,
                source_name="ETSI TS 138 331 primary PDF",
                stage="primary",
            )

            primary_ok = (
                primary_judge["verdict"] in ["SUPPORTED", "PARTIALLY_SUPPORTED", "SKIPPED"]
                and not is_bad_answer(primary_answer)
                and "not enough information" not in primary_answer.lower()
            )

            if primary_ok:
                return {
                    "route": "primary_pdf",
                    "answer": primary_answer,
                    "question": question,
                    "primary_verdict": primary_judge["verdict"],
                    "primary_reason": primary_judge["reason"],
                    "primary_entailment": primary_judge["entailment_avg"],
                    "primary_support": primary_judge["support_score"],
                    "primary_contradiction": primary_judge["contradiction_max"],
                    "adaptive_verdict": "not used",
                    "adaptive_reason": "not used",
                    "adaptive_entailment": None,
                    "adaptive_support": None,
                    "adaptive_contradiction": None,
                    "primary_top_score": primary_top_score,
                    "adaptive_top_score": None,
                    "evidence": primary_evidence,
                    "runtime_sec": time.time() - start,
                }

        adaptive_results = self.adaptive_retriever.search(
            query=question,
            top_k=self.args.adaptive_top_k,
            prefetch_k=self.args.prefetch_k,
        )

        adaptive_top_score = adaptive_results[0]["score"] if adaptive_results else 0.0

        adaptive_evidence = extract_evidence(
            question=question,
            results=adaptive_results,
            max_evidence=self.args.max_evidence,
            min_overlap=self.args.min_evidence_overlap,
        )

        if not evidence_good(
            adaptive_evidence,
            adaptive_top_score,
            max(0.25, self.args.min_top_score * 0.5),
        ):
            return {
                "route": "failed_evidence",
                "answer": "Not enough information",
                "question": question,
                "primary_verdict": primary_judge["verdict"],
                "primary_reason": primary_judge["reason"],
                "adaptive_verdict": "NO_GOOD_ADAPTIVE_EVIDENCE",
                "adaptive_reason": "Adaptive evidence was also weak.",
                "primary_top_score": primary_top_score,
                "adaptive_top_score": adaptive_top_score,
                "primary_entailment": primary_judge["entailment_avg"],
                "primary_support": primary_judge["support_score"],
                "primary_contradiction": primary_judge["contradiction_max"],
                "adaptive_entailment": None,
                "adaptive_support": None,
                "adaptive_contradiction": None,
                "evidence": primary_evidence or adaptive_evidence,
                "runtime_sec": time.time() - start,
            }

        adaptive_answer, adaptive_judge = self.run_generation_and_judge(
            question=question,
            evidence=adaptive_evidence,
            source_name="embedded allowed 3GPP/RRC/5G sources",
            stage="adaptive",
        )

        adaptive_ok = (
            adaptive_judge["verdict"] in ["SUPPORTED", "PARTIALLY_SUPPORTED", "SKIPPED"]
            and not is_bad_answer(adaptive_answer)
            and "not enough information" not in adaptive_answer.lower()
        )

        final_answer = adaptive_answer if adaptive_ok else "Not enough information"

        return {
            "route": "adaptive_embedded_sources" if adaptive_ok else "failed_both",
            "answer": final_answer,
            "question": question,
            "primary_verdict": primary_judge["verdict"],
            "primary_reason": primary_judge["reason"],
            "adaptive_verdict": adaptive_judge["verdict"],
            "adaptive_reason": adaptive_judge["reason"],
            "adaptive_entailment": adaptive_judge["entailment_avg"],
            "adaptive_support": adaptive_judge["support_score"],
            "adaptive_contradiction": adaptive_judge["contradiction_max"],
            "primary_top_score": primary_top_score,
            "adaptive_top_score": adaptive_top_score,
            "primary_entailment": primary_judge["entailment_avg"],
            "primary_support": primary_judge["support_score"],
            "primary_contradiction": primary_judge["contradiction_max"],
            "evidence": adaptive_evidence,
            "runtime_sec": time.time() - start,
        }


# =========================================================
# UI
# =========================================================

def summary_html(result: Dict[str, Any]) -> str:
    return f"""
    <div class="metrics">
        <div class="metric runtime-only">
            <span>Runtime</span>
            <b>{fmt(result.get("runtime_sec"))}s</b>
        </div>
    </div>
    """


def pipeline_md(result: Dict[str, Any]) -> str:
    return f"""
### Pipeline Details

- Question: `{result.get("question", "")}`
- Route: `{result.get("route", "N/A")}`
- Primary verdict: `{result.get("primary_verdict", "N/A")}`
- Adaptive verdict: `{result.get("adaptive_verdict", "N/A")}`
- Primary entailment: `{fmt(result.get("primary_entailment"))}`
- Adaptive entailment: `{fmt(result.get("adaptive_entailment"))}`
- Primary support: `{fmt(result.get("primary_support"))}`
- Adaptive support: `{fmt(result.get("adaptive_support"))}`
- Primary contradiction: `{fmt(result.get("primary_contradiction"))}`
- Adaptive contradiction: `{fmt(result.get("adaptive_contradiction"))}`
"""


def evidence_md(result: Dict[str, Any]) -> str:
    evidence = result.get("evidence", [])

    if not evidence:
        return "### Evidence\n\nNo evidence selected."

    lines = ["### Evidence Used"]

    for i, ev in enumerate(evidence, start=1):
        lines.append(f"\n**E{i}.** {ev['sentence']}")
        lines.append(f"- Source: `{citation(ev)}`")
        lines.append(f"- Overlap: `{fmt(ev.get('overlap_score'))}`")
        lines.append(f"- Chunk score: `{fmt(ev.get('chunk_score'))}`")

    return "\n".join(lines)


def launch_ui(bot: SpecBot, args):
    import gradio as gr

    counter = {"value": 1}

    def ask(question):
        try:
            result = bot.answer(question)

            if args.save_each_result:
                os.makedirs(args.out_dir, exist_ok=True)
                out_path = os.path.join(args.out_dir, f"result_{counter['value']:03d}.json")
                write_json(out_path, result)
                counter["value"] += 1

            return (
                f"### Final Answer\n\n{result.get('answer', 'Not enough information')}",
                summary_html(result),
                pipeline_md(result),
                evidence_md(result),
            )

        except Exception as e:
            err = {
                "route": "error",
                "runtime_sec": 0,
                "primary_verdict": "error",
                "adaptive_verdict": "error",
                "primary_top_score": None,
                "adaptive_top_score": None,
                "evidence": [],
            }

            return (
                f"### Final Answer\n\n```text\n{str(e)}\n```",
                summary_html(err),
                "### Pipeline Details\n\nThe run failed. Read the terminal error.",
                "### Evidence\n\nNo evidence because the run failed.",
            )

    def clear():
        return "", "### Final Answer", "<div class='empty'>No run yet.</div>", "", ""

    css = """
    .gradio-container {
        max-width: 1180px !important;
        margin: auto !important;
    }

    .top-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
    }

    .top-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
        color: #0f172a;
    }

    .top-subtitle {
        color: #475569;
        line-height: 1.5;
        font-size: 15px;
    }

    .chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 14px;
    }

    .chip {
        padding: 6px 10px;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 13px;
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
        gap: 10px;
    }

    .metric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 12px;
    }

    .metric span {
        display: block;
        font-size: 12px;
        color: #64748b;
        margin-bottom: 5px;
    }

    .metric b {
        color: #0f172a;
        font-size: 14px;
        word-break: break-word;
    }

    .empty {
        border: 1px dashed #cbd5e1;
        color: #64748b;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
    }
    """

    with gr.Blocks(
        title="Evidence-Aware Spec Bot",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:

        gr.HTML(
            """
            <div class="top-card">
                <div class="top-title">Evidence-Aware Spec Bot</div>
                <div class="top-subtitle">
                    The system answers only from the embedded TS 38.331 PDF and approved 3GPP/RRC/5G sources.
                    It retrieves evidence, generates a grounded answer, verifies support with an NLI judge,
                    and uses adaptive retrieval only when primary PDF support is weak.
                </div>
                <div class="chips">
                    <div class="chip">TS 38.331 PDF</div>
                    <div class="chip">Embedded 5G/RRC Sources</div>
                    <div class="chip">Evidence Selection</div>
                    <div class="chip">Qwen Generator</div>
                    <div class="chip">NLI Judge</div>
                    <div class="chip">Adaptive Retrieval</div>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                question = gr.Textbox(
                    label="Ask a question",
                    placeholder="Type your 3GPP/RAN/RRC question here...",
                    lines=4,
                )

                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=5):
                gr.Markdown("### Run Summary")
                summary = gr.HTML("<div class='empty'>No run yet.</div>")

        with gr.Tabs():
            with gr.Tab("Answer"):
                answer = gr.Markdown("### Final Answer")

            with gr.Tab("Pipeline"):
                pipeline = gr.Markdown("")

            with gr.Tab("Evidence"):
                evidence = gr.Markdown("")

        ask_btn.click(ask, inputs=question, outputs=[answer, summary, pipeline, evidence])
        question.submit(ask, inputs=question, outputs=[answer, summary, pipeline, evidence])
        clear_btn.click(clear, outputs=[question, answer, summary, pipeline, evidence])

    demo.queue(default_concurrency_limit=1)

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )


# =========================================================
# CLI
# =========================================================

def run_cli(bot: SpecBot):
    print("\nCLI ready. Type exit to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        result = bot.answer(question)

        print("\nAnswer:")
        print(result.get("answer", "Not enough information"))
        print("\nRuntime:", fmt(result.get("runtime_sec")), "s")
        print("Route:", result.get("route"))
        print("Primary:", result.get("primary_verdict"))
        print("Adaptive:", result.get("adaptive_verdict"))


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf_path", type=str, default=None)

    parser.add_argument("--generator_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--nli_model_name", type=str, default="cross-encoder/nli-deberta-v3-base")

    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--nli_cpu", action="store_true")
    parser.add_argument("--skip_nli", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")

    parser.add_argument("--words_per_chunk", type=int, default=420)
    parser.add_argument("--overlap_words", type=int, default=70)

    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--adaptive_top_k", type=int, default=8)
    parser.add_argument("--prefetch_k", type=int, default=120)

    parser.add_argument("--min_top_score", type=float, default=1.0)
    parser.add_argument("--max_evidence", type=int, default=18)
    parser.add_argument("--min_evidence_overlap", type=float, default=0.01)

    parser.add_argument("--max_new_tokens", type=int, default=650)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--primary_entailment_threshold", type=float, default=0.50)
    parser.add_argument("--primary_partial_threshold", type=float, default=0.25)
    parser.add_argument("--primary_max_contradiction", type=float, default=0.75)
    parser.add_argument("--primary_min_support_score", type=float, default=0.34)

    parser.add_argument("--adaptive_entailment_threshold", type=float, default=0.35)
    parser.add_argument("--adaptive_partial_threshold", type=float, default=0.20)
    parser.add_argument("--adaptive_max_contradiction", type=float, default=0.80)
    parser.add_argument("--adaptive_min_support_score", type=float, default=0.20)

    parser.add_argument("--source_cache_dir", type=str, default="source_cache")
    parser.add_argument("--primary_cache_json", type=str, default="primary_ts38331_cache.json")
    parser.add_argument("--adaptive_cache_json", type=str, default="adaptive_embedded_sources_cache.json")
    parser.add_argument("--use_primary_cache", action="store_true")
    parser.add_argument("--no_adaptive_cache", action="store_true")
    parser.add_argument("--rebuild_source_cache", action="store_true")

    parser.add_argument("--save_each_result", action="store_true")
    parser.add_argument("--out_dir", type=str, default="interactive_outputs")

    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()

    bot = SpecBot(args)

    if args.cli:
        run_cli(bot)
    else:
        launch_ui(bot, args)


if __name__ == "__main__":
    main()