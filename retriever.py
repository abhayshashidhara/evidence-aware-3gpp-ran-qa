from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List, Dict, Any
import os
import requests
from rank_bm25 import BM25Okapi
from pypdf import PdfReader

from config import DEFAULT_PRIMARY_PDF_URL
from utils import normalize_text, tokenize, split_sentences, is_generic_noise_text, safe_filename_from_url, read_json, write_json


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    source_type: str
    source_title: str
    source_url: Optional[str]


def extract_pdf_pages_from_bytes(pdf_bytes: bytes, source_title: str) -> List[Dict[str, Any]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = normalize_text(text)
        if text:
            pages.append({"page": i, "text": text, "source_title": source_title})
    if not pages:
        raise ValueError("No extractable text found in PDF.")
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
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 Evidence-Aware-RAG/1.0"}, timeout=60)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path


def build_chunks_from_pages(pages, source_title, source_type, source_url, words_per_chunk=550, overlap_words=80, min_words=80):
    stream = []
    for page in pages:
        page_num = int(page.get("page", 0))
        text = normalize_text(page.get("text", ""))
        if len(text.split()) < 20 or is_generic_noise_text(text):
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
            chunks.append(Chunk(
                chunk_id=f"{source_type}_{idx:06d}", text=text,
                page_start=min(page_nums), page_end=max(page_nums),
                source_type=source_type, source_title=source_title, source_url=source_url,
            ))
            idx += 1
        start += step
    return chunks


class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokenized = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if chunks else None

    def search(self, query: str, top_k: int = 8, prefetch_k: int = 120):
        if not self.chunks or self.bm25 is None:
            return []
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:prefetch_k]
        out = []
        for i in ranked:
            c = self.chunks[i]
            if is_generic_noise_text(c.text):
                continue
            out.append({
                "rank": len(out) + 1, "chunk_id": c.chunk_id, "text": c.text,
                "score": float(scores[i]), "source_type": c.source_type,
                "source_title": c.source_title, "source_url": c.source_url,
                "page_start": c.page_start, "page_end": c.page_end,
            })
            if len(out) >= top_k:
                break
        return out


def build_primary_retriever(pdf_path="datasets/TS_38_331.pdf", cache_json="primary_ts38331_cache.json", use_cache=True, rebuild=False, words_per_chunk=550, overlap_words=80, source_cache_dir="source_cache"):
    if use_cache and os.path.exists(cache_json) and not rebuild:
        cached = read_json(cache_json)
        chunks = [Chunk(**x) for x in cached.get("chunks", [])]
        if chunks:
            return BM25Retriever(chunks)
    if pdf_path and os.path.exists(pdf_path):
        pages = extract_pdf_pages_from_file(pdf_path)
        title = os.path.basename(pdf_path)
        url = pdf_path
    else:
        downloaded = download_pdf_to_cache(DEFAULT_PRIMARY_PDF_URL, source_cache_dir)
        pages = extract_pdf_pages_from_file(downloaded)
        title = "ETSI TS 138 331 v18.1.0 PDF"
        url = DEFAULT_PRIMARY_PDF_URL
    chunks = build_chunks_from_pages(pages, title, "primary_pdf", url, words_per_chunk, overlap_words)
    write_json(cache_json, {"chunks": [c.__dict__ for c in chunks]})
    return BM25Retriever(chunks)


def overlap_score(question: str, sentence: str) -> float:
    q = set(tokenize(question))
    s = set(tokenize(sentence))
    if not q or not s:
        return 0.0
    return len(q.intersection(s)) / max(1, len(q))


def extract_evidence(question: str, results: List[Dict[str, Any]], max_evidence=18, min_overlap=0.01):
    candidates = []
    for r in results:
        for sent in split_sentences(r.get("text", "")):
            if is_generic_noise_text(sent):
                continue
            score = overlap_score(question, sent)
            if score >= min_overlap:
                candidates.append({
                    "sentence": sent,
                    "overlap_score": score,
                    "chunk_score": r.get("score", 0.0),
                    "source_title": r.get("source_title"),
                    "source_url": r.get("source_url"),
                    "page_start": r.get("page_start"),
                    "page_end": r.get("page_end"),
                })
    candidates.sort(key=lambda x: (x["overlap_score"], x["chunk_score"]), reverse=True)
    return candidates[:max_evidence]
