import argparse
import os
import re
import time
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

from config import DEFAULT_ADAPTIVE_URLS
from retriever import Chunk, BM25Retriever, extract_pdf_pages_from_bytes
from utils import normalize_text, is_generic_noise_text, read_json, write_json


def load_urls(path="datasets/adaptive_urls.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if urls:
            return urls
    return DEFAULT_ADAPTIVE_URLS


def fetch_source_text(url: str, timeout: int = 30):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 Evidence-Aware-RAG/1.0"}, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        pages = extract_pdf_pages_from_bytes(response.content, source_title=url)
        return {"url": url, "title": "ETSI TS 138 331 PDF", "text": " ".join(p["text"] for p in pages)}
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "footer", "nav"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = normalize_text(soup.get_text(" ", strip=True))
    return {"url": url, "title": title, "text": text}


def build_adaptive_chunks(page, words_per_chunk=360, overlap_words=60):
    words = page["text"].split()
    if len(words) < 80:
        return []
    parsed = urlparse(page["url"])
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", parsed.netloc + parsed.path)
    chunks = []
    step = max(1, words_per_chunk - overlap_words)
    idx = 0
    for start in range(0, len(words), step):
        part = words[start:start + words_per_chunk]
        if len(part) < 80:
            break
        text = " ".join(part)
        if not is_generic_noise_text(text):
            chunks.append(Chunk(
                chunk_id=f"adaptive_{safe}_{idx:04d}", text=text,
                page_start=None, page_end=None, source_type="adaptive_url",
                source_title=page["title"], source_url=page["url"],
            ))
            idx += 1
    return chunks


def build_adaptive_retriever(urls_path="datasets/adaptive_urls.txt", cache_json="adaptive_embedded_sources_cache.json", use_cache=True, rebuild=False):
    urls = load_urls(urls_path)
    if use_cache and os.path.exists(cache_json) and not rebuild:
        cached = read_json(cache_json)
        chunks = [Chunk(**x) for x in cached.get("chunks", [])]
        if chunks:
            return BM25Retriever(chunks)
    all_chunks = []
    for url in urls:
        try:
            print(f"Fetching adaptive source: {url}")
            page = fetch_source_text(url)
            chunks = build_adaptive_chunks(page)
            print(f"Built {len(chunks)} chunks from {page['title']}")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Could not fetch {url}: {e}")
        time.sleep(0.2)
    write_json(cache_json, {"urls": urls, "chunks": [c.__dict__ for c in all_chunks]})
    return BM25Retriever(all_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls_path", default="datasets/adaptive_urls.txt")
    parser.add_argument("--cache_json", default="adaptive_embedded_sources_cache.json")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    retriever = build_adaptive_retriever(args.urls_path, args.cache_json, rebuild=args.rebuild)
    print(f"Adaptive chunks ready: {len(retriever.chunks)}")
