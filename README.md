# Evidence-Aware Hallucination Reduction for 3GPP RAN

This repository contains the code for an evidence-aware RAG system for 3GPP RAN/RRC question answering. The system retrieves evidence from **3GPP TS 38.331**, generates an answer using Qwen, verifies the answer with an NLI judge, and uses adaptive retrieval over a fixed list of selected 5G/RRC sources only when the primary evidence is weak.

## Project structure

```text
full_pipeline.py                 # Main end-to-end Gradio/CLI pipeline
text_extraction.py               # Extracts PDF text, tables, and metadata
retriever.py                     # Primary BM25 retriever and evidence selection
adaptive_retrieval.py            # Adaptive retrieval over selected 5G/RRC URLs
generator.py                     # Qwen generator and prompt construction
judge.py                         # NLI judge using cross-encoder/nli-deberta-v3-base
config.py                        # Default primary PDF URL and adaptive URL list
utils.py                         # Shared helpers, normalization, tokenization, filters
requirements.txt                 # Python dependencies
datasets/
  TS_38_331.pdf                  # Primary 3GPP TS 38.331 PDF
  adaptive_urls.txt              # 10 selected URLs used for adaptive retrieval
outputs/                         # Optional output directory
source_cache/                    # Optional downloaded/cache files
```

## Dataset and adaptive sources

The primary dataset is the local PDF:

```text
datasets/TS_38_331.pdf
```

The adaptive retrieval URLs are stored in:

```text
datasets/adaptive_urls.txt
```

These adaptive URLs are fixed and controlled. The system does **not** perform open web search during answering.

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Optional: extract PDF text first

This creates JSON and plain text versions of the PDF content.

```bash
python text_extraction.py \
  --pdf_path datasets/TS_38_331.pdf \
  --output_json datasets/extracted_data.json \
  --output_txt datasets/extracted_text.txt
```

## Run the full Gradio pipeline

This is the main command for the final demo. It uses the local TS 38.331 PDF, BM25 retrieval, Qwen generation, NLI judging, and adaptive retrieval.

```bash
python full_pipeline.py \
  --pdf_path datasets/TS_38_331.pdf \
  --generator_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --nli_model_name cross-encoder/nli-deberta-v3-base \
  --words_per_chunk 550 \
  --overlap_words 80 \
  --top_k 8 \
  --adaptive_top_k 8 \
  --prefetch_k 120 \
  --max_evidence 18 \
  --min_evidence_overlap 0.01 \
  --max_new_tokens 450 \
  --temperature 0.0 \
  --primary_entailment_threshold 0.50 \
  --primary_partial_threshold 0.25 \
  --primary_max_contradiction 0.75 \
  --primary_min_support_score 0.34 \
  --adaptive_entailment_threshold 0.35 \
  --adaptive_partial_threshold 0.20 \
  --adaptive_max_contradiction 0.80 \
  --adaptive_min_support_score 0.20 \
  --use_primary_cache \
  --save_each_result \
  --server_name 127.0.0.1 \
  --server_port 7862
```

Then open:

```text
http://127.0.0.1:7862
```

If you are running on a remote server and need a public temporary Gradio link, add:

```bash
--share
```

If CUDA is not available and you only want to test on CPU, add:

```bash
--allow_cpu --nli_cpu
```

CPU mode will be much slower.

## Run in CLI mode

```bash
python full_pipeline.py \
  --pdf_path datasets/TS_38_331.pdf \
  --generator_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --nli_model_name cross-encoder/nli-deberta-v3-base \
  --words_per_chunk 550 \
  --overlap_words 80 \
  --top_k 8 \
  --adaptive_top_k 8 \
  --prefetch_k 120 \
  --max_evidence 18 \
  --min_evidence_overlap 0.01 \
  --max_new_tokens 450 \
  --temperature 0.0 \
  --use_primary_cache \
  --cli
```

## How the pipeline works

1. The user enters a 3GPP RAN/RRC question.
2. The primary retriever searches the local TS 38.331 PDF using BM25.
3. The system keeps the top retrieved chunks and selects up to 18 evidence sentences.
4. The generator creates a draft answer using only the selected evidence.
5. The NLI judge checks whether the generated answer is supported by evidence.
6. If the primary answer is supported, it is returned.
7. If the primary evidence is weak, adaptive retrieval searches the fixed 10 URLs.
8. The generator and judge run again on adaptive evidence.
9. If the adaptive answer is still unsupported, the system returns `Not enough information.`

## Important parameters

| Parameter | Meaning |
|---|---|
| `--words_per_chunk 550` | Primary PDF chunk size |
| `--overlap_words 80` | Primary PDF overlap |
| `--prefetch_k 120` | Initial BM25 candidate pool |
| `--top_k 8` | Final primary retrieved chunks |
| `--adaptive_top_k 8` | Final adaptive retrieved chunks |
| `--max_evidence 18` | Max selected evidence sentences |
| `--min_evidence_overlap 0.01` | Minimum lexical overlap for evidence sentence selection |
| `--max_new_tokens 450` | Max generation length |
| `--temperature 0.0` | Deterministic generation |

## Notes

- The repository keeps `full_pipeline.py` as the main runnable file.
- The separate files are included to make the project easier to explain in the report and on GitHub.
- The adaptive retrieval source list is fixed in `datasets/adaptive_urls.txt` and also mirrored in `config.py`.
- The Gradio interface displays the final answer, run summary, pipeline route, and evidence used.
