#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def extract_with_pdfplumber(pdf_path: Path):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages, start=1):
            pages.append({
                "page": i,
                "width": page.width,
                "height": page.height,
                "text": page.extract_text() or "",
                "tables": page.extract_tables() or [],
            })
        return {"method": "pdfplumber", "metadata": {"num_pages": len(pdf.pages), "metadata": pdf.metadata}, "pages": pages}


def extract_with_pymupdf(pdf_path: Path):
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pages.append({
            "page": page_num + 1,
            "text": page.get_text() or "",
            "num_images": len(page.get_images()),
            "blocks": len(page.get_text("dict").get("blocks", [])),
        })
    metadata = {"num_pages": doc.page_count, "metadata": doc.metadata}
    doc.close()
    return {"method": "PyMuPDF", "metadata": metadata, "pages": pages}


def extract_with_pypdf2(pdf_path: Path):
    import PyPDF2
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append({"page": i, "text": page.extract_text() or ""})
        return {"method": "PyPDF2", "metadata": {"num_pages": len(reader.pages), "metadata": dict(reader.metadata) if reader.metadata else {}}, "pages": pages}


def save_text(data, output_txt: Path):
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"PDF Extraction - {data['method']}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of pages: {data['metadata'].get('num_pages', 'N/A')}\n\n")
        for page in data.get("pages", []):
            f.write(f"\n--- Page {page['page']} ---\n\n")
            f.write(page.get("text", ""))
            f.write("\n")
            for idx, table in enumerate(page.get("tables", []) or [], start=1):
                f.write(f"\nTable {idx}:\n")
                for row in table:
                    f.write(" | ".join(str(cell) if cell else "" for cell in row) + "\n")


def extract_pdf(pdf_path: str, output_json: str, output_txt: str):
    pdf_path = Path(pdf_path)
    methods = [extract_with_pdfplumber, extract_with_pymupdf, extract_with_pypdf2]
    last_error = None
    for method in methods:
        try:
            data = method(pdf_path)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            save_text(data, Path(output_txt))
            print(f"Extracted with {data['method']}")
            print(f"Saved JSON: {output_json}")
            print(f"Saved text: {output_txt}")
            return data
        except Exception as e:
            last_error = e
            print(f"{method.__name__} failed: {e}")
    raise RuntimeError(f"PDF extraction failed. Last error: {last_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", default="datasets/TS_38_331.pdf")
    parser.add_argument("--output_json", default="datasets/extracted_data.json")
    parser.add_argument("--output_txt", default="datasets/extracted_text.txt")
    args = parser.parse_args()
    extract_pdf(args.pdf_path, args.output_json, args.output_txt)
