#!/usr/bin/env python3
"""
normalize_metadata.py
Reads PDFs, fetches Crossref metadata via DOI, outputs normalized JSON for vector DB ingestion.
"""

import re
import json
from pathlib import Path
import requests
from PyPDF2 import PdfReader

PDF_FOLDER = Path("./pdfs")
OUTPUT_FILE = Path("metadata_normalized.json")
EXTRACT_PAGES = 2
DOI_REGEX = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"

def extract_pdf_text(pdf_path, pages=EXTRACT_PAGES):
    reader = PdfReader(str(pdf_path))
    text_pieces = []
    for page in reader.pages[:pages]:
        page_text = page.extract_text() or ""
        text_pieces.append(page_text)
    return "\n".join(text_pieces)

def extract_doi(text):
    m = re.search(DOI_REGEX, text, re.I)
    return m.group(0).strip().lower() if m else None

def fetch_metadata(doi):
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"Accept": "application/json", "User-Agent": "NormalizeScript/1.0 (mailto:you@example.com)"}
    r = requests.get(url, headers=headers, timeout=12)
    if r.status_code != 200:
        return None
    data = r.json().get("message", {})
    # Extract year robustly
    year = None
    for key in ("published-print", "published-online", "created", "issued"):
        if key in data:
            parts = data[key].get("date-parts", [[None]])
            year = parts[0][0]
            if year:
                break
    return {
        "title": (data.get("title") or [""])[0],
        "authors": [f"{a.get('given','').strip()} {a.get('family','').strip()}".strip()
                    for a in data.get("author", [])] if data.get("author") else [],
        "year": year,
        "journal": (data.get("container-title") or [""])[0],
        "doi": data.get("DOI"),
        "abstract": data.get("abstract", "")
    }

def slugify(text):
    import re
    return re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-')

def main():
    pdf_files = sorted([p for p in PDF_FOLDER.iterdir() if p.suffix.lower() == ".pdf"])
    normalized = []

    for pdf in pdf_files:
        text = extract_pdf_text(pdf)
        doi = extract_doi(text)
        if not doi:
            print(f"⚠️ DOI not found in {pdf.name}, skipping")
            continue
        meta = fetch_metadata(doi)
        if not meta:
            print(f"⚠️ Could not fetch metadata for DOI {doi}, skipping")
            continue
        normalized.append({
            "id": slugify(meta.get("title", pdf.stem)),
            "title": meta.get("title", ""),
            "authors": meta.get("authors", []),
            "year": meta.get("year"),
            "journal": meta.get("journal", ""),
            "doi": meta.get("doi", ""),
            "abstract": meta.get("abstract", ""),
            "source_pdf": str(pdf)
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=4, ensure_ascii=False)

    print(f"✅ Normalized metadata saved to {OUTPUT_FILE}, {len(normalized)} papers processed.")

if __name__ == "__main__":
    main()
