#!/usr/bin/env python3
"""
enrich_pdfs.py
Adds normalized metadata (title, author, DOI, journal, year) to PDF files.
Renames enriched PDFs to match their title.
Uses metadata_normalized.json (from normalize_metadata.py).
Saves enriched PDFs into ./pdfs_enriched/
"""

import json
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter

# Paths
PDF_FOLDER = Path("./pdfs")
OUTPUT_FOLDER = Path("./pdfs_enriched")
METADATA_FILE = Path("metadata_normalized.json")  # use normalized JSON

def safe_filename(name):
    """Return a filesystem-safe filename from a title."""
    return "".join(c for c in name if c.isalnum() or c in " _-").strip()

def enrich_pdf(pdf_path, metadata, output_folder):
    """Write normalized metadata into a new PDF file and rename it."""
    try:
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()

        # Copy all pages
        for page in reader.pages:
            writer.add_page(page)

        # Build normalized metadata
        meta = {
            "/Title": metadata.get("title", pdf_path.stem),
            "/Author": ", ".join(metadata.get("authors", [])),
            "/Subject": metadata.get("journal", ""),
            "/Keywords": metadata.get("doi", ""),
            "/Producer": "PyPDF2",
            "/Creator": "Normalized PDF Metadata Script",
        }

        # Add year if available
        year = metadata.get("year")
        if year:
            meta["/CreationDate"] = f"D:{year}0101000000Z"

        writer.add_metadata(meta)

        # Save PDF with filename = title
        out_name = f"{safe_filename(metadata.get('title', pdf_path.stem))}.pdf"
        output_path = output_folder / out_name
        with open(output_path, "wb") as f:
            writer.write(f)

        print(f"✅ Enriched: {pdf_path.name} → {out_name}")

    except Exception as e:
        print(f"❌ Error enriching {pdf_path.name}: {e}")

def main():
    if not METADATA_FILE.exists():
        print(f"⚠️ Metadata file not found: {METADATA_FILE}")
        return

    if not PDF_FOLDER.exists():
        print(f"⚠️ PDF folder not found: {PDF_FOLDER}")
        return

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_entries = json.load(f)

    for entry in metadata_entries:
        pdf_path = Path(entry.get("source_pdf", ""))
        if not pdf_path.exists():
            print(f"⚠️ Missing PDF: {pdf_path}")
            continue
        enrich_pdf(pdf_path, entry, OUTPUT_FOLDER)

    print("\n🎉 Done! Enriched PDFs are saved in ./pdfs_enriched/")

if __name__ == "__main__":
    main()
