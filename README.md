Repository for scripts and notes used in the literature selection and PDF enrichment workflow for my master thesis.

Overview
--------
This repository contains scripts and documentation used to extract DOI-based metadata from collected PDFs, enrich those PDFs with structured metadata (title, authors, year, abstract, journal reference), and index the enriched documents in a vector database (Pinecone) using OpenAI embeddings for semantic retrieval. The enrichment pipeline was used exclusively to support literature selection and the review process for the thesis; it is not part of the prototype presented in Chapter 3.

Key components
--------------
- PDF metadata enrichment script (extracts DOI-based metadata via the Crossref API)
- Metadata schema and examples (JSON)
- Indexing pipeline that creates OpenAI embeddings and pushes vectors to Pinecone for semantic search
- Utilities for APA-style citation formatting during retrieval

Why this was done
-----------------
Using DOI-based metadata and embeddings allowed:
- Faster, more accurate retrieval of thematically relevant literature
- Semantic search across abstracts, paragraphs, and pages
- On-the-fly APA-style citation formatting during retrieval to support writing and review

Important note
--------------
The enrichment and indexing scripts were used only as a literature selection aid. They did not form part of the prototype implemented in Chapter 3 of the thesis.

Quick start
-----------
1. Clone the repository:
   git clone https://github.com/nienalt1/literature_review.git

2. Install dependencies (example, adjust to the script's requirements):
   pip install -r requirements.txt

3. Configure environment variables (see docs/pdf-enrichment.md for details):
   - OPENAI_API_KEY
   - PINECONE_API_KEY
   - PINECONE_ENV
   - (optional) EMAIL_CONTACT for Crossref polite usage

4. Run enrichment (this produces the enriched metadata sidecar files first):
   python scripts/enrich_pdfs.py --input ./pdfs --output ./enriched/metadata

   - Output: enriched/metadata/*.json (DOI-based metadata, abstract, authors, timestamp)
   - Optional: sidecar JSONs or embedded XMP inside PDFs (see script flags)

5. Index vectors in Pinecone (uses enriched metadata + text chunks for embeddings):
   python scripts/index_with_pinecone.py --metadata-dir ./enriched/metadata



Contact
-------
Repository owner: nienalt1
