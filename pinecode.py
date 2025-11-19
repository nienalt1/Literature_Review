#!/usr/bin/env python3
"""
Upload enriched PDFs to Pinecone using OpenAI embeddings and provide a small
retrieval helper that formats APA 7th in-text citations (Author, Year, p. X).

If Pinecone can't be initialized, the script falls back to a local JSONL
store (local_vectors.jsonl) and local similarity search so you can test end-to-end.
"""
import os
import re
import json
import time
import math
import importlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Load secrets
# ----------------------
load_dotenv()

OPENAI_API_KEY = ""
PINECONE_API_KEY = ""
PINECONE_ENV = "us-east-1-aws"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env")

# ----------------------
# Config
# ----------------------
PDF_FOLDER = Path("./pdfs_enriched")
METADATA_FILE = Path("metadata_normalized.json")
LOCAL_VECTORS_FILE = Path("local_vectors.jsonl")

PINECONE_INDEX_NAME = "literature-review"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 10
EMBED_RETRIES = 3
UPSERT_RETRIES = 3
MAX_META_EXCERPT = 800
ABSTRACT_SEARCH_PAGES = 3

# ----------------------
# Initialize OpenAI (support both old and new SDKs)
# ----------------------
openai_client = None
OPENAI_V2 = False

try:
    # new-style client (openai>=1.0)
    from openai import OpenAI as OpenAIClient  # type: ignore
    openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
    OPENAI_V2 = True
    logger.info("Using OpenAI v1+ client (openai.OpenAI)")
except Exception:
    # fallback to legacy openai module interface
    openai.api_key = OPENAI_API_KEY
    OPENAI_V2 = False
    logger.info("Using legacy openai module interface (openai.Embedding.create if available)")

# ----------------------
# Pinecone / Local fallback
# ----------------------
index = None
_pc = None
_pinecone_module = None
LOCAL_MODE = False
_local_buffer: List[Dict[str, Any]] = []

class _IndexProxy:
    """Adapter for pinecone.PineconeClient.indexes to expose upsert/query like legacy Index."""
    def __init__(self, client_obj, name: str):
        self._client = client_obj
        self._name = name

    def upsert(self, vectors=None):
        try:
            return self._client.indexes.upsert(index=self._name, vectors=vectors)
        except TypeError:
            return self._client.indexes.upsert(vectors=vectors, index=self._name)

    def query(self, vector=None, top_k=10, include_metadata=True):
        try:
            return self._client.indexes.query(index=self._name, vector=vector, top_k=top_k, include_metadata=include_metadata)
        except TypeError:
            return self._client.indexes.query(vector=vector, top_k=top_k, include_metadata=include_metadata, index=self._name)

def _make_index_from_client(client_obj):
    """Try common ways to obtain an Index object from a client-style object."""
    try:
        return client_obj.Index(PINECONE_INDEX_NAME)
    except Exception:
        pass
    try:
        return client_obj.index(PINECONE_INDEX_NAME)
    except Exception:
        pass
    try:
        getter = getattr(client_obj, "index_client", None)
        if callable(getter):
            return getter(PINECONE_INDEX_NAME)
    except Exception:
        pass
    return None

def init_pinecone_or_fallback():
    """
    Try to initialize Pinecone for v7+ (PineconeClient with .indexes) first,
    then fall back to legacy module-style API. If anything fails, enable LOCAL_MODE.
    """
    global index, _pc, _pinecone_module, LOCAL_MODE
    import importlib
    _pinecone_module = None
    _pc = None
    index = None

    if not PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY not set — using local mode.")
        LOCAL_MODE = True
        _load_local_vectors()
        return

    try:
        _p = importlib.import_module("pinecone")
        _pinecone_module = _p
        logger.info("Imported pinecone module: %s", getattr(_p, "__version__", "unknown"))
    except Exception as e:
        logger.warning("Unable to import pinecone: %s — using local mode", e)
        LOCAL_MODE = True
        _load_local_vectors()
        return

    # Prefer new-style PineconeClient with .indexes (v7+)
    client_cls = getattr(_p, "Pinecone", None) or getattr(_p, "PineconeClient", None) or getattr(_p, "Client", None)
    if client_cls:
        try:
            try:
                _pc = client_cls(api_key=PINECONE_API_KEY, environment=PINECONE_ENV or None)
            except TypeError:
                _pc = client_cls(api_key=PINECONE_API_KEY)
            logger.info("Instantiated Pinecone client: %s", type(_pc))

            if hasattr(_pc, "indexes"):
                try:
                    listed = _pc.indexes.list()
                    existing_names = []
                    if isinstance(listed, (list, tuple)):
                        for item in listed:
                            if hasattr(item, "name"):
                                existing_names.append(item.name)
                            elif isinstance(item, dict) and "name" in item:
                                existing_names.append(item["name"])
                            else:
                                existing_names.append(str(item))
                    else:
                        existing_names = list(listed)
                except Exception as e:
                    logger.debug("Could not list indexes via client.indexes.list(): %s", e)
                    existing_names = []

                if PINECONE_INDEX_NAME not in existing_names:
                    try:
                        _pc.indexes.create(name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine")
                        logger.info("Created index %s via client.indexes.create", PINECONE_INDEX_NAME)
                    except Exception as e:
                        logger.debug("Failed to create index via client.indexes.create: %s", e)

                index = _IndexProxy(_pc, PINECONE_INDEX_NAME)
                logger.info("Using Pinecone via client.indexes (IndexProxy).")
                return

            idx = _make_index_from_client(_pc)
            if idx is not None:
                index = idx
                logger.info("Connected to Pinecone index via client-style API.")
                return

        except Exception as e:
            logger.debug("Client-style Pinecone attempt failed: %s", e)

    try:
        if hasattr(_p, "init"):
            try:
                _p.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV or None)
            except TypeError:
                _p.init(api_key=PINECONE_API_KEY)
            except Exception as e:
                logger.debug("pinecone.init raised: %s", e)
            try:
                listed = _p.list_indexes()
                if isinstance(listed, (list, tuple)):
                    existing_names = list(listed)
                else:
                    existing_names = list(listed)
            except Exception:
                existing_names = []
            if PINECONE_INDEX_NAME not in existing_names:
                try:
                    _p.create_index(name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine")
                    logger.info("Created Pinecone index %s (module-style)", PINECONE_INDEX_NAME)
                except Exception as e:
                    logger.debug("module-style create_index failed: %s", e)
            try:
                index = _p.Index(PINECONE_INDEX_NAME)
                logger.info("Connected to Pinecone via module-style API")
                return
            except Exception as e:
                logger.debug("Module-style Index(...) failed: %s", e)
    except Exception as e:
        logger.debug("Module-style pinecone init attempt failed: %s", e)

    logger.warning("Pinecone client API not usable in this environment — falling back to local mode.")
    LOCAL_MODE = True
    _load_local_vectors()

def _load_local_vectors():
    global _local_buffer
    _local_buffer = []
    if LOCAL_VECTORS_FILE.exists():
        try:
            with LOCAL_VECTORS_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    _local_buffer.append(json.loads(line))
            logger.info("Loaded %d local vectors from %s", len(_local_buffer), LOCAL_VECTORS_FILE)
        except Exception as e:
            logger.warning("Failed to load local vectors: %s", e)

def _flush_local_vectors():
    if not _local_buffer:
        return
    try:
        with LOCAL_VECTORS_FILE.open("a", encoding="utf-8") as f:
            for v in _local_buffer:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        logger.info("Flushed %d vectors to %s", len(_local_buffer), LOCAL_VECTORS_FILE)
        _local_buffer.clear()
    except Exception as e:
        logger.error("Failed to flush local vectors: %s", e)

init_pinecone_or_fallback()

def _with_retries(fn, retries=3, base_delay=1, *args, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            sleep = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed: %s — retrying in %ds", attempt, e, sleep)
            time.sleep(sleep)
    raise last_exc

def extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for i, page in enumerate(reader.pages):
            pages.append((i, page.extract_text() or ""))
        return pages
    except Exception as e:
        logger.error("Failed to read %s: %s", pdf_path, e)
        return []

def extract_abstract_from_pdf(pdf_path: Path, max_pages: int = ABSTRACT_SEARCH_PAGES) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        combined = ""
        num_pages = min(max_pages, len(reader.pages))
        for i in range(num_pages):
            combined += "\n" + (reader.pages[i].extract_text() or "")
        m = re.search(r'(?:^|\n)\s*(?:Abstract|ABSTRACT)\s*[:\n\r]+\s*(.+?)(?=\n\s*\n|Introduction|INTRODUCTION|1\.)', combined, re.S | re.I)
        if m:
            return re.sub(r'\s+', ' ', m.group(1).strip())
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined[:1000].strip()
    except Exception as e:
        logger.warning("Failed to extract abstract from %s: %s", pdf_path, e)
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def generate_embedding(text_chunk: str):
    def _call_new(t):
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=t)
        if isinstance(resp, dict):
            return resp["data"][0]["embedding"]
        if hasattr(resp, "data"):
            data0 = resp.data[0]
            if hasattr(data0, "embedding"):
                return list(data0.embedding)
            return data0["embedding"]
        raise RuntimeError("Unexpected embeddings response shape from OpenAI client")

    def _call_legacy(t):
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=t)
        return resp["data"][0]["embedding"]

    if OPENAI_V2 and openai_client is not None:
        return _with_retries(_call_new, retries=EMBED_RETRIES, base_delay=1, t=text_chunk)
    else:
        return _with_retries(_call_legacy, retries=EMBED_RETRIES, base_delay=1, t=text_chunk)

def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def safe_upsert(vectors: List[dict]):
    if index is not None:
        def _call(vs):
            try:
                return index.upsert(vectors=vs)
            except TypeError:
                return index.upsert(vs)
        return _with_retries(_call, retries=UPSERT_RETRIES, base_delay=1, vs=vectors)
    else:
        for v in vectors:
            _local_buffer.append(v)
        _flush_local_vectors()
        return None

def _get_author_lastnames(authors: List[str]) -> List[str]:
    lastnames = []
    for a in (authors or []):
        a = a.strip()
        if not a:
            continue
        if "," in a:
            last = a.split(",")[0].strip()
        else:
            parts = a.split()
            last = parts[-1] if parts else a
        lastnames.append(last)
    return lastnames

def format_apa_intext(authors: List[str], year: Optional[int], page: Optional[int] = None) -> str:
    year_str = str(year) if year else "n.d."
    lastnames = _get_author_lastnames(authors)
    if not lastnames:
        author_part = "Author"
    elif len(lastnames) == 1:
        author_part = lastnames[0]
    elif len(lastnames) == 2:
        author_part = f"{lastnames[0]} & {lastnames[1]}"
    else:
        author_part = f"{lastnames[0]} et al."
    if page:
        return f"({author_part}, {year_str}, p. {page})"
    return f"({author_part}, {year_str})"

def ingest(save_metadata_back: bool = True):
    if not METADATA_FILE.exists():
        logger.error("Metadata file not found: %s", METADATA_FILE)
        return
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_entries = json.load(f)
    vectors: List[dict] = []
    modified = False
    for entry in metadata_entries:
        src = Path(entry.get("source_pdf", ""))
        if not src.exists():
            src = PDF_FOLDER / src.name
            if not src.exists():
                logger.warning("PDF not found for entry %s -> %s", entry.get("id"), entry.get("source_pdf"))
                continue
        abstract_text = (entry.get("abstract") or "").strip()
        if not abstract_text:
            extracted = extract_abstract_from_pdf(src)
            if extracted:
                abstract_text = extracted
                entry["abstract"] = abstract_text
                modified = True
                logger.info("Extracted abstract for %s", entry.get("id"))
        if abstract_text:
            try:
                emb = generate_embedding(abstract_text)
                vectors.append({
                    "id": f"{entry['id']}_abstract",
                    "values": emb,
                    "metadata": {
                        **entry,
                        "part": "abstract",
                        "citation": f"{src.name}#abstract",
                        "text": abstract_text[:MAX_META_EXCERPT]
                    }
                })
            except Exception as e:
                logger.warning("Embedding abstract failed for %s: %s", entry.get("id"), e)
        pages = extract_pdf_text(src)
        if not pages:
            logger.warning("No text extracted from %s", src.name)
            continue
        for page_idx, page_text in pages:
            page_chunks = chunk_text(page_text)
            for i, chunk in enumerate(page_chunks):
                try:
                    embedding = generate_embedding(chunk)
                except Exception as e:
                    logger.warning("Embedding failed for %s p%d c%d: %s", src.name, page_idx + 1, i, e)
                    continue
                excerpt = chunk[:MAX_META_EXCERPT].replace("\n", " ")
                metadata = {
                    **entry,
                    "part": "body",
                    "page_number": page_idx + 1,
                    "chunk_index": i,
                    "text": excerpt,
                    "citation": f"{src.name}, p. {page_idx + 1}"
                }
                vectors.append({
                    "id": f"{entry['id']}_p{page_idx+1}_c{i}",
                    "values": embedding,
                    "metadata": metadata
                })
                if len(vectors) >= BATCH_SIZE:
                    try:
                        safe_upsert(vectors)
                        logger.info("Upserted %d vectors", len(vectors))
                    except Exception as e:
                        logger.error("Upsert failed: %s", e)
                    vectors = []
    if vectors:
        try:
            safe_upsert(vectors)
            logger.info("Upserted final %d vectors", len(vectors))
        except Exception as e:
            logger.error("Final upsert failed: %s", e)
    if modified and save_metadata_back:
        try:
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata_entries, f, ensure_ascii=False, indent=2)
            logger.info("Updated metadata saved to %s", METADATA_FILE)
        except Exception as e:
            logger.warning("Failed to write updated metadata: %s", e)
    logger.info("Ingestion complete.")

def _load_all_local_vectors() -> List[dict]:
    items = []
    if LOCAL_VECTORS_FILE.exists():
        try:
            with LOCAL_VECTORS_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        except Exception as e:
            logger.warning("Failed to read local vectors file: %s", e)
    return items

def retrieve_with_apa(query: str, top_k: int = 5) -> List[dict]:
    try:
        q_emb = generate_embedding(query)
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
        return []
    if index is not None:
        try:
            try:
                resp = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
            except TypeError:
                resp = index.query(q_emb, top_k=top_k, include_metadata=True)
        except Exception as e:
            logger.error("Pinecone query failed: %s", e)
            return []
        results = []
        matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
        for m in matches:
            meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
            text = meta.get("text") or meta.get("text_excerpt") or ""
            authors = meta.get("authors", []) or []
            year = meta.get("year")
            page = meta.get("page_number") or meta.get("pdf_page") or None
            apa = format_apa_intext(authors, year, page)
            results.append({
                "score": m.get("score") if isinstance(m, dict) else getattr(m, "score", None),
                "text": text,
                "metadata": meta,
                "apa_intext": apa
            })
        return results
    else:
        items = _load_all_local_vectors()
        scored = []
        for it in items:
            emb = it.get("values")
            if not emb:
                continue
            score = _cosine_sim(q_emb, emb)
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, it in scored[:top_k]:
            meta = it.get("metadata", {})
            text = meta.get("text") or meta.get("text_excerpt") or ""
            authors = meta.get("authors", []) or []
            year = meta.get("year")
            page = meta.get("page_number") or meta.get("pdf_page") or None
            apa = format_apa_intext(authors, year, page)
            results.append({
                "score": float(score),
                "text": text,
                "metadata": meta,
                "apa_intext": apa
            })
        return results

if __name__ == "__main__":
    import sys
    try:
        if len(sys.argv) >= 2 and sys.argv[1] in ("ingest", "index"):
            ingest(save_metadata_back=True)
        elif len(sys.argv) >= 3 and sys.argv[1] == "query":
            q = " ".join(sys.argv[2:])
            hits = retrieve_with_apa(q, top_k=5)
            for h in hits:
                print(f"{h['apa_intext']} — score={h['score']}\n{h['text']}\n---\n")
        else:
            print("Usage:")
            print("  python pinecode.py ingest         # run ingestion / upsert")
            print("  python pinecode.py query <terms>  # run a quick retrieval (top-5) and print APA in-text citations")
    except Exception as e:
        logger.exception("Unhandled error in CLI: %s", e)
