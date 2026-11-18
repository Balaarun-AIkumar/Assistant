"""Read policy PDFs and Markdown, preserve citations, and index stable chunks."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
import pdfplumber
import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VECTOR_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "corvanta_policies"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHUNK_TOKENS = 700
OVERLAP_TOKENS = 100

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class PolicyChunk:
    text: str
    metadata: dict[str, Any]
    chunk_id: str


def _tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _parse_header(text: str, source: str) -> dict[str, Any]:
    def value(label: str, default: str = "") -> str:
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else default

    lower = source.lower()
    if "travel-policy-india" in lower:
        country = "IN"
    elif "travel-policy-sri-lanka" in lower:
        country = "LK"
    else:
        country = "ALL"

    if "it-helpdesk-faq" in lower:
        doc_type = "it"
    elif "laptop-policy" in lower:
        doc_type = "laptop"
    elif "leave-policy" in lower:
        doc_type = "leave"
    else:
        doc_type = "travel"

    effective = value("Effective date")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", effective):
        effective = "unknown"

    return {
        "source": source,
        "doc_type": doc_type,
        "country": country,
        "version": value("Version", "unknown"),
        "effective_date": effective,
        "is_current": not lower.endswith("leave-policy-2025.pdf"),
    }


def _paragraph_blocks(text: str) -> list[str]:
    """Group paragraphs, list runs, and tables so steps and rows stay together."""
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    current_kind = ""

    def flush() -> None:
        nonlocal current, current_kind
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)
        current = []
        current_kind = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            kind = "table"
        elif re.match(r"^(?:\d+[.)]|[-*+]\s)", stripped):
            kind = "list"
        elif current_kind == "list":
            # PDF text extraction may wrap one numbered step onto another line.
            kind = "list"
        else:
            kind = "text"

        if current and kind != current_kind:
            flush()
        current.append(stripped)
        current_kind = kind
    flush()
    return blocks


def _chunk_blocks(blocks: Iterable[str], heading: str, base: dict[str, Any]) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    current: list[str] = []

    def add(blocks_to_add: list[str]) -> None:
        body = "\n\n".join(blocks_to_add).strip()
        content = f"{heading}\n\n{body}" if heading else body
        if not content:
            return
        index = len(chunks)
        metadata = dict(base)
        metadata["section"] = heading.lstrip("# ").strip() or "Document"
        digest = hashlib.sha256(f"{metadata['source']}\0{metadata.get('page','')}\0{index}\0{content}".encode("utf-8")).hexdigest()
        chunks.append(PolicyChunk(content, metadata, digest))

    for block in blocks:
        # Keep oversized numbered procedures and tables whole; this is safer than
        # losing a step or separating table values from their headings.
        if _tokens(block) > CHUNK_TOKENS and not (block.startswith("|") or re.match(r"^(?:\d+[.)]|[-*+]\s)", block)):
            words = block.split()
            pieces: list[str] = []
            piece: list[str] = []
            for word in words:
                candidate = " ".join(piece + [word])
                if piece and _tokens(candidate) > CHUNK_TOKENS:
                    pieces.append(" ".join(piece))
                    piece = [word]
                else:
                    piece.append(word)
            if piece:
                pieces.append(" ".join(piece))
            candidates = pieces
        else:
            candidates = [block]

        for candidate in candidates:
            proposed = current + [candidate]
            if current and _tokens("\n\n".join(proposed)) > CHUNK_TOKENS:
                add(current)
                overlap: list[str] = []
                for old in reversed(current):
                    if _tokens("\n\n".join([old] + overlap)) > OVERLAP_TOKENS:
                        break
                    overlap.insert(0, old)
                current = overlap
            current.append(candidate)
    if current:
        add(current)
    return chunks


def _markdown_chunks(path: Path) -> list[PolicyChunk]:
    text = path.read_text(encoding="utf-8")
    base = _parse_header(text, path.name)
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def flush_section() -> None:
        if current_lines:
            title = " > ".join(title for _, title in heading_stack)
            sections.append((title, list(current_lines)))
            current_lines.clear()

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            flush_section()
            level, title = len(match.group(1)), match.group(2).strip()
            heading_stack[:] = [(n, name) for n, name in heading_stack if n < level]
            heading_stack.append((level, title))
        else:
            current_lines.append(line)
    flush_section()

    chunks: list[PolicyChunk] = []
    for title, lines in sections:
        content = "\n".join(lines)
        # Header metadata is kept in document metadata rather than repeated in chunks.
        content = re.sub(r"^(?:Document ID|Version|Effective date|Owner|Applies to|Supersedes):.*$", "", content, flags=re.MULTILINE)
        chunks.extend(_chunk_blocks(_paragraph_blocks(content), f"## {title}" if title else "", base))
    return chunks


def _table_markdown(rows: list[list[str | None]]) -> str:
    normalized = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    normalized = [row for row in normalized if any(row)]
    if not normalized:
        return ""
    width = max(map(len, normalized))
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    lines = ["| " + " | ".join(normalized[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _pdf_chunks(path: Path) -> list[PolicyChunk]:
    all_text = ""
    with pdfplumber.open(path) as pdf:
        pages = [page for page in pdf.pages]
        all_text = "\n".join(page.extract_text() or "" for page in pages)
        base = _parse_header(all_text, path.name)
        chunks: list[PolicyChunk] = []
        for page_number, page in enumerate(pages, start=1):
            tables = page.find_tables()
            boxes = [table.bbox for table in tables]

            def outside_tables(obj: dict[str, Any]) -> bool:
                if obj.get("object_type") != "char":
                    return True
                x0, x1 = obj.get("x0", 0), obj.get("x1", 0)
                top, bottom = obj.get("top", 0), obj.get("bottom", 0)
                return not any(x0 >= bx0 and x1 <= bx2 and top >= by0 and bottom <= by2 for bx0, by0, bx2, by2 in boxes)

            page_items: list[tuple[float, str]] = []
            filtered = page.filter(outside_tables)
            outside = filtered.extract_text() or ""
            if outside.strip():
                page_items.append((0.0, outside))
            for table in tables:
                table_text = _table_markdown(table.extract() or [])
                if table_text:
                    page_items.append((table.bbox[1], table_text))
            page_text = "\n\n".join(text for _, text in sorted(page_items, key=lambda item: item[0]))
            heading_match = re.search(r"^\s*((?:\d+(?:\.\d+)*\s+)?[^\n]{3,100})\s*$", page_text, re.MULTILINE)
            heading = f"Page {page_number}"
            if heading_match:
                heading = f"Page {page_number} | {heading_match.group(1).strip()}"
            page_base = dict(base, page=page_number)
            chunks.extend(_chunk_blocks(_paragraph_blocks(page_text), heading, page_base))
        return chunks


def load_policy_chunks(data_dir: Path = DATA_DIR) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    for path in sorted(data_dir.iterdir()):
        if (path.suffix.lower() == ".md" and ("policy" in path.name.lower() or "faq" in path.name.lower())) or path.suffix.lower() == ".pdf":
            if path.suffix.lower() == ".pdf":
                chunks.extend(_pdf_chunks(path))
            elif "policy" in path.name.lower() or "faq" in path.name.lower():
                chunks.extend(_markdown_chunks(path))
    return chunks


def _embed(texts: list[str], client: Any, task_type: str, batch_size: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[start : start + batch_size],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(item.values for item in response.embeddings)
    return vectors


def ingest_documents(
    data_dir: Path = DATA_DIR,
    persist_dir: Path = VECTOR_DIR,
    gemini_client: Any | None = None,
) -> int:
    """Index all policy files; stable IDs make repeat ingestion idempotent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and gemini_client is None:
        raise RuntimeError("Set GEMINI_API_KEY before building the policy index.")
    client = gemini_client or genai.Client(api_key=api_key)
    chunks = load_policy_chunks(data_dir)
    collection = chromadb.PersistentClient(path=str(persist_dir)).get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    by_source: dict[str, list[PolicyChunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata["source"], []).append(chunk)

    for source, source_chunks in by_source.items():
        old = collection.get(where={"source": source}, include=["metadatas"])
        old_ids = set(old.get("ids", []))
        ids = [chunk.chunk_id for chunk in source_chunks]
        if old_ids - set(ids):
            collection.delete(ids=sorted(old_ids - set(ids)))
        embeddings = _embed([chunk.text for chunk in source_chunks], client, "RETRIEVAL_DOCUMENT")
        collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in source_chunks],
            metadatas=[chunk.metadata for chunk in source_chunks],
            embeddings=embeddings,
        )
    return len(chunks)


def get_collection(persist_dir: Path = VECTOR_DIR):
    return chromadb.PersistentClient(path=str(persist_dir)).get_collection(COLLECTION_NAME)


def embed_query(query: str, client: Any | None = None) -> list[float]:
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before searching policies.")
        client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


if __name__ == "__main__":
    print(f"Indexed {ingest_documents()} policy chunks.")
