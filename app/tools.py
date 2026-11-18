"""Small, auditable policy, employee, and ticket tools."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .ingestion import DATA_DIR, VECTOR_DIR, embed_query, get_collection

EMPLOYEE_FILE = DATA_DIR / "employees.json"
TICKET_FILE = DATA_DIR / "tickets.json"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.60"))
MAX_RESULTS = int(os.getenv("POLICY_SEARCH_LIMIT", "2"))
VALID_CATEGORIES = {"hardware", "software", "access", "network", "other"}
_TICKET_LOCK = threading.Lock()


def search_policies(
    query: str,
    country: str | None = None,
    doc_type: str | None = None,
    *,
    collection: Any | None = None,
    gemini_client: Any | None = None,
    threshold: float | None = None,
    persist_dir: Path = VECTOR_DIR,
) -> list[dict[str, Any]]:
    """Return current, sufficiently similar passages with citation metadata."""
    query = query.strip()
    if not query:
        return []
    country = country.upper() if country else None
    if country and country not in {"IN", "LK"}:
        raise ValueError("country must be IN or LK.")
    if doc_type and doc_type.lower().strip() not in {"it", "laptop", "leave", "travel"}:
        raise ValueError("doc_type must be it, laptop, leave, or travel.")
    filters: list[dict[str, Any]] = [{"is_current": True}]
    if country in {"IN", "LK"}:
        filters.append({"$or": [{"country": country}, {"country": "ALL"}]})
    if doc_type:
        filters.append({"doc_type": doc_type.lower().strip()})
    where = filters[0] if len(filters) == 1 else {"$and": filters}

    vector = embed_query(query, gemini_client)
    collection = collection or get_collection(persist_dir)
    response = collection.query(
        query_embeddings=[vector],
        n_results=MAX_RESULTS,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict[str, Any]] = []
    min_score = SIMILARITY_THRESHOLD if threshold is None else threshold
    ids = (response.get("ids") or [[]])[0]
    docs = (response.get("documents") or [[]])[0]
    metadata_rows = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    for chunk_id, text, metadata, distance in zip(ids, docs, metadata_rows, distances):
        score = max(0.0, 1.0 - float(distance))
        if score < min_score:
            continue
        results.append(
            {
                "chunk_id": chunk_id,
                "content": text,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page"),
                "section": metadata.get("section", "Document"),
                "effective_date": metadata.get("effective_date", "unknown"),
                "version": metadata.get("version", "unknown"),
                "country": metadata.get("country", "ALL"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "score": round(score, 4),
            }
        )
    return results


def get_employee_profile(employee_id: str, employee_file: Path = EMPLOYEE_FILE) -> dict[str, Any]:
    employee_id = employee_id.strip().upper()
    records = json.loads(employee_file.read_text(encoding="utf-8"))
    for record in records:
        if record.get("employee_id", "").upper() == employee_id:
            return {key: record.get(key) for key in (
                "employee_id", "country", "office", "department", "grade",
                "job_title", "manager_id",
            )}
    return {"error": f"Unknown employee ID: {employee_id}."}


def create_helpdesk_ticket(
    category: str,
    summary: str,
    ticket_file: Path = TICKET_FILE,
) -> dict[str, str]:
    category = category.strip().lower()
    summary = re.sub(r"\s+", " ", summary).strip()
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category. Choose one of: {', '.join(sorted(VALID_CATEGORIES))}.")
    if not summary:
        raise ValueError("Ticket summary cannot be empty.")
    ticket_file.parent.mkdir(parents=True, exist_ok=True)

    with _TICKET_LOCK:
        records = json.loads(ticket_file.read_text(encoding="utf-8")) if ticket_file.exists() else []
        if not isinstance(records, list):
            raise ValueError("Ticket store must contain a JSON array.")
        highest = 0
        for item in records:
            match = re.fullmatch(r"HD-(\d+)", str(item.get("ticket_id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        ticket_id = f"HD-{highest + 1:04d}"
        records.append({"ticket_id": ticket_id, "category": category, "summary": summary})
        temp_file = ticket_file.with_suffix(ticket_file.suffix + ".tmp")
        temp_file.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        temp_file.replace(ticket_file)
    return {"ticket_id": ticket_id, "category": category, "summary": summary}


def is_explicit_ticket_request(message: str) -> bool:
    """Detect a direct request to create a ticket, not a question about process."""
    text = message.strip().lower()
    if re.search(r"\bhow\s+(?:do|can|should)\s+i\s+(?:raise|create|open|log|file)\b", text):
        return False
    action = r"(?:raise|create|open|log|file|submit)"
    ticket = r"(?:a|the|me\s+(?:a|one))\s+(?:helpdesk\s+)?ticket"
    patterns = [
        rf"\bplease\s+{action}\s+{ticket}\b",
        rf"\b(?:can|could|would)\s+you\s+(?:please\s+)?{action}\s+{ticket}\b",
        rf"\b{action}\s+{ticket}\b",
        rf"\bi\s+(?:want|need)\s+to\s+{action}\s+{ticket}\b",
        r"\bi\s+need\s+(?:a|one)\s+(?:helpdesk\s+)?ticket\b",
        rf"\bplease\s+(?:raise|log|file)\s+(?:this|an issue|a ticket)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)
