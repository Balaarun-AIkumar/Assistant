from app.ingestion import DATA_DIR, _markdown_chunks, get_collection, ingest_documents, load_policy_chunks
from app.tools import get_employee_profile, search_policies, create_helpdesk_ticket


def test_all_six_policy_documents_are_loaded():
    chunks = load_policy_chunks(DATA_DIR)
    sources = {chunk.metadata["source"] for chunk in chunks}
    assert sources == {
        "it-helpdesk-faq.md",
        "laptop-policy.pdf",
        "leave-policy-2025.pdf",
        "leave-policy-2026.md",
        "travel-policy-india.md",
        "travel-policy-sri-lanka.md",
    }


def test_laptop_approval_table_keeps_page_and_cost_bands():
    chunks = load_policy_chunks(DATA_DIR)
    matching = [
        chunk for chunk in chunks
        if chunk.metadata["source"] == "laptop-policy.pdf" and chunk.metadata.get("page") == 3
    ]
    text = "\n".join(chunk.text for chunk in matching)
    assert "Approval Limits by Laptop Cost" in text
    assert "INR 90,001" in text
    assert "IT Director" in text


def test_helpdesk_numbered_macos_steps_stay_together():
    chunks = _markdown_chunks(DATA_DIR / "it-helpdesk-faq.md")
    matching = [chunk for chunk in chunks if "2.2 Install and connect on macOS" in chunk.metadata["section"]]
    assert len(matching) == 1
    for number in range(1, 8):
        assert f"{number}." in matching[0].text


def test_superseded_leave_policy_is_marked_non_current():
    chunks = load_policy_chunks(DATA_DIR)
    old = [chunk for chunk in chunks if chunk.metadata["source"] == "leave-policy-2025.pdf"]
    current = [chunk for chunk in chunks if chunk.metadata["source"] == "leave-policy-2026.md"]
    assert old and current
    assert all(chunk.metadata["is_current"] is False for chunk in old)
    assert all(chunk.metadata["is_current"] is True for chunk in current)


def test_search_applies_country_type_and_similarity_filters():
    class FakeEmbeddings:
        def embed_content(self, **_):
            return type("Response", (), {"embeddings": [type("Item", (), {"values": [0.1, 0.2]})()]})()

    class FakeCollection:
        where = None

        def query(self, **kwargs):
            self.where = kwargs["where"]
            return {
                "ids": [["good", "weak"]],
                "documents": [["current passage", "weak passage"]],
                "metadatas": [[
                    {"source": "travel-policy-sri-lanka.md", "section": "Allowance", "country": "LK", "effective_date": "2026-02-01", "version": "2.1", "doc_type": "travel"},
                    {"source": "travel-policy-sri-lanka.md", "section": "Other", "country": "LK", "effective_date": "2026-02-01", "version": "2.1", "doc_type": "travel"},
                ]],
                "distances": [[0.2, 0.6]],
            }

    collection = FakeCollection()
    results = search_policies(
        "hotel allowance", "LK", "travel", collection=collection,
        gemini_client=type("Client", (), {"models": FakeEmbeddings()})(), threshold=0.6,
    )
    assert [row["chunk_id"] for row in results] == ["good"]
    assert collection.where == {
        "$and": [
            {"is_current": True},
            {"$or": [{"country": "LK"}, {"country": "ALL"}]},
            {"doc_type": "travel"},
        ]
    }


def test_employee_lookup_returns_country_and_grade():
    profile = get_employee_profile("E1003")
    assert profile["country"] == "IN"
    assert profile["grade"] == "L3"
    assert not {"name", "email", "manager_name"} & profile.keys()


def test_ticket_write_uses_valid_category_and_hd_id(tmp_path):
    ticket_file = tmp_path / "tickets.json"
    ticket = create_helpdesk_ticket("hardware", "Laptop screen cracked", ticket_file=ticket_file)
    assert ticket["ticket_id"].startswith("HD-")
    assert ticket["category"] == "hardware"
    assert ticket_file.read_text(encoding="utf-8").count(ticket["ticket_id"]) == 1


def test_reingestion_does_not_duplicate_chunks(tmp_path):
    import hashlib

    class FakeEmbeddings:
        def embed_content(self, model, contents, config):
            rows = []
            for text in contents:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                vector = [byte / 255 for byte in digest[:8]]
                rows.append(type("Item", (), {"values": vector})())
            return type("Response", (), {"embeddings": rows})()

    client = type("Client", (), {"models": FakeEmbeddings()})()
    persist_dir = tmp_path / "chroma"
    first = ingest_documents(DATA_DIR, persist_dir, gemini_client=client)
    first_count = get_collection(persist_dir).count()
    second = ingest_documents(DATA_DIR, persist_dir, gemini_client=client)
    second_count = get_collection(persist_dir).count()
    assert first == second
    assert first_count == second_count == first
