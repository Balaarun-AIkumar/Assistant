from fastapi.testclient import TestClient

from app import main


def test_empty_message_returns_validation_error():
    client = TestClient(main.app)
    response = client.post("/chat", json={"message": "  ", "session_id": "empty-1"})
    assert response.status_code == 422


def test_unknown_employee_is_reported_without_model_call():
    client = TestClient(main.app)
    response = client.post("/chat", json={"message": "What policy applies?", "employee_id": "E9999", "session_id": "unknown-1"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Unknown employee ID: E9999."
    assert response.json()["confidence_flag"] == "low"


def test_session_history_continues_after_clarification(monkeypatch):
    seen = []

    def fake_run_agent(message, employee_id=None, session_id="", history=None):
        seen.append((message, list(history or [])))
        return {"answer": "Which country is your home office in?" if len(seen) == 1 else "Here is the answer.", "sources": [], "confidence_flag": "low", "tool_calls": [], "ticket_id": None}

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    client = TestClient(main.app)
    client.post("/chat", json={"message": "What's the policy?", "session_id": "memory-1"})
    client.post("/chat", json={"message": "Sri Lanka", "session_id": "memory-1"})
    assert seen[1][1] == [
        {"role": "user", "content": "What's the policy?"},
        {"role": "assistant", "content": "Which country is your home office in?"},
    ]


def test_model_failure_returns_clear_error_without_traceback(monkeypatch):
    def fail(**_):
        raise RuntimeError("Set GEMINI_API_KEY to use the assistant.")

    monkeypatch.setattr(main, "run_agent", fail)
    response = TestClient(main.app).post("/chat", json={"message": "Help", "session_id": "failure-1"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Set GEMINI_API_KEY to use the assistant."}
    assert "traceback" not in response.text.lower()
