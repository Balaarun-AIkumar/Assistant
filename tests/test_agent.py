"""Focused behavior tests; run these in Phase 4."""

import json
from types import SimpleNamespace

from app.agent import FALLBACK, run_agent


class FakeFunctionCall:
    def __init__(self, name, arguments, call_id):
        self.name = name
        self.args = arguments
        self.id = call_id
        self.call_id = call_id


class FakeResponse:
    def __init__(self, output_text="", output=None):
        self.text = output_text
        parts = [SimpleNamespace(function_call=call) for call in (output or [])]
        if output_text:
            parts.append(SimpleNamespace(function_call=None, text=output_text))
        self.candidates = [SimpleNamespace(content=SimpleNamespace(parts=parts))]


class FakeClient:
    def __init__(self, responses):
        self.responses_to_return = list(responses)
        self.models = self

    def generate_content(self, **kwargs):
        return self.responses_to_return.pop(0)


def passage(source, section, content):
    return {
        "source": source,
        "page": None,
        "section": section,
        "effective_date": "2026-01-01",
        "version": "2026",
        "content": content,
        "score": 0.82,
    }


def test_single_policy_answer_has_source_citation():
    client = FakeClient([
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "carry forward", "country": None, "doc_type": "leave"}, "c1")]),
        FakeResponse(output_text="You can carry forward up to 15 days, which must be used by 30 June."),
    ])
    functions = {
        "search_policies": lambda **_: [passage("leave-policy-2026.md", "Carry Forward", "Up to 15 days; use by 30 June.")],
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
    }
    result = run_agent("How much annual leave can I carry forward?", gemini_client=client, tool_functions=functions)
    assert "leave-policy-2026.md" in result["answer"]
    assert result["sources"] == ["leave-policy-2026.md - Carry Forward"]


def test_multi_policy_question_searches_both_policies():
    searched = []

    def search(query, country=None, doc_type=None):
        searched.append(doc_type)
        if doc_type == "leave":
            return [passage("leave-policy-2026.md", "Business Travel", "Business days are OD-T.")]
        return [passage("travel-policy-india.md", "Approvals", "International approvals apply.")]

    client = FakeClient([
        FakeResponse(output=[
            FakeFunctionCall("search_policies", {"query": "business travel leave", "country": "IN", "doc_type": "leave"}, "c1"),
            FakeFunctionCall("search_policies", {"query": "international travel approvals", "country": "IN", "doc_type": "travel"}, "c2"),
        ]),
        FakeResponse(output_text="Business days are recorded as OD-T. The India travel policy sets trip approvals."),
    ])
    result = run_agent("What approvals for business leave in Colombo?", employee_id="E1003", gemini_client=client, tool_functions={
        "search_policies": search,
        "get_employee_profile": lambda _: {"employee_id": "E1003", "country": "IN", "grade": "L3"},
        "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
    })
    assert searched == ["leave", "travel"]
    assert {call["tool"] for call in result["tool_calls"]} >= {"get_employee_profile", "search_policies"}
    assert "travel-policy-india.md" in result["answer"]


def test_travel_question_without_home_country_asks_first():
    result = run_agent("What is my daily allowance for a trip to Bengaluru?", gemini_client=FakeClient([]))
    assert "home office" in result["answer"]
    assert result["tool_calls"] == []


def test_country_reply_uses_prior_question_and_filters_policy():
    class InspectClient(FakeClient):
        def __init__(self, responses):
            super().__init__(responses)
            self.requests = []

        def generate_content(self, **kwargs):
            self.requests.append(kwargs)
            return super().generate_content(**kwargs)

    client = InspectClient([
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "daily allowance", "country": None, "doc_type": "travel"}, "c1")]),
        FakeResponse(output_text="The allowance depends on grade."),
    ])
    searched = []
    history = [
        {"role": "user", "content": "What is my daily allowance for a trip to Bengaluru?"},
        {"role": "assistant", "content": "Which country is your home office in: India or Sri Lanka?"},
    ]
    result = run_agent("My home office is Sri Lanka.", session_id="travel-clarification", history=history,
        gemini_client=client, tool_functions={
            "search_policies": lambda query, country=None, doc_type=None: searched.append((country, doc_type)) or [passage("travel-policy-sri-lanka.md", "Daily Allowance", "Rates vary by grade.")],
            "get_employee_profile": lambda *_: {},
            "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
        })
    assert searched == [("LK", "travel")]
    assert "What is my daily allowance for a trip to Bengaluru?" in str(client.requests[0]["contents"])
    assert "travel-policy-sri-lanka.md" in result["answer"]


def test_no_evidence_returns_exact_fallback():
    client = FakeClient([
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "capital of Australia", "country": None, "doc_type": None}, "c1")]),
        FakeResponse(output_text="The capital is Canberra."),
    ])
    result = run_agent("What is the capital of Australia?", gemini_client=client, tool_functions={
        "search_policies": lambda **_: [],
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
    })
    assert result["answer"] == FALLBACK
    assert result["confidence_flag"] == "low"


def test_question_does_not_create_unsolicited_ticket():
    created = []
    client = FakeClient([
        FakeResponse(output=[FakeFunctionCall("create_helpdesk_ticket", {"category": "hardware", "summary": "Cracked laptop screen"}, "c1")]),
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "broken laptop screen steps", "country": None, "doc_type": "it"}, "c2")]),
        FakeResponse(output_text="Stop using the laptop if liquid was spilled. Raise a hardware ticket for damage."),
    ])
    result = run_agent("What should I do if my laptop screen cracks?", gemini_client=client, tool_functions={
        "search_policies": lambda **_: [passage("it-helpdesk-faq.md", "Broken or Damaged Laptop", "Raise a hardware ticket.")],
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **kwargs: created.append(kwargs),
    })
    assert created == []
    assert result["ticket_id"] is None
    assert "it-helpdesk-faq.md" in result["answer"]


def test_explicit_ticket_request_returns_ticket_id():
    created = []
    client = FakeClient([
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "broken screen helpdesk steps", "country": None, "doc_type": "it"}, "c1")]),
        FakeResponse(output=[FakeFunctionCall("create_helpdesk_ticket", {"category": "hardware", "summary": "Laptop screen cracked"}, "c2")]),
        FakeResponse(output_text="I found the laptop damage steps and raised a ticket."),
    ])
    result = run_agent("My laptop screen is cracked; please raise a ticket.", gemini_client=client, tool_functions={
        "search_policies": lambda **_: [passage("it-helpdesk-faq.md", "Broken or Damaged Laptop", "Raise a hardware ticket.")],
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **kwargs: created.append(kwargs) or {"ticket_id": "HD-1001"},
    })
    assert len(created) == 1
    assert result["ticket_id"] == "HD-1001"
    assert "HD-1001" in result["answer"]


def test_source_from_each_retrieved_policy_is_cited():
    client = FakeClient([
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "travel and leave", "country": "IN", "doc_type": "leave"}, "c1")]),
        FakeResponse(output_text="Business days use OD-T; personal days use annual leave."),
    ])
    sources = [
        passage("leave-policy-2026.md", "Business Travel", "Business days are OD-T."),
        passage("travel-policy-india.md", "Approvals", "Trips require approval."),
    ]
    result = run_agent("How do I combine business travel and leave?", gemini_client=client, tool_functions={
        "search_policies": lambda **_: sources,
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
    })
    assert all(source in result["answer"] for source in result["sources"])


def test_tool_failure_returns_clear_message_without_traceback():
    client = FakeClient([
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "leave", "country": None, "doc_type": "leave"}, "c1")]),
        FakeResponse(output_text=FALLBACK),
    ])

    def broken_search(**_):
        raise RuntimeError("database credentials should not be exposed")

    result = run_agent("How much leave can I carry forward?", gemini_client=client, tool_functions={
        "search_policies": broken_search,
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
    })
    assert result["answer"] == "Policy search failed. Please try again."
    assert "credentials" not in result["answer"]


def test_tool_call_limit_stops_at_five():
    calls = [FakeResponse(output=[FakeFunctionCall("search_policies", {"query": f"query {n}", "country": None, "doc_type": "leave"}, f"c{n}")]) for n in range(5)]
    calls.append(FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "one more", "country": None, "doc_type": "leave"}, "c6")]))
    calls.append(FakeResponse(output_text="PARTIAL: I found limited evidence."))
    result = run_agent("Tell me about leave policy.", gemini_client=FakeClient(calls), tool_functions={
        "search_policies": lambda **_: [passage("leave-policy-2026.md", "Overview", "Leave policy applies.")],
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
    })
    assert len(result["tool_calls"]) == 5
    assert result["confidence_flag"] == "partial"


def test_grounding_prompt_and_deterministic_setting_are_sent():
    class InspectClient(FakeClient):
        calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return super().generate_content(**kwargs)

    client = InspectClient([
        FakeResponse(output=[FakeFunctionCall("search_policies", {"query": "VPN", "country": None, "doc_type": "it"}, "c1")]),
        FakeResponse(output_text="Open GlobalProtect to connect."),
    ])
    run_agent("How do I connect to VPN?", gemini_client=client, tool_functions={
        "search_policies": lambda **_: [passage("it-helpdesk-faq.md", "VPN Setup", "Open GlobalProtect.")],
        "get_employee_profile": lambda *_: {},
        "create_helpdesk_ticket": lambda **_: {"ticket_id": "HD-0001"},
    })
    assert all(call["config"].temperature == 0 for call in client.calls)
    assert "directly supports" in client.calls[0]["config"].system_instruction
    declarations = client.calls[0]["config"].tools[0].function_declarations
    assert "create_helpdesk_ticket" not in {item.name for item in declarations}
