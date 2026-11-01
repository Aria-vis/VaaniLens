"""Tests for VaaniBot intent dialogue system (SW-10)."""

import pytest
from src.chatbot import chat


def test_first_turn_disclosure():
    res = chat("Hello")
    assert "automated assistant" in res["reply"]
    assert res["state"]["turn_count"] == 1


def test_intent_greet():
    res = chat("Hi there", state={"turn_count": 1})
    assert res["intent"] == "greet"
    assert res["confidence"] > 0.0


def test_intent_help():
    res_empty = chat("", state={"turn_count": 1})
    assert res_empty["intent"] == "help"

    res_help = chat("What can you do?", state={"turn_count": 1})
    assert res_help["intent"] == "help"


def test_frustration_escalation():
    res = chat("I am extremely angry and furious with this terrible service!", state={"turn_count": 1})
    assert res["escalate"] is True
    assert res["intent"] == "escalate"
    assert "escalated" in res["reply"].lower()


def test_privacy_inquiry():
    res = chat("Do you store my personal review data on disk?", state={"turn_count": 1})
    assert res["intent"] == "privacy"
    assert "purely in memory" in res["reply"].lower() or "stored" in res["reply"].lower()


def test_context_state_tracking():
    state = {"turn_count": 2, "last_analysis": {"sentiment": {"label": "POSITIVE", "score": 0.95}, "explanation": [("excellent", 0.45)]}}
    res = chat("Why did you give this prediction?", state=state)
    assert res["intent"] == "explain_last"
    assert "excellent" in res["reply"]
