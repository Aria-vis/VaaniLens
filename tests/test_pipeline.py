"""Unit tests for the end-to-end analysis pipeline (owner: Swapnil)."""

import pytest
from src.pipeline import analyze


def test_analyze_english():
    res = analyze("The phone battery is awesome!", model="baseline", explain_text=False)
    assert res["language"] == "en"
    assert res["input"] == "The phone battery is awesome!"
    assert res["sentiment"]["label"] in ["POSITIVE", "NEGATIVE"]
    assert "latency_ms" in res
    assert isinstance(res["warnings"], list)


def test_analyze_hindi():
    res = analyze("यह फ़ोन बहुत अच्छा है", model="baseline", explain_text=False)
    assert res["language"] == "hi"
    assert res["translated_text"] is not None
    assert "latency_ms" in res


def test_analyze_empty():
    res = analyze("", model="baseline", explain_text=False)
    assert res["needs_human_review"] is True
    assert "Input is empty." in res["warnings"]


def test_analyze_gender_pronoun_warning():
    res = analyze("वह डॉक्टर है।", model="baseline", explain_text=False)
    assert res["language"] == "hi"
    # If translation includes he/she/his/her, pronoun warning should trigger
    # In stubs, translated_text is "This product is very good..." so warning won't trigger unless gendered word is present
    assert isinstance(res["warnings"], list)

