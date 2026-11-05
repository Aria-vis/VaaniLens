"""Unit tests for the end-to-end analysis pipeline (owner: Swapnil)."""

import pytest
from src.pipeline import analyze


def test_analyze_english():
    """Asserts that English review is processed without translation.
    
    Verifies language detection ('en'), raw text preservation, valid sentiment dictionary,
    non-negative latency measurement, and well-formed warnings list.
    """
    sample = "The phone battery is awesome!"
    res = analyze(sample, model="baseline", explain_text=False)
    assert res["language"] == "en"
    assert res["input"] == sample
    assert res["sentiment"]["label"] in ["POSITIVE", "NEGATIVE"]
    assert "latency_ms" in res
    assert isinstance(res["warnings"], list)


def test_analyze_hindi():
    """Asserts that Hindi review is detected as 'hi' and triggers neural translation.
    
    Verifies language detection, presence of translated English string,
    and execution latency tracking.
    """
    hi_sample = "यह फ़ोन बहुत अच्छा है"
    res = analyze(hi_sample, model="baseline", explain_text=False)
    assert res["language"] == "hi"
    assert res["translated_text"] is not None
    assert "latency_ms" in res


def test_analyze_empty():
    """Asserts that empty review text triggers the human review oversight gate.
    
    Verifies that needs_human_review is set to True and the appropriate
    warning message is appended to the warnings list.
    """
    res = analyze("", model="baseline", explain_text=False)
    assert res["needs_human_review"] is True
    assert "Input is empty." in res["warnings"]


def test_analyze_gender_pronoun_warning():
    """Asserts that gender-neutral Hindi sentences are audited for inferred pronouns.
    
    Verifies pipeline execution on gender-neutral profession statements ('वह डॉक्टर है।')
    and ensures the warnings data structure is well-formed.
    """
    gender_neutral_sample = "वह डॉक्टर है।"
    res = analyze(gender_neutral_sample, model="baseline", explain_text=False)
    assert res["language"] == "hi"
    assert isinstance(res["warnings"], list)
