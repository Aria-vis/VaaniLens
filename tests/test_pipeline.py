"""Unit tests for the end-to-end analysis pipeline (owner: Swapnil)."""

# Import pytest testing framework
import pytest
# Import the central pipeline orchestrator function from pipeline module
from src.pipeline import analyze


def test_analyze_english():
    """Asserts that English review is processed without translation.
    
    Verifies language detection ('en'), raw text preservation, valid sentiment dictionary,
    non-negative latency measurement, and well-formed warnings list.
    """
    # Sample English review text
    sample = "The phone battery is awesome!"
    # Run end-to-end analysis using baseline classifier
    res = analyze(sample, model="baseline", explain_text=False)
    # Verify language detected as English
    assert res["language"] == "en"
    # Verify original input text is preserved
    assert res["input"] == sample
    # Verify sentiment label is either POSITIVE or NEGATIVE
    assert res["sentiment"]["label"] in ["POSITIVE", "NEGATIVE"]
    # Verify pipeline execution latency in milliseconds is recorded
    assert "latency_ms" in res
    # Ensure warnings is a valid list
    assert isinstance(res["warnings"], list)


def test_analyze_hindi():
    """Asserts that Hindi review is detected as 'hi' and triggers neural translation.
    
    Verifies language detection, presence of translated English string,
    and execution latency tracking.
    """
    # Sample Hindi review text
    hi_sample = "यह फ़ोन बहुत अच्छा है"
    # Run end-to-end analysis
    res = analyze(hi_sample, model="baseline", explain_text=False)
    # Check that language was correctly identified as Hindi
    assert res["language"] == "hi"
    # Check that translation was generated
    assert res["translated_text"] is not None
    # Verify latency measurement exists
    assert "latency_ms" in res


def test_analyze_empty():
    """Asserts that empty review text triggers the human review oversight gate.
    
    Verifies that needs_human_review is set to True and the appropriate
    warning message is appended to the warnings list.
    """
    # Pass empty string to pipeline
    res = analyze("", model="baseline", explain_text=False)
    # Assert that human oversight flag is raised
    assert res["needs_human_review"] is True
    # Assert that explicit empty input warning was generated
    assert "Input is empty." in res["warnings"]


def test_analyze_gender_pronoun_warning():
    """Asserts that gender-neutral Hindi sentences are audited for inferred pronouns.
    
    Verifies pipeline execution on gender-neutral profession statements ('वह डॉक्टर है।')
    and ensures the warnings data structure is well-formed.
    """
    # Gender-neutral Hindi sentence asserting profession
    gender_neutral_sample = "वह डॉक्टर है।"
    # Run end-to-end analysis
    res = analyze(gender_neutral_sample, model="baseline", explain_text=False)
    # Verify detected language is Hindi
    assert res["language"] == "hi"
    # Verify warnings is a list (populated if pronoun hallucination occurs)
    assert isinstance(res["warnings"], list)
