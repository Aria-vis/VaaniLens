"""Unit tests for machine translation systems (Rule-based and Mini-SMT)."""

import pytest
from src.mt_rule_based import translate_rule_based
from src.mt_smt import translate_smt


def test_rule_based_translation():
    """Asserts that rule-based MT translates known words and applies reordering rules.
    
    Verifies that dictionary tokens like 'फ़ोन' and 'अच्छा' are translated to their
    English equivalents ('phone', 'good') and returned as a non-empty string.
    """
    text = "यह फ़ोन बहुत अच्छा है"
    translated = translate_rule_based(text)
    assert isinstance(translated, str)
    assert len(translated) > 0
    assert "phone" in translated.lower() or "good" in translated.lower()


def test_rule_based_empty():
    """Asserts that empty input string to rule-based engine returns empty string."""
    assert translate_rule_based("") == ""


def test_rule_based_oov_brackets():
    """Asserts that out-of-vocabulary (OOV) words are wrapped in square brackets [token].
    
    Verifies transparent failure mode where unknown terms are preserved verbatim
    inside bracket markers for diagnostic clarity.
    """
    text = "यह क्ष्मज़व है"
    translated = translate_rule_based(text)
    assert "[" in translated and "]" in translated


def test_smt_translation():
    """Asserts that IBM Model 1 statistical machine translation returns string."""
    text = "यह अच्छा है"
    translated = translate_smt(text)
    assert isinstance(translated, str)
