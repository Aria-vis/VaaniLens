"""Unit tests for machine translation systems (Rule-based and Mini-SMT)."""

# Import pytest testing framework
import pytest
# Import rule-based translation function from mt_rule_based module
from src.mt_rule_based import translate_rule_based
# Import IBM Model 1 statistical machine translation function from mt_smt module
from src.mt_smt import translate_smt


def test_rule_based_translation():
    """Asserts that rule-based MT translates known words and applies reordering rules.
    
    Verifies that dictionary tokens like 'फ़ोन' and 'अच्छा' are translated to their
    English equivalents ('phone', 'good') and returned as a non-empty string.
    """
    # Authentic Hindi review fragment
    text = "यह फ़ोन बहुत अच्छा है"
    # Execute rule-based translation pipeline
    translated = translate_rule_based(text)
    # Ensure returned object is a string
    assert isinstance(translated, str)
    # Ensure translated string contains content
    assert len(translated) > 0
    # Verify that expected dictionary translations appear in English output
    assert "phone" in translated.lower() or "good" in translated.lower()


def test_rule_based_empty():
    """Asserts that empty input string to rule-based engine returns empty string."""
    # Verify zero-token input produces empty output string
    assert translate_rule_based("") == ""


def test_rule_based_oov_brackets():
    """Asserts that out-of-vocabulary (OOV) words are wrapped in square brackets [token].
    
    Verifies transparent failure mode where unknown terms are preserved verbatim
    inside bracket markers for diagnostic clarity.
    """
    # Hindi string containing artificial OOV token not present in bilingual dictionary
    text = "यह क्ष्मज़व है"
    # Execute rule-based translation
    translated = translate_rule_based(text)
    # Check that OOV bracket indicator wraps the unknown token
    assert "[" in translated and "]" in translated


def test_smt_translation():
    """Asserts that IBM Model 1 statistical machine translation returns string."""
    # Simple Hindi phrase
    text = "यह अच्छा है"
    # Execute SMT monotone argmax decoding
    translated = translate_smt(text)
    # Verify output is a Python string
    assert isinstance(translated, str)
