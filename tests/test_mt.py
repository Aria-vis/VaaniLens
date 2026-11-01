"""Unit tests for machine translation systems (Rule-based and Mini-SMT)."""

import pytest
from src.mt_rule_based import translate_rule_based
from src.mt_smt import translate_smt


def test_rule_based_translation():
    text = "यह फ़ोन बहुत अच्छा है"
    translated = translate_rule_based(text)
    assert isinstance(translated, str)
    assert len(translated) > 0
    # Should contain dictionary words
    assert "phone" in translated.lower() or "good" in translated.lower()


def test_rule_based_empty():
    assert translate_rule_based("") == ""


def test_rule_based_oov_brackets():
    text = "यह क्ष्मज़व है"  # nonsense OOV word
    translated = translate_rule_based(text)
    assert "[" in translated and "]" in translated


def test_smt_translation():
    text = "यह अच्छा है"
    translated = translate_smt(text)
    assert isinstance(translated, str)

