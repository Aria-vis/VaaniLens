"""Unit tests for language detection and MarianMT translation (owner: Swapnil)."""

import pytest
from src.translate import detect_language, translate_hi_to_en


def test_detect_language_hindi():
    """Asserts that predominantly Devanagari text is correctly classified as 'hi'."""
    hi_sample = "यह फ़ोन बहुत अच्छा है"
    assert detect_language(hi_sample) == "hi"


def test_detect_language_english():
    """Asserts that ASCII Latin text is correctly classified as 'en'."""
    en_sample = "The battery lasts for two full days."
    assert detect_language(en_sample) == "en"


def test_detect_language_empty():
    """Asserts that empty or whitespace-only inputs default safely to 'en'."""
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"


def test_translate_hi_to_en_empty():
    """Asserts that empty input to translation engine returns an empty string immediately."""
    assert translate_hi_to_en("") == ""


def test_translate_hi_to_en_real():
    """Validates end-to-end neural machine translation of authentic Hindi review.
    
    Verifies that MarianMT loads weights, performs beam decoding, and produces
    a non-empty translated English string.
    """
    hi_text = "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।"
    translated = translate_hi_to_en(hi_text)
    assert isinstance(translated, str)
    assert len(translated.strip()) > 0
