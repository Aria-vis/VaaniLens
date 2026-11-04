"""Unit tests for language detection and MarianMT translation (owner: Swapnil)."""

import pytest
from src.translate import detect_language, translate_hi_to_en


def test_detect_language_hindi():
    assert detect_language("यह फ़ोन बहुत अच्छा है") == "hi"


def test_detect_language_english():
    assert detect_language("The battery lasts for two full days.") == "en"


def test_detect_language_empty():
    assert detect_language("") == "en"


def test_translate_hi_to_en_empty():
    assert translate_hi_to_en("") == ""


def test_translate_hi_to_en_real():
    hi_text = "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।"
    translated = translate_hi_to_en(hi_text)
    assert isinstance(translated, str)
    assert len(translated.strip()) > 0

