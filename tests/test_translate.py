"""Unit tests for language detection and MarianMT translation (owner: Swapnil)."""

# Import pytest testing framework
import pytest
# Import language detector and neural translation function from translate module
from src.translate import detect_language, translate_hi_to_en


def test_detect_language_hindi():
    """Asserts that predominantly Devanagari text is correctly classified as 'hi'."""
    # Hindi review string containing only Devanagari characters and spaces
    hi_sample = "यह फ़ोन बहुत अच्छा है"
    # Verify that detector returns 'hi' ISO language code
    assert detect_language(hi_sample) == "hi"


def test_detect_language_english():
    """Asserts that ASCII Latin text is correctly classified as 'en'."""
    # English review string containing Latin alphabet characters
    en_sample = "The battery lasts for two full days."
    # Verify that detector returns 'en' ISO language code
    assert detect_language(en_sample) == "en"


def test_detect_language_empty():
    """Asserts that empty or whitespace-only inputs default safely to 'en'."""
    # Check that empty string defaults to 'en' without raising exceptions
    assert detect_language("") == "en"
    # Check that whitespace string defaults to 'en'
    assert detect_language("   ") == "en"


def test_translate_hi_to_en_empty():
    """Asserts that empty input to translation engine returns an empty string immediately."""
    # Verify zero-token short-circuit behavior
    assert translate_hi_to_en("") == ""


def test_translate_hi_to_en_real():
    """Validates end-to-end neural machine translation of authentic Hindi review.
    
    Verifies that MarianMT loads weights, performs beam decoding, and produces
    a non-empty translated English string.
    """
    # Sample Hindi customer review text
    hi_text = "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।"
    # Execute neural translation via Helsinki-NLP/opus-mt-hi-en
    translated = translate_hi_to_en(hi_text)
    # Ensure translated output is a Python string
    assert isinstance(translated, str)
    # Ensure translated string is non-empty after stripping leading/trailing whitespace
    assert len(translated.strip()) > 0
