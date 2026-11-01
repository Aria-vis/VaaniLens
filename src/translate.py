# STUB - owner: Kaushik - replace entirely
"""Translation and language detection interface module."""


def detect_language(text: str) -> str:
    """Detects whether text is Hindi ('hi') or English ('en').
    Returns 'hi' if >= 30% of alphabetic letters are Devanagari, else 'en'.
    """
    if not text:
        return "en"
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    letters_count = sum(1 for c in text if c.isalpha())
    if letters_count == 0:
        return "hi" if devanagari_count > 0 else "en"
    return "hi" if (devanagari_count / letters_count) >= 0.3 else "en"


def translate_hi_to_en(text: str) -> str:
    """Translates Hindi text to English.
    Returns:
        str: Translated English text.
    """
    if not text or not text.strip():
        return ""
    return "This product is very good, the battery lasts all day."
