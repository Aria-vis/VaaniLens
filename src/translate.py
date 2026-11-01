"""Translation and language detection module (owner: Kaushik/Swapnil).

Provides:
- detect_language(text): Devanagari character ratio detection.
- translate_hi_to_en(text): MarianMT (Helsinki-NLP/opus-mt-hi-en) translation with batching.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
from typing import Optional

_marian_tokenizer = None
_marian_model = None
MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"


def detect_language(text: str) -> str:
    """Detects whether text is Hindi ('hi') or English ('en').

    Returns 'hi' if >= 30% of alphabetic letters are Devanagari, else 'en'.
    """
    if not text or not text.strip():
        return "en"

    # Count devanagari characters (\u0900 to \u097F)
    devanagari_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    # Count total letters (Unicode alphabetic characters)
    letters_count = sum(1 for c in text if c.isalpha())

    if letters_count == 0:
        return "hi" if devanagari_count > 0 else "en"

    ratio = devanagari_count / letters_count
    return "hi" if ratio >= 0.3 else "en"


def _load_marian():
    global _marian_tokenizer, _marian_model
    if _marian_model is None or _marian_tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _marian_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _marian_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _marian_tokenizer, _marian_model


def translate_hi_to_en(text: str) -> str:
    """Translates Hindi text to English using MarianMT.

    Splits long text on sentence marks (।, ., ?, !), translates in batches,
    and returns joined English text.
    """
    if not text or not text.strip():
        return ""

    tokenizer, model = _load_marian()

    # Split on sentence boundaries preserving context
    raw_sentences = re.split(r"([।\.\?!]+)", text)
    sentences = []
    # Reassemble sentence text with its punctuation
    current = ""
    for part in raw_sentences:
        current += part
        if re.search(r"[।\.\?!]+", part):
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    if not sentences:
        sentences = [text.strip()]

    translated_parts = []
    batch_size = 8
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        outputs = model.generate(**inputs, num_beams=4, max_length=256)
        decoded = [tokenizer.decode(t, skip_special_tokens=True).strip() for t in outputs]
        translated_parts.extend(decoded)

    return " ".join(translated_parts)
