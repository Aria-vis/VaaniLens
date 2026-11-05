"""Translation and language detection module (owner: Swapnil).

Provides:
- detect_language(text): Devanagari character ratio detection.
- translate_hi_to_en(text): MarianMT (Helsinki-NLP/opus-mt-hi-en) translation with batching.
"""

# Import operating system interface to inspect environment variables
import os

# Check if offline mode is enforced to prevent unintended remote Hugging Face network requests
if os.environ.get("VAANI_OFFLINE") == "1":
    # Suppress remote model searches in huggingface_hub
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Suppress remote weight searches in transformers library
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Regular expression library for text tokenization and punctuation boundary extraction
import re
# Type hinting helper for optional parameters and return values
from typing import Optional

# Global cache variables for singleton lazy-loading of tokenizer and model weights
_marian_tokenizer = None
_marian_model = None

# Hugging Face model identifier for Helsinki-NLP Hindi-to-English MarianMT sequence-to-sequence model
MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"


def detect_language(text: str) -> str:
    """Detects whether text is Hindi ('hi') or English ('en').

    Returns 'hi' if >= 30% of alphabetic letters are Devanagari, else 'en'.
    """
    # Guard against empty, null, or whitespace-only inputs by defaulting safely to English
    if not text or not text.strip():
        return "en"

    # Count characters that fall strictly within the Unicode Devanagari script range (\u0900 to \u097F)
    devanagari_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    
    # Count total Unicode alphabetic characters in the string to filter out numbers and symbols
    letters_count = sum(1 for c in text if c.isalpha())

    # If the text contains no standard alphabetic letters (e.g. only numbers or punctuation)
    if letters_count == 0:
        # Route to Hindi if any Devanagari symbol exists, otherwise fallback to English
        return "hi" if devanagari_count > 0 else "en"

    # Calculate proportion of Devanagari script characters among all detected alphabetic characters
    ratio = devanagari_count / letters_count
    
    # Decision threshold: classified as Hindi ('hi') if >= 30% Devanagari, handling Hinglish/code-mixed reviews
    return "hi" if ratio >= 0.3 else "en"


def _load_marian():
    """Lazily loads and caches the MarianMT tokenizer and model in memory on first call."""
    global _marian_tokenizer, _marian_model
    # Check if either model component is uninitialized
    if _marian_model is None or _marian_tokenizer is None:
        # Import Hugging Face Seq2Seq modules locally to keep cold-start times minimal
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        # Initialize sentencepiece subword tokenizer for Hindi vocabulary
        _marian_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Load 74M parameter 6-layer sequence-to-sequence transformer model weights onto CPU
        _marian_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
    # Return initialized singleton references
    return _marian_tokenizer, _marian_model


def translate_hi_to_en(text: str) -> str:
    """Translates Hindi text to English using MarianMT.

    Splits long text on sentence marks (।, ., ?, !), translates in batches,
    and returns joined English text.
    """
    # Guard against empty or blank inputs to avoid unnecessary tensor allocations
    if not text or not text.strip():
        return ""

    # Retrieve cached MarianMT tokenizer and model instances
    tokenizer, model = _load_marian()

    # Split text while preserving punctuation marks (Hindi danda '।' as well as '.', '?', '!')
    raw_sentences = re.split(r"([।\.\?!]+)", text)
    sentences = []
    current = ""
    
    # Reassemble parsed chunks so each sentence retains its terminating punctuation mark
    for part in raw_sentences:
        current += part
        # When a punctuation delimiter is encountered, finalize the current sentence unit
        if re.search(r"[।\.\?!]+", part):
            if current.strip():
                sentences.append(current.strip())
            current = ""
            
    # Capture any trailing text segment that lacked terminal punctuation
    if current.strip():
        sentences.append(current.strip())

    # Fallback if sentence boundary detection yielded no distinct units
    if not sentences:
        sentences = [text.strip()]

    translated_parts = []
    # Set mini-batch size to 8 to balance memory throughput on CPU
    batch_size = 8
    
    # Iterate across sentence chunks in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        # Tokenize batch with dynamic padding and truncation capped at 256 subword tokens
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        # Generate translations using beam search (num_beams=4) for high fluency and translation fidelity
        outputs = model.generate(**inputs, num_beams=4, max_length=256)
        # Decode output token IDs back into clean English text, skipping special tokens like </s>
        decoded = [tokenizer.decode(t, skip_special_tokens=True).strip() for t in outputs]
        # Append decoded sentences to the running translation list
        translated_parts.extend(decoded)

    # Recombine all translated segments into a single cohesive English paragraph
    return " ".join(translated_parts)
