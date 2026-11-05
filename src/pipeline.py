"""End-to-end analysis pipeline for VaaniLens (owner: Swapnil).

Integrates:
- Language detection & translation (MarianMT)
- Sentiment analysis (DistilBERT SST-2 or TF-IDF baseline)
- Named Entity Recognition (spaCy)
- Explainability (LIME)
- Ethical safeguards: confidence thresholding, gender pronoun inference warnings, privacy-first (no storage).
"""

# High-resolution timing library for calculating total end-to-end pipeline latency
import time
# Regular expression engine for regex pattern matching on gender pronouns
import re
# Centralized threshold constant for posterior confidence cutoff (0.75)
from src.config import CONF_THRESHOLD
# Translation subsystem functions for script detection and sequence-to-sequence inference
from src.translate import detect_language, translate_hi_to_en
# Dual-engine sentiment classifier wrapper
from src.sentiment import predict_sentiment
# Named entity extraction interface using spaCy
from src.ner import extract_entities
# Local surrogate explainability module using LIME
from src.explain import explain


def analyze(text: str, model: str = "transformer", explain_text: bool = True) -> dict:
    """Runs end-to-end multilingual review analysis pipeline.

    Args:
        text (str): Input text in Hindi or English.
        model (str): "transformer" or "baseline".
        explain_text (bool): Whether to run LIME explanation.

    Returns:
        dict: Complete structured analysis output.
    """
    # Start high-resolution wall-clock timer to measure total execution duration
    start_time = time.time()
    # List accumulator for operational and ethical warnings
    warnings = []
    # Boolean flag indicating whether human-in-the-loop oversight is required
    needs_human_review = False

    # Strip leading/trailing whitespace to prevent empty token allocations
    cleaned_input = text.strip() if text else ""
    # Edge case handler: immediately handle empty or whitespace-only inputs safely
    if not cleaned_input:
        warnings.append("Input is empty.")
        needs_human_review = True
        return {
            "input": text,
            "language": "en",
            "translated_text": None,
            "text_for_analysis": "",
            "sentiment": {"label": "POSITIVE", "score": 0.5, "model": model},
            "entities": [],
            "explanation": [],
            "needs_human_review": True,
            "warnings": warnings,
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    # ---------------------------------------------------------
    # 1. Language & Script Detection
    # ---------------------------------------------------------
    language = "en"
    try:
        # Determine language based on the 30% Devanagari character threshold
        language = detect_language(cleaned_input)
    except Exception as e:
        # Flag error without terminating the pipeline
        warnings.append(f"Language detection failed: {e}")
        needs_human_review = True

    # ---------------------------------------------------------
    # 2. Machine Translation (for Hindi reviews)
    # ---------------------------------------------------------
    translated_text = None
    text_for_analysis = cleaned_input
    if language == "hi":
        try:
            # Translate Hindi text into English using MarianMT sequence-to-sequence model
            translated_text = translate_hi_to_en(cleaned_input)
            # Guard against edge-case blank or degenerate translation outputs
            if not translated_text or len(translated_text.strip()) < 3:
                warnings.append("Translation produced an empty or very short output.")
                needs_human_review = True
                text_for_analysis = cleaned_input
            else:
                # Direct translated English representation to downstream classifiers
                text_for_analysis = translated_text

            # -----------------------------------------------------
            # 2b. Gender Pronoun Hallucination Guard
            # -----------------------------------------------------
            # List of gender-neutral third-person pronouns in Hindi
            neutral_hindi_pronouns = ["वह", "वो", "उसने", "उसका", "उसकी", "उसके"]
            # Check whether source Hindi sentence utilized ungendered third-person pronouns
            has_hindi_neutral = any(p in cleaned_input for p in neutral_hindi_pronouns)
            # Search English translation for binary gendered pronouns (he/she/his/her)
            gendered_english_words = re.findall(r"\b(he|she|his|her|him|hers)\b", text_for_analysis, re.IGNORECASE)

            # If neutral Hindi produced binary English pronouns, issue an ethical inference warning
            if has_hindi_neutral and gendered_english_words:
                warnings.append(
                    "Pronoun gender was inferred by the translation model, not stated in the source."
                )
        except Exception as e:
            # Fallback to source text if translation module throws an exception
            warnings.append(f"Translation failed: {e}")
            needs_human_review = True

    # ---------------------------------------------------------
    # 3. Sentiment Classification
    # ---------------------------------------------------------
    sentiment = {"label": "POSITIVE", "score": 0.5, "model": model}
    try:
        # Run selected model (DistilBERT SST-2 or TF-IDF Baseline) on the English text stream
        sentiment = predict_sentiment(text_for_analysis, model=model)
        # Check against the 0.75 confidence review threshold
        if sentiment.get("score", 0.0) < CONF_THRESHOLD:
            # Low confidence triggers human oversight banner
            needs_human_review = True
            warnings.append(
                f"Low confidence sentiment ({sentiment.get('score', 0):.2f} < {CONF_THRESHOLD:.2f}). Needs human review."
            )
    except Exception as e:
        warnings.append(f"Sentiment analysis failed: {e}")
        needs_human_review = True

    # ---------------------------------------------------------
    # 4. Named Entity Recognition (spaCy)
    # ---------------------------------------------------------
    entities = []
    try:
        # Extract organizations, products, locations, and dates from the English text
        entities = extract_entities(text_for_analysis)
    except Exception as e:
        warnings.append(f"Entity extraction failed: {e}")
        needs_human_review = True

    # ---------------------------------------------------------
    # 5. Explainability (LIME Local Surrogate)
    # ---------------------------------------------------------
    explanation = []
    if explain_text:
        try:
            # Compute token-level attribution weights via local perturbation sampling
            explanation = explain(text_for_analysis, num_features=8, model=model)
        except Exception as e:
            warnings.append(f"Explanation failed: {e}")

    # Compute total elapsed wall-clock latency in milliseconds
    latency_ms = int((time.time() - start_time) * 1000)

    # Return cohesive analysis payload
    return {
        "input": text,
        "language": language,
        "translated_text": translated_text,
        "text_for_analysis": text_for_analysis,
        "sentiment": sentiment,
        "entities": entities,
        "explanation": explanation,
        "needs_human_review": needs_human_review,
        "warnings": warnings,
        "latency_ms": latency_ms,
    }

