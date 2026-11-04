"""End-to-end analysis pipeline for VaaniLens (owner: Swapnil).

Integrates:
- Language detection & translation (MarianMT)
- Sentiment analysis (DistilBERT SST-2 or TF-IDF baseline)
- Named Entity Recognition (spaCy)
- Explainability (LIME)
- Ethical safeguards: confidence thresholding, gender pronoun inference warnings, privacy-first (no storage).
"""

import time
import re
from src.config import CONF_THRESHOLD
from src.translate import detect_language, translate_hi_to_en
from src.sentiment import predict_sentiment
from src.ner import extract_entities
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
    start_time = time.time()
    warnings = []
    needs_human_review = False

    cleaned_input = text.strip() if text else ""
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

    # 1. Language Detection
    language = "en"
    try:
        language = detect_language(cleaned_input)
    except Exception as e:
        warnings.append(f"Language detection failed: {e}")
        needs_human_review = True

    # 2. Translation (if Hindi)
    translated_text = None
    text_for_analysis = cleaned_input
    if language == "hi":
        try:
            translated_text = translate_hi_to_en(cleaned_input)
            if not translated_text or len(translated_text.strip()) < 3:
                warnings.append("Translation produced an empty or very short output.")
                needs_human_review = True
                text_for_analysis = cleaned_input
            else:
                text_for_analysis = translated_text

            # Gender pronoun warning check:
            # If Hindi source contains gender-neutral pronoun (वह, वो) and English contains he/she/his/her
            neutral_hindi_pronouns = ["वह", "वो", "उसने", "उसका", "उसकी", "उसके"]
            has_hindi_neutral = any(p in cleaned_input for p in neutral_hindi_pronouns)
            gendered_english_words = re.findall(r"\b(he|she|his|her|him|hers)\b", text_for_analysis, re.IGNORECASE)

            if has_hindi_neutral and gendered_english_words:
                warnings.append(
                    "Pronoun gender was inferred by the translation model, not stated in the source."
                )
        except Exception as e:
            warnings.append(f"Translation failed: {e}")
            needs_human_review = True

    # 3. Sentiment Prediction
    sentiment = {"label": "POSITIVE", "score": 0.5, "model": model}
    try:
        sentiment = predict_sentiment(text_for_analysis, model=model)
        if sentiment.get("score", 0.0) < CONF_THRESHOLD:
            needs_human_review = True
            warnings.append(
                f"Low confidence sentiment ({sentiment.get('score', 0):.2f} < {CONF_THRESHOLD:.2f}). Needs human review."
            )
    except Exception as e:
        warnings.append(f"Sentiment analysis failed: {e}")
        needs_human_review = True

    # 4. Named Entity Recognition
    entities = []
    try:
        entities = extract_entities(text_for_analysis)
    except Exception as e:
        warnings.append(f"Entity extraction failed: {e}")
        needs_human_review = True

    # 5. Explainability (LIME)
    explanation = []
    if explain_text:
        try:
            explanation = explain(text_for_analysis, num_features=8, model=model)
        except Exception as e:
            warnings.append(f"Explanation failed: {e}")

    latency_ms = int((time.time() - start_time) * 1000)

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

