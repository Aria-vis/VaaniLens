"""Sentiment analysis contract wrapper (owner: Swapnil).

Implements:
- predict_sentiment(text, model="transformer") -> dict
- predict_proba(texts, model="transformer") -> np.ndarray
Lazy-loads models and handles empty / very long texts robustly.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from typing import List, Dict
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
TRANSFORMER_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_baseline_model = None
_transformer_pipeline = None
_transformer_tokenizer = None
_transformer_model = None


def _get_baseline():
    global _baseline_model
    if _baseline_model is None:
        if not BASELINE_PATH.exists():
            from src.sentiment_baseline import train_baseline
            _baseline_model, _ = train_baseline()
        else:
            _baseline_model = joblib.load(BASELINE_PATH)
    return _baseline_model


def _get_transformer_components():
    global _transformer_tokenizer, _transformer_model
    if _transformer_tokenizer is None or _transformer_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
        _transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME)
    return _transformer_tokenizer, _transformer_model


def predict_sentiment(text: str, model: str = "transformer") -> Dict:
    """Predicts sentiment for input text.

    Args:
        text (str): Input text.
        model (str): "transformer" or "baseline".

    Returns:
        dict: {"label": "POSITIVE" | "NEGATIVE", "score": float, "model": str}
        score = probability of predicted label (0.5 to 1.0).
    """
    cleaned = text.strip() if text else ""
    if not cleaned:
        return {"label": "POSITIVE", "score": 0.5, "model": model}

    probas = predict_proba([cleaned], model=model)[0]
    p_neg, p_pos = float(probas[0]), float(probas[1])

    if p_pos >= p_neg:
        return {"label": "POSITIVE", "score": round(p_pos, 4), "model": model}
    else:
        return {"label": "NEGATIVE", "score": round(p_neg, 4), "model": model}


def predict_proba(texts: List[str], model: str = "transformer") -> np.ndarray:
    """Predicts class probabilities for a list of texts.

    Args:
        texts (list[str]): List of texts.
        model (str): "transformer" or "baseline".

    Returns:
        np.ndarray: shape (n, 2), columns = [NEGATIVE, POSITIVE].
    """
    if not texts:
        return np.empty((0, 2))

    cleaned_texts = [t.strip() if (t and isinstance(t, str) and t.strip()) else "empty" for t in texts]

    if model == "baseline":
        clf = _get_baseline()
        return clf.predict_proba(cleaned_texts)

    # Default: Transformer (DistilBERT)
    try:
        import torch
        tokenizer, trans_model = _get_transformer_components()
        inputs = tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = trans_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # SST-2 order: index 0 = NEGATIVE, index 1 = POSITIVE
        return probs
    except Exception as e:
        # Fallback to baseline if transformer fails
        print(f"Transformer predict_proba fallback due to: {e}")
        clf = _get_baseline()
        return clf.predict_proba(cleaned_texts)
