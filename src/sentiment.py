# STUB - owner: Swapnil - replace entirely
"""Sentiment analysis interface module."""

import numpy as np


def predict_sentiment(text: str, model: str = "transformer") -> dict:
    """Predicts sentiment for input text.
    Returns:
        dict: {"label": "POSITIVE" | "NEGATIVE", "score": float, "model": str}
    """
    if not text or not text.strip():
        return {"label": "POSITIVE", "score": 0.5, "model": model}
    return {"label": "POSITIVE", "score": 0.85, "model": model}


def predict_proba(texts: list[str], model: str = "transformer") -> np.ndarray:
    """Predicts class probabilities for a list of texts.
    Returns:
        np.ndarray: shape (n, 2), columns = [NEGATIVE, POSITIVE]
    """
    n = len(texts)
    # Return dummy probabilities summing to 1.0
    return np.array([[0.15, 0.85]] * n, dtype=float)
