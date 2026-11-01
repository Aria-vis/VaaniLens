"""Unit tests for sentiment analysis module (owner: Swapnil).

Tests interface contracts from Section 5 of PROJECT.md:
- predict_sentiment(text: str, model: str) -> dict
  {"label": "POSITIVE" | "NEGATIVE", "score": float, "model": str}
- predict_proba(texts: list[str], model: str) -> np.ndarray shape (n, 2)
"""

import pytest
import numpy as np


def test_predict_sentiment_contract():
    from src.sentiment import predict_sentiment

    res = predict_sentiment("Great product, highly recommend!", model="baseline")
    assert isinstance(res, dict)
    assert res["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0.5 <= res["score"] <= 1.0
    assert "model" in res


def test_predict_sentiment_empty_input():
    from src.sentiment import predict_sentiment

    res = predict_sentiment("", model="baseline")
    assert isinstance(res, dict)
    assert res["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0.5 <= res["score"] <= 1.0


def test_predict_proba_shape_and_sum():
    from src.sentiment import predict_proba

    samples = ["Loved it!", "Terrible experience.", "Average quality."]
    probas = predict_proba(samples, model="baseline")
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (len(samples), 2)
    # Check probabilities sum to ~1.0
    sums = probas.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(len(samples)), atol=1e-3)

