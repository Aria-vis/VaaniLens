"""Unit tests for sentiment analysis module (owner: Swapnil).

Tests interface contracts from Section 5 of PROJECT.md:
- predict_sentiment(text: str, model: str) -> dict
  {"label": "POSITIVE" | "NEGATIVE", "score": float, "model": str}
- predict_proba(texts: list[str], model: str) -> np.ndarray shape (n, 2)
"""

import pytest
import numpy as np


def test_predict_sentiment_contract():
    """Validates the output schema and value ranges of predict_sentiment().
    
    Verifies that the returned dictionary matches the exact contract specification:
    - Dict with 'label', 'score', and 'model' keys
    - 'label' is strictly 'POSITIVE' or 'NEGATIVE'
    - 'score' represents confidence of the winning class (range [0.5, 1.0])
    """
    from src.sentiment import predict_sentiment

    res = predict_sentiment("Great product, highly recommend!", model="baseline")
    assert isinstance(res, dict)
    assert res["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0.5 <= res["score"] <= 1.0
    assert "model" in res


def test_predict_sentiment_empty_input():
    """Verifies graceful handling of empty string input without crashing.
    
    Ensures that empty review inputs do not produce IndexError or ZeroDivisionError
    and still return a valid, well-formed sentiment dictionary.
    """
    from src.sentiment import predict_sentiment

    res = predict_sentiment("", model="baseline")
    assert isinstance(res, dict)
    assert res["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0.5 <= res["score"] <= 1.0


def test_predict_proba_shape_and_sum():
    """Validates multidimensional probability distribution properties.
    
    Verifies that predict_proba returns a NumPy ndarray of shape (n_samples, 2),
    where row vectors represent valid categorical probability distributions
    summing to 1.0 within numerical float tolerance.
    """
    from src.sentiment import predict_proba

    samples = ["Loved it!", "Terrible experience.", "Average quality."]
    probas = predict_proba(samples, model="baseline")
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (len(samples), 2)
    sums = probas.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(len(samples)), atol=1e-3)
