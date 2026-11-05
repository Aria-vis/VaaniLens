"""Unit tests for sentiment analysis module (owner: Swapnil).

Tests interface contracts from Section 5 of PROJECT.md:
- predict_sentiment(text: str, model: str) -> dict
  {"label": "POSITIVE" | "NEGATIVE", "score": float, "model": str}
- predict_proba(texts: list[str], model: str) -> np.ndarray shape (n, 2)
"""

# Import pytest testing framework
import pytest
# Import numpy for validating probability array dimensions and numerical tolerances
import numpy as np


def test_predict_sentiment_contract():
    """Validates the output schema and value ranges of predict_sentiment().
    
    Verifies that the returned dictionary matches the exact contract specification:
    - Dict with 'label', 'score', and 'model' keys
    - 'label' is strictly 'POSITIVE' or 'NEGATIVE'
    - 'score' represents confidence of the winning class (range [0.5, 1.0])
    """
    # Import predict_sentiment function dynamically to respect lazy loading
    from src.sentiment import predict_sentiment

    # Execute prediction on an unequivocally positive review using baseline classifier
    res = predict_sentiment("Great product, highly recommend!", model="baseline")
    # Verify return object is a standard Python dictionary
    assert isinstance(res, dict)
    # Check that sentiment label is an allowed binary class string
    assert res["label"] in ["POSITIVE", "NEGATIVE"]
    # Verify that confidence score is bounded between 0.5 (random guess) and 1.0 (certain)
    assert 0.5 <= res["score"] <= 1.0
    # Confirm presence of model identifier key in output dictionary
    assert "model" in res


def test_predict_sentiment_empty_input():
    """Verifies graceful handling of empty string input without crashing.
    
    Ensures that empty review inputs do not produce IndexError or ZeroDivisionError
    and still return a valid, well-formed sentiment dictionary.
    """
    # Import predict_sentiment function
    from src.sentiment import predict_sentiment

    # Pass empty review string to baseline model
    res = predict_sentiment("", model="baseline")
    # Verify output is still a dictionary
    assert isinstance(res, dict)
    # Ensure label is valid
    assert res["label"] in ["POSITIVE", "NEGATIVE"]
    # Ensure score conforms to probability range
    assert 0.5 <= res["score"] <= 1.0


def test_predict_proba_shape_and_sum():
    """Validates multidimensional probability distribution properties.
    
    Verifies that predict_proba returns a NumPy ndarray of shape (n_samples, 2),
    where row vectors represent valid categorical probability distributions
    summing to 1.0 within numerical float tolerance.
    """
    # Import predict_proba function
    from src.sentiment import predict_proba

    # Sample batch containing positive, negative, and neutral product reviews
    samples = ["Loved it!", "Terrible experience.", "Average quality."]
    # Query prediction probabilities for batch
    probas = predict_proba(samples, model="baseline")
    # Ensure output structure is a NumPy array
    assert isinstance(probas, np.ndarray)
    # Ensure output matrix dimensions match (number of samples, 2 classes)
    assert probas.shape == (len(samples), 2)
    # Sum probabilities across each row vector
    sums = probas.sum(axis=1)
    # Verify that each row's sum is approximately 1.0 with 1e-3 absolute tolerance
    np.testing.assert_allclose(sums, np.ones(len(samples)), atol=1e-3)
