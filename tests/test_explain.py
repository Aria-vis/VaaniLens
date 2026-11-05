"""Unit tests for LIME explainability module (owner: Swapnil)."""

# Import pytest testing framework
import pytest
# Import LIME explainability generator and HTML highlighter from explain module
from src.explain import explain, highlight_html


def test_explain_contract():
    """Validates the output contract and tuple schema of explain().
    
    Verifies that explain() returns a list of (feature_word, weight) tuples,
    where feature_word is a string and weight is a positive or negative float.
    """
    # Sample English review sentence with prominent positive sentiment
    sample = "The phone battery is awesome"
    # Generate top-5 LIME feature attributions using baseline model
    exp = explain(sample, num_features=5, model="baseline")
    # Verify return object is a Python list
    assert isinstance(exp, list)
    # Check structure and data types of each feature attribution pair
    for item in exp:
        # Verify item is a tuple
        assert isinstance(item, tuple)
        # Verify tuple has exactly two elements (word, score)
        assert len(item) == 2
        # Check that the first element is the word string
        assert isinstance(item[0], str)
        # Check that the second element is a numeric attribution weight
        assert isinstance(item[1], (float, int))


def test_explain_empty():
    """Asserts that empty input string safely yields an empty list of attributions."""
    # Verify zero-token input short-circuits to empty list without raising errors
    assert explain("", model="baseline") == []


def test_highlight_html():
    """Validates HTML markup generation for LIME word attributions.
    
    Verifies that positive words receive green styling, negative words receive
    red styling, neutral words remain unstyled, and HTML entities are escaped.
    """
    # Sample text with positive, negative, and neutral tokens
    text = "Great device but terrible battery & display"
    # Simulated LIME weights
    weights = [("great", 0.45), ("terrible", -0.52)]
    # Generate highlighted HTML string
    html_output = highlight_html(text, weights)
    # Verify return is string
    assert isinstance(html_output, str)
    # Check green span for 'Great'
    assert "background-color: rgba(46, 204, 113" in html_output
    # Check red span for 'terrible'
    assert "background-color: rgba(231, 76, 60" in html_output
    # Check HTML ampersand escaping
    assert "&amp;" in html_output

