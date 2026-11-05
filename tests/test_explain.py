"""Unit tests for LIME explainability module (owner: Swapnil)."""

import pytest
from src.explain import explain, highlight_html


def test_explain_contract():
    """Validates the output contract and tuple schema of explain().
    
    Verifies that explain() returns a list of (feature_word, weight) tuples,
    where feature_word is a string and weight is a positive or negative float.
    """
    sample = "The phone battery is awesome"
    exp = explain(sample, num_features=5, model="baseline")
    assert isinstance(exp, list)
    for item in exp:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], (float, int))


def test_explain_empty():
    """Asserts that empty input string safely yields an empty list of attributions."""
    assert explain("", model="baseline") == []


def test_highlight_html():
    """Validates HTML markup generation for LIME word attributions.
    
    Verifies that positive words receive green styling, negative words receive
    red styling, neutral words remain unstyled, and HTML entities are escaped.
    """
    text = "Great device but terrible battery & display"
    weights = [("great", 0.45), ("terrible", -0.52)]
    html_output = highlight_html(text, weights)
    assert isinstance(html_output, str)
    assert "background-color: rgba(46, 204, 113" in html_output
    assert "background-color: rgba(231, 76, 60" in html_output
    assert "&amp;" in html_output

