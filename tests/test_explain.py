"""Unit tests for LIME explainability module (owner: Swapnil)."""

import pytest
from src.explain import explain


def test_explain_contract():
    exp = explain("The phone battery is awesome", num_features=5, model="baseline")
    assert isinstance(exp, list)
    for item in exp:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], (float, int))


def test_explain_empty():
    assert explain("", model="baseline") == []

