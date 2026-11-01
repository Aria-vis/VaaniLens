"""Unit tests for Named Entity Recognition (owner: Kaushik/Swapnil)."""

import pytest
from src.ner import extract_entities, LABEL_DESCRIPTIONS


def test_extract_entities_contract():
    ents = extract_entities("Apple released iPhone in California.")
    assert isinstance(ents, list)
    for ent in ents:
        assert "text" in ent
        assert "label" in ent
        assert "start" in ent
        assert "end" in ent
        assert isinstance(ent["start"], int)
        assert isinstance(ent["end"], int)


def test_extract_entities_empty():
    assert extract_entities("") == []


def test_label_descriptions_present():
    assert "ORG" in LABEL_DESCRIPTIONS
    assert "GPE" in LABEL_DESCRIPTIONS

