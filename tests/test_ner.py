"""Unit tests for Named Entity Recognition (owner: Swapnil)."""

import pytest
from src.ner import extract_entities, LABEL_DESCRIPTIONS


def test_extract_entities_contract():
    """Validates the output schema and data types of extract_entities().
    
    Verifies that the function returns a list of dictionaries with keys:
    - 'text' (substring matched by NER model)
    - 'label' (uppercase OntoNotes category code)
    - 'start' (integer character offset)
    - 'end' (integer character offset)
    """
    sample = "Apple released iPhone in California."
    ents = extract_entities(sample)
    assert isinstance(ents, list)
    for ent in ents:
        assert "text" in ent
        assert "label" in ent
        assert "start" in ent
        assert "end" in ent
        assert isinstance(ent["start"], int)
        assert isinstance(ent["end"], int)


def test_extract_entities_empty():
    """Asserts that empty input string safely yields an empty list of entities."""
    assert extract_entities("") == []


def test_label_descriptions_present():
    """Asserts that the entity label description dictionary covers key OntoNotes categories."""
    assert "ORG" in LABEL_DESCRIPTIONS
    assert "GPE" in LABEL_DESCRIPTIONS
    assert "PERSON" in LABEL_DESCRIPTIONS
