"""Unit tests for Named Entity Recognition (owner: Swapnil)."""

# Import pytest testing framework
import pytest
# Import entity extraction function and label mapping from ner module
from src.ner import extract_entities, LABEL_DESCRIPTIONS


def test_extract_entities_contract():
    """Validates the output schema and data types of extract_entities().
    
    Verifies that the function returns a list of dictionaries with keys:
    - 'text' (substring matched by NER model)
    - 'label' (uppercase OntoNotes category code)
    - 'start' (integer character offset)
    - 'end' (integer character offset)
    """
    # Sample sentence containing multiple named entities
    sample = "Apple released iPhone in California."
    # Run spaCy entity extraction pipeline
    ents = extract_entities(sample)
    # Ensure return value is a Python list
    assert isinstance(ents, list)
    # Validate structure of each extracted entity dictionary
    for ent in ents:
        # Check presence of entity surface text
        assert "text" in ent
        # Check presence of category label
        assert "label" in ent
        # Check presence of start offset
        assert "start" in ent
        # Check presence of end offset
        assert "end" in ent
        # Confirm start offset is an integer
        assert isinstance(ent["start"], int)
        # Confirm end offset is an integer
        assert isinstance(ent["end"], int)


def test_extract_entities_empty():
    """Asserts that empty input string safely yields an empty list of entities."""
    # Verify zero-token input produces empty entity list without errors
    assert extract_entities("") == []


def test_label_descriptions_present():
    """Asserts that the entity label description dictionary covers key OntoNotes categories."""
    # Ensure ORG (Companies, agencies, institutions) is defined
    assert "ORG" in LABEL_DESCRIPTIONS
    # Ensure GPE (Countries, cities, states) is defined
    assert "GPE" in LABEL_DESCRIPTIONS
    # Ensure PERSON is defined
    assert "PERSON" in LABEL_DESCRIPTIONS
