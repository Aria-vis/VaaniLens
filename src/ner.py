# STUB - owner: Kaushik - replace entirely
"""Named Entity Recognition interface module."""

LABEL_DESCRIPTIONS = {
    "ORG": "Organization or Brand",
    "PRODUCT": "Product or Model",
    "GPE": "Geopolitical Entity (Country, City)",
    "PERSON": "Person Name",
    "LOC": "Non-GPE Location",
    "DATE": "Date or Period",
}


def extract_entities(text: str) -> list[dict]:
    """Extracts named entities from English text.
    Returns:
        list[dict]: [{"text": str, "label": str, "start": int, "end": int}, ...]
    """
    if not text:
        return []
    # Dummy stub entity
    return [{"text": "Redmi Note", "label": "PRODUCT", "start": 4, "end": 14}]
