# STUB - owner: Tathya - replace entirely
"""Explainability interface module using LIME."""


def explain(text: str, num_features: int = 8, model: str = "transformer") -> list[tuple[str, float]]:
    """Explains sentiment prediction for input text.
    Returns:
        list[tuple[str, float]]: [(word, weight)], positive weight pushes toward POSITIVE.
    """
    if not text:
        return []
    words = text.split()[:num_features]
    return [(w, 0.25) for w in words]
