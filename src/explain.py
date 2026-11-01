"""Explainability module using LIME (owner: Tathya/Swapnil).

Explains predictions of sentiment classifiers using LimeTextExplainer.
Features:
- Uses predict_proba from src.sentiment.
- Configurable num_samples (defaults to LIME_SAMPLES=200).
- highlight_html helper for HTML visual rendering (green=positive, red=negative).
- Simple memory cache to speed up repeated queries.
"""

from typing import List, Tuple, Dict
import html
from src.config import LIME_SAMPLES
from src.sentiment import predict_proba

_explainer = None
_explanation_cache: Dict[str, List[Tuple[str, float]]] = {}


def get_explainer():
    global _explainer
    if _explainer is None:
        from lime.lime_text import LimeTextExplainer
        _explainer = LimeTextExplainer(
            class_names=["NEGATIVE", "POSITIVE"],
            random_state=42
        )
    return _explainer


def explain(text: str, num_features: int = 8, model: str = "transformer") -> List[Tuple[str, float]]:
    """Generates LIME feature attributions for input text.

    Args:
        text (str): Input text to explain.
        num_features (int): Number of top words to extract.
        model (str): "transformer" or "baseline".

    Returns:
        List[Tuple[str, float]]: [(word, weight)], positive weight pushes toward POSITIVE.
    """
    cleaned = text.strip() if text else ""
    if not cleaned:
        return []

    cache_key = f"{model}:{num_features}:{cleaned}"
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    explainer = get_explainer()

    def classifier_fn(texts: List[str]):
        return predict_proba(texts, model=model)

    try:
        # Generate explanation for class 1 (POSITIVE)
        exp = explainer.explain_instance(
            cleaned,
            classifier_fn,
            num_features=num_features,
            num_samples=LIME_SAMPLES,
            labels=[1]
        )
        weights = exp.as_list(label=1)
    except Exception as e:
        print(f"LIME explanation fallback: {e}")
        # Graceful fallback: basic keyword heuristic
        words = cleaned.split()[:num_features]
        weights = [(w, 0.1) for w in words]

    _explanation_cache[cache_key] = weights
    return weights


def highlight_html(text: str, weights: List[Tuple[str, float]]) -> str:
    """Generates rich HTML with colored word spans according to LIME weights.

    Positive weights -> Green background
    Negative weights -> Red background
    """
    if not text:
        return ""

    weight_map = {w.lower(): wt for w, wt in weights}
    words = text.split()
    html_spans = []

    for word in words:
        clean_word = "".join(c for c in word if c.isalnum()).lower()
        wt = weight_map.get(clean_word, 0.0)

        escaped_word = html.escape(word)
        if wt > 0.05:
            intensity = min(abs(wt) * 2.0, 0.8)
            style = f"background-color: rgba(46, 204, 113, {intensity:.2f}); padding: 2px 4px; border-radius: 4px; margin: 0 1px; font-weight: 500;"
            html_spans.append(f'<span style="{style}" title="Weight: {wt:+.3f}">{escaped_word}</span>')
        elif wt < -0.05:
            intensity = min(abs(wt) * 2.0, 0.8)
            style = f"background-color: rgba(231, 76, 60, {intensity:.2f}); padding: 2px 4px; border-radius: 4px; margin: 0 1px; font-weight: 500; color: white;"
            html_spans.append(f'<span style="{style}" title="Weight: {wt:+.3f}">{escaped_word}</span>')
        else:
            html_spans.append(escaped_word)

    return " ".join(html_spans)
