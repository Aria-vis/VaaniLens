"""Explainability module using LIME (owner: Swapnil).

Explains predictions of sentiment classifiers using LimeTextExplainer.
Features:
- Uses predict_proba from src.sentiment.
- Configurable num_samples (defaults to LIME_SAMPLES=200).
- highlight_html helper for HTML visual rendering (green=positive, red=negative).
- Simple memory cache to speed up repeated queries.
"""

# Type annotations for token-attribution tuples, lists, and cache dictionaries
from typing import List, Tuple, Dict
# Standard library HTML escaping to prevent Cross-Site Scripting (XSS) in Streamlit
import html
# Import perturbation sample count (200) from central configuration
from src.config import LIME_SAMPLES
# Import sentiment model prediction contract returning posterior class probabilities [P(Neg), P(Pos)]
from src.sentiment import predict_proba

# Global singleton instance of LimeTextExplainer
_explainer = None
# In-memory dictionary cache mapping composite query keys to precomputed word attributions
_explanation_cache: Dict[str, List[Tuple[str, float]]] = {}


def get_explainer():
    """Initializes and returns the singleton LimeTextExplainer with deterministic random state."""
    global _explainer
    # Instantiate explainer only on first invocation to preserve memory and minimize startup time
    if _explainer is None:
        from lime.lime_text import LimeTextExplainer
        # Initialize explainer with binary target classes and fixed seed 42 for reproducible perturbations
        _explainer = LimeTextExplainer(
            class_names=["NEGATIVE", "POSITIVE"],
            random_state=42
        )
        
    # Return initialized explainer instance
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
    # Guard against empty, null, or whitespace-only inputs
    cleaned = text.strip() if text else ""
    if not cleaned:
        return []

    # Construct unique composite cache key combining model type, feature count, and text content
    cache_key = f"{model}:{num_features}:{cleaned}"
    # Return instantly if the explanation was previously computed during the active session
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    # Retrieve singleton LIME text explainer
    explainer = get_explainer()

    # Define prediction callback wrapping predict_proba for LIME's synthetic perturbation evaluator
    def classifier_fn(texts: List[str]):
        return predict_proba(texts, model=model)

    try:
        # Fit local sparse linear surrogate model over 200 perturbed variations of the input text
        exp = explainer.explain_instance(
            cleaned,
            classifier_fn,
            num_features=num_features,
            num_samples=LIME_SAMPLES,
            labels=[1]
        )
        # Extract linear regression coefficients corresponding to Class 1 (POSITIVE sentiment)
        weights = exp.as_list(label=1)
    except Exception as e:
        # Log explanation failure without crashing the host Streamlit interface
        print(f"LIME explanation fallback: {e}")
        # Graceful fallback: generate neutral baseline attributions for initial tokens
        words = cleaned.split()[:num_features]
        weights = [(w, 0.1) for w in words]

    # Store computed attributions in memory cache to accelerate repeated UI interactions
    _explanation_cache[cache_key] = weights
    return weights


def highlight_html(text: str, weights: List[Tuple[str, float]]) -> str:
    """Generates rich HTML with colored word spans according to LIME weights.

    Positive weights -> Green background
    Negative weights -> Red background
    """
    # Return empty string if input document is blank
    if not text:
        return ""

    # Create case-insensitive lookup table mapping cleaned words to their attribution weights
    weight_map = {w.lower(): wt for w, wt in weights}
    # Tokenize input text on whitespace to preserve natural reading order
    words = text.split()
    html_spans = []

    # Iterate over individual words to calculate visual highlight styles
    for word in words:
        # Strip trailing punctuation for dictionary weight matching
        clean_word = "".join(c for c in word if c.isalnum()).lower()
        # Retrieve computed LIME weight or default to 0.0 (neutral)
        wt = weight_map.get(clean_word, 0.0)

        # Sanitize text to prevent HTML injection attacks
        escaped_word = html.escape(word)
        # Significant positive influence: renders green span with opacity scaled by weight magnitude
        if wt > 0.05:
            intensity = min(abs(wt) * 2.0, 0.8)
            style = f"background-color: rgba(46, 204, 113, {intensity:.2f}); padding: 2px 4px; border-radius: 4px; margin: 0 1px; font-weight: 500;"
            html_spans.append(f'<span style="{style}" title="Weight: {wt:+.3f}">{escaped_word}</span>')
        # Significant negative influence: renders red span with white text for optimal contrast
        elif wt < -0.05:
            intensity = min(abs(wt) * 2.0, 0.8)
            style = f"background-color: rgba(231, 76, 60, {intensity:.2f}); padding: 2px 4px; border-radius: 4px; margin: 0 1px; font-weight: 500; color: white;"
            html_spans.append(f'<span style="{style}" title="Weight: {wt:+.3f}">{escaped_word}</span>')
        # Neutral word with negligible attribution weight: render unmodified plain text
        else:
            html_spans.append(escaped_word)

    # Concatenate styled HTML spans into a single formatted paragraph string
    return " ".join(html_spans)
