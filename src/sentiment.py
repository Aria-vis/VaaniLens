"""Sentiment analysis contract wrapper (owner: Swapnil).

Implements:
- predict_sentiment(text, model="transformer") -> dict
- predict_proba(texts, model="transformer") -> np.ndarray
Lazy-loads models and handles empty / very long texts robustly.
"""

# Operating system interface for checking offline environment variables
import os

# Suppress remote weight check attempts when operating in air-gapped or offline mode
if os.environ.get("VAANI_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Filesystem path abstraction library
from pathlib import Path
# Type hinting annotations for list and dictionary data structures
from typing import List, Dict
# Numerical linear algebra library for probability array manipulations
import numpy as np
# Serialization utility for loading pre-trained Scikit-learn pipelines
import joblib

# Determine absolute path to repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Path to serialized classical baseline model artifact (TF-IDF + Logistic Regression)
BASELINE_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
# Pretrained Hugging Face transformer model checkpoint fine-tuned on SST-2
TRANSFORMER_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Global module caches to avoid reloading large model weights across consecutive queries
_baseline_model = None
_transformer_pipeline = None
_transformer_tokenizer = None
_transformer_model = None


def _get_baseline():
    """Lazily loads the serialized scikit-learn baseline model or trains it if missing."""
    global _baseline_model
    if _baseline_model is None:
        # Check if pre-trained joblib weights exist on disk
        if not BASELINE_PATH.exists():
            # Trigger on-the-fly training if the artifact is missing
            from src.sentiment_baseline import train_baseline
            _baseline_model, _ = train_baseline()
        else:
            # Deserialize pre-trained Scikit-Learn Pipeline from disk
            _baseline_model = joblib.load(BASELINE_PATH)
            
    # Return initialized baseline pipeline
    return _baseline_model


def _get_transformer_components():
    """Lazily loads the DistilBERT tokenizer and classification model onto CPU memory."""
    global _transformer_tokenizer, _transformer_model
    if _transformer_tokenizer is None or _transformer_model is None:
        # Import Hugging Face components inside function to minimize cold-start time
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # Load subword WordPiece tokenizer
        _transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
        # Load 66M parameter DistilBERT sequence classification model
        _transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME)
        
    # Return initialized tokenizer and model instances
    return _transformer_tokenizer, _transformer_model


def predict_sentiment(text: str, model: str = "transformer") -> Dict:
    """Predicts sentiment for input text.

    Args:
        text (str): Input text.
        model (str): "transformer" or "baseline".

    Returns:
        dict: {"label": "POSITIVE" | "NEGATIVE", "score": float, "model": str}
        score = probability of predicted label (0.5 to 1.0).
    """
    # Sanitize input string by removing whitespace padding
    cleaned = text.strip() if text else ""
    # Safe boundary condition: empty inputs receive default neutral 0.5 probability
    if not cleaned:
        return {"label": "POSITIVE", "score": 0.5, "model": model}

    # Generate posterior probabilities [P(NEGATIVE), P(POSITIVE)] via batch predictor
    probas = predict_proba([cleaned], model=model)[0]
    p_neg, p_pos = float(probas[0]), float(probas[1])

    # Assign winning categorical label based on majority probability score
    if p_pos >= p_neg:
        return {"label": "POSITIVE", "score": round(p_pos, 4), "model": model}
    else:
        return {"label": "NEGATIVE", "score": round(p_neg, 4), "model": model}


def predict_proba(texts: List[str], model: str = "transformer") -> np.ndarray:
    """Predicts class probabilities for a list of texts.

    Args:
        texts (list[str]): List of texts.
        model (str): "transformer" or "baseline".

    Returns:
        np.ndarray: shape (n, 2), columns = [NEGATIVE, POSITIVE].
    """
    # Guard against empty input text batches
    if not texts:
        return np.empty((0, 2))

    # Sanitize every string in the batch; replace empty strings with neutral placeholder
    cleaned_texts = [t.strip() if (t and isinstance(t, str) and t.strip()) else "empty" for t in texts]

    # Branch 1: Classical Statistical Baseline (TF-IDF + Logistic Regression)
    if model == "baseline":
        clf = _get_baseline()
        # Compute probabilities directly using Logistic Regression sigmoid/softmax
        return clf.predict_proba(cleaned_texts)

    # Branch 2: Deep Learning Neural Transformer (DistilBERT SST-2)
    try:
        import torch
        # Retrieve cached tokenizer and model weights
        tokenizer, trans_model = _get_transformer_components()
        # Tokenize batch with dynamic zero-padding and truncation capped at 256 tokens
        inputs = tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        # Execute forward pass with gradient tracking disabled to minimize CPU memory footprint
        with torch.no_grad():
            outputs = trans_model(**inputs)
            # Raw unnormalized prediction logits of shape (batch_size, 2)
            logits = outputs.logits
            # Convert logits to calibrated probability distributions via softmax across last dimension
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
        # SST-2 class convention: column 0 = NEGATIVE, column 1 = POSITIVE
        return probs
    except Exception as e:
        # Resilient fallback: use classical baseline if PyTorch or transformer inference encounters errors
        print(f"Transformer predict_proba fallback due to: {e}")
        clf = _get_baseline()
        return clf.predict_proba(cleaned_texts)
