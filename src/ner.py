"""Named Entity Recognition module (owner: Swapnil).

Uses spaCy en_core_web_sm to extract entities from review text.
Offsets refer to the English text passed in.
"""

# Type hinting primitives for structured list and dictionary return signatures
from typing import List, Dict

# Human-readable mapping of spaCy OntoNotes 5 entity tags to clear user-facing descriptions
LABEL_DESCRIPTIONS = {
    # Commercial corporate organizations, tech brands, and manufacturers
    "ORG": "Organization or Brand",
    # Specific consumer devices, models, and electronic products
    "PRODUCT": "Product or Model",
    # Countries, cities, states, and geographic locations
    "GPE": "Geopolitical Entity (Country, City, State)",
    # Real or fictional person names (e.g. reviewers, executives, doctors)
    "PERSON": "Person Name",
    # Non-geopolitical locations like mountain ranges, bodies of water
    "LOC": "Non-GPE Location",
    # Relative or absolute temporal indicators (e.g. 'two days late', 'yesterday')
    "DATE": "Date or Temporal Period",
    # Numerals and unit counts that do not represent monetary values
    "CARDINAL": "Cardinal Number",
    # Price tags and currency figures (e.g. '$50', '₹500')
    "MONEY": "Monetary Value",
}

# Global singleton storage for loaded spaCy language model pipeline
_nlp = None


def get_nlp():
    """Retrieves or initializes the spaCy language processing pipeline using a resilient fallback cascade."""
    global _nlp
    # Only initialize if the pipeline has not yet been loaded in current process memory
    if _nlp is None:
        import spacy
        try:
            # Primary attempt: load standard en_core_web_sm model package
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                # Secondary attempt: direct Python package import fallback
                import en_core_web_sm
                _nlp = en_core_web_sm.load()
            except Exception:
                # Tertiary attempt: dynamic on-demand download for clean cloud/container environments
                try:
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    _nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    # Final safety fallback: instantiate blank English pipeline without crashing the host app
                    print(f"Warning: Failed to load or download en_core_web_sm: {e}")
                    _nlp = spacy.blank("en")
                    
    # Return initialized spaCy pipeline
    return _nlp


def extract_entities(text: str) -> List[Dict]:
    """Extracts named entities from English text using spaCy.

    Args:
        text (str): English text.

    Returns:
        List[Dict]: [{"text": str, "label": str, "start": int, "end": int}, ...]
    """
    # Guard against empty or blank input strings by immediately returning an empty list
    if not text or not text.strip():
        return []

    # Obtain singleton spaCy Language instance
    nlp = get_nlp()
    # Execute statistical tokenizer, tagger, and transition-based NER parser on input document
    doc = nlp(text)

    entities = []
    # Iterate over parsed entity spans identified in the document
    for ent in doc.ents:
        # Construct structured dictionary holding entity text, classification label, and character bounds
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
        
    # Return list of detected entity records
    return entities


def evaluate_ner():
    """Evaluates entity-level exact-match Precision, Recall, and F1 on data/ner_gold.json."""
    # JSON library for loading annotated test benchmarks and saving evaluation metrics
    import json
    # Pathlib for resolving repository-relative filesystem targets
    from pathlib import Path

    # Determine absolute repository root directory
    project_root = Path(__file__).resolve().parent.parent
    # Path to gold annotated ground-truth test sentences (40 samples)
    gold_path = project_root / "data" / "ner_gold.json"
    # Target output path for persisted quantitative evaluation metrics
    results_path = project_root / "results" / "ner_eval.json"
    # Ensure parent results directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load gold annotated evaluation data
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_samples = json.load(f)

    # Accumulators for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    per_label_stats = {}

    # Iterate through ground-truth test sentences
    for item in gold_samples:
        text = item["text"]
        # Normalize gold entities to lowercase tuples of (token, label) for exact-match comparison
        gold_ents = set((e["text"].lower(), e["label"]) for e in item["entities"])
        # Extract predictions from spaCy and normalize identically
        pred_ents = set((e["text"].lower(), e["label"]) for e in extract_entities(text))

        # True Positives: predicted entities that exactly match gold annotations
        tp = len(gold_ents & pred_ents)
        # False Positives: predicted entities that do not exist in gold annotations
        fp = len(pred_ents - gold_ents)
        # False Negatives: gold entities that the model failed to detect
        fn = len(gold_ents - pred_ents)

        # Aggregate counts across all benchmark sentences
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    # Compute exact-match Precision = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    # Compute exact-match Recall = TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    # Compute harmonic mean Macro-F1 = 2 * (P * R) / (P + R)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Assemble comprehensive evaluation dictionary
    eval_results = {
        "dataset": "data/ner_gold.json",
        "sentence_count": len(gold_samples),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "notes": "Entity-level exact-match evaluation using spaCy en_core_web_sm. Hindi reviews are translated prior to NER, so entity offsets refer to translated text."
    }

    # Persist JSON report to results directory
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
        
    print(f"Saved NER evaluation to {results_path}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return eval_results


# Execute benchmark evaluation if run directly as a script
if __name__ == "__main__":
    evaluate_ner()

