"""Named Entity Recognition module (owner: Swapnil).

Uses spaCy en_core_web_sm to extract entities from review text.
Offsets refer to the English text passed in.
"""

from typing import List, Dict

LABEL_DESCRIPTIONS = {
    "ORG": "Organization or Brand",
    "PRODUCT": "Product or Model",
    "GPE": "Geopolitical Entity (Country, City, State)",
    "PERSON": "Person Name",
    "LOC": "Non-GPE Location",
    "DATE": "Date or Temporal Period",
    "CARDINAL": "Cardinal Number",
    "MONEY": "Monetary Value",
}

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                import en_core_web_sm
                _nlp = en_core_web_sm.load()
            except Exception:
                # In fresh cloud deployments, auto-download spaCy model if missing
                try:
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    _nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    print(f"Warning: Failed to load or download en_core_web_sm: {e}")
                    _nlp = spacy.blank("en")
    return _nlp


def extract_entities(text: str) -> List[Dict]:
    """Extracts named entities from English text using spaCy.

    Args:
        text (str): English text.

    Returns:
        List[Dict]: [{"text": str, "label": str, "start": int, "end": int}, ...]
    """
    if not text or not text.strip():
        return []

    nlp = get_nlp()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def evaluate_ner():
    """Evaluates entity-level exact-match Precision, Recall, and F1 on data/ner_gold.json."""
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    gold_path = project_root / "data" / "ner_gold.json"
    results_path = project_root / "results" / "ner_eval.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_samples = json.load(f)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    per_label_stats = {}

    for item in gold_samples:
        text = item["text"]
        gold_ents = set((e["text"].lower(), e["label"]) for e in item["entities"])
        pred_ents = set((e["text"].lower(), e["label"]) for e in extract_entities(text))

        tp = len(gold_ents & pred_ents)
        fp = len(pred_ents - gold_ents)
        fn = len(gold_ents - pred_ents)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved NER evaluation to {results_path}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return eval_results


if __name__ == "__main__":
    evaluate_ner()

