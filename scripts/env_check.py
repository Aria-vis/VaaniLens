"""Environment check script (owner: Kaushik/Swapnil).

Loads:
- MarianMT (Helsinki-NLP/opus-mt-hi-en)
- spaCy (en_core_web_sm)
- DistilBERT SST-2 (distilbert-base-uncased-finetuned-sst-2-english)
Performs a tiny test translation and records versions to results/env_check.txt.
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure offline execution with cached models
if os.environ.get("VAANI_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ENV_CHECK_PATH = RESULTS_DIR / "env_check.txt"
LOCK_PATH = Path(__file__).resolve().parent.parent / "requirements.lock.txt"


def run_check():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    lines = []
    lines.append(f"Python version: {sys.version}")

    # Check PyTorch
    import torch
    lines.append(f"PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")

    # Check Transformers & DistilBERT
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
    lines.append(f"Transformers version: {transformers.__version__}")

    print("Checking DistilBERT sentiment model...")
    distilbert_name = "distilbert-base-uncased-finetuned-sst-2-english"
    clf = pipeline("sentiment-analysis", model=distilbert_name)
    sample_sentiment = clf("This phone has fantastic battery life.")
    lines.append(f"DistilBERT SST-2 test result: {sample_sentiment}")

    # Check spaCy
    import spacy
    lines.append(f"spaCy version: {spacy.__version__}")
    print("Checking spaCy model en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Apple released the iPhone in California.")
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    lines.append(f"spaCy NER entities test: {ents}")

    # Check MarianMT
    print("Checking MarianMT translation model...")
    marian_name = "Helsinki-NLP/opus-mt-hi-en"
    marian_tok = AutoTokenizer.from_pretrained(marian_name)
    marian_mod = AutoModelForSeq2SeqLM.from_pretrained(marian_name)
    hi_sample = "नमस्ते दुनिया"
    inputs = marian_tok(hi_sample, return_tensors="pt")
    outputs = marian_mod.generate(**inputs)
    translated = marian_tok.decode(outputs[0], skip_special_tokens=True)
    lines.append(f"MarianMT tiny translation ('{hi_sample}'): {translated}")

    report = "\n".join(lines)
    print("\n--- ENVIRONMENT CHECK REPORT ---")
    print(report)

    with open(ENV_CHECK_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved report to {ENV_CHECK_PATH}")

    # Generate pip freeze into requirements.lock.txt
    print("Generating requirements.lock.txt...")
    freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(freeze_output)
    print(f"Saved pip freeze to {LOCK_PATH}")


if __name__ == "__main__":
    run_check()

