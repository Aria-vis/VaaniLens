"""Environment check script (owner: Swapnil).

Loads:
- MarianMT (Helsinki-NLP/opus-mt-hi-en)
- spaCy (en_core_web_sm)
- DistilBERT SST-2 (distilbert-base-uncased-finetuned-sst-2-english)
Performs a tiny test translation and records versions to results/env_check.txt.
"""

# Import os for inspecting and modifying system environment variables
import os
# Import sys for standard interpreter properties and stream encoding
import sys
# Import subprocess for running pip freeze to produce dependency lockfiles
import subprocess
# Import Path for cross-platform file path resolution
from pathlib import Path

# Ensure offline execution with cached models if VAANI_OFFLINE flag is asserted
if os.environ.get("VAANI_OFFLINE") == "1":
    # Prevent huggingface hub from trying to ping remote servers
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Enforce transformers to load purely from local snapshot cache
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Define path to results directory in repository root
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# Ensure results folder exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Define destination path for text environment report
ENV_CHECK_PATH = RESULTS_DIR / "env_check.txt"
# Define destination path for exact package version snapshot
LOCK_PATH = Path(__file__).resolve().parent.parent / "requirements.lock.txt"


def run_check():
    """Runs end-to-end environment sanity checks and outputs diagnostic report.
    
    Verifies installed Python version, PyTorch CPU/GPU support, Hugging Face
    Transformers pipeline inference, spaCy NER parsing, and MarianMT Hindi translation.
    """
    # Ensure stdout handles UTF-8 strings gracefully on Windows PowerShell
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Buffer to hold diagnostic log lines
    lines = []
    # Record Python interpreter version and build details
    lines.append(f"Python version: {sys.version}")

    # Check PyTorch installation and device acceleration availability
    import torch
    lines.append(f"PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")

    # Check Transformers library version and load core model classes
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
    lines.append(f"Transformers version: {transformers.__version__}")

    # Verify DistilBERT SST-2 sentiment model functionality
    print("Checking DistilBERT sentiment model...")
    # Hugging Face hub repository ID for DistilBERT fine-tuned on SST-2
    distilbert_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # Instantiate sentiment analysis pipeline on CPU
    clf = pipeline("sentiment-analysis", model=distilbert_name)
    # Execute sample prediction on positive test sentence
    sample_sentiment = clf("This phone has fantastic battery life.")
    # Log prediction label and score
    lines.append(f"DistilBERT SST-2 test result: {sample_sentiment}")

    # Check spaCy natural language processing library
    import spacy
    lines.append(f"spaCy version: {spacy.__version__}")
    print("Checking spaCy model en_core_web_sm...")
    # Load small English statistical pipeline
    nlp = spacy.load("en_core_web_sm")
    # Parse sample sentence containing ORG, PRODUCT, and GPE entities
    doc = nlp("Apple released the iPhone in California.")
    # Extract recognized entity spans and their labels
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    # Log extracted entities to diagnostics
    lines.append(f"spaCy NER entities test: {ents}")

    # Check MarianMT Hindi-to-English neural translation model
    print("Checking MarianMT translation model...")
    # Model identifier for Helsinki-NLP Hindi to English MarianMT
    marian_name = "Helsinki-NLP/opus-mt-hi-en"
    # Load Marian tokenizer from local cache
    marian_tok = AutoTokenizer.from_pretrained(marian_name)
    # Load Marian Seq2Seq model weights
    marian_mod = AutoModelForSeq2SeqLM.from_pretrained(marian_name)
    # Define simple Hindi greeting
    hi_sample = "नमस्ते दुनिया"
    # Tokenize input text into PyTorch tensor dictionary
    inputs = marian_tok(hi_sample, return_tensors="pt")
    # Generate autoregressive output token IDs
    outputs = marian_mod.generate(**inputs)
    # Decode token IDs into English text string
    translated = marian_tok.decode(outputs[0], skip_special_tokens=True)
    # Log translation result
    lines.append(f"MarianMT tiny translation ('{hi_sample}'): {translated}")

    # Join report lines with newlines
    report = "\n".join(lines)
    # Print formatted report to console
    print("\n--- ENVIRONMENT CHECK REPORT ---")
    print(report)

    # Save diagnostic summary to results/env_check.txt
    with open(ENV_CHECK_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved report to {ENV_CHECK_PATH}")

    # Generate pip freeze into requirements.lock.txt for reproducible builds
    print("Generating requirements.lock.txt...")
    # Execute pip freeze as a child process using the active Python executable
    freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    # Write full pip freeze output to requirements.lock.txt
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(freeze_output)
    # Confirm lockfile creation
    print(f"Saved pip freeze to {LOCK_PATH}")


# Standard Python module execution guard
if __name__ == "__main__":
    # Execute environment audit
    run_check()
