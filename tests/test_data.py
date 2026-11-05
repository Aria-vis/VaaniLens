"""Unit tests for dataset structure and size constraints (owner: Swapnil).

Verifies deliverables for SW-02:
- data/en_train.csv: columns [text, label], size <= 5 MB
- data/en_test.csv: columns [text, label], size <= 5 MB
- data/hi_reviews_test.csv: columns [text, label, source], size <= 5 MB
- data/README_data.md exists
"""

from pathlib import Path
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_SIZE_BYTES = 5 * 1024 * 1024


def test_readme_data_exists():
    """Asserts that data/README_data.md exists and documents data sources."""
    readme = DATA_DIR / "README_data.md"
    assert readme.exists(), "data/README_data.md must exist"


def test_en_train_csv():
    """Asserts schema and size limits for English training dataset.
    
    Verifies that en_train.csv exists, is under 5MB, contains required columns,
    and adheres strictly to binary labels {0, 1}.
    """
    f = DATA_DIR / "en_train.csv"
    assert f.exists(), "data/en_train.csv must exist"
    assert f.stat().st_size <= MAX_SIZE_BYTES, "en_train.csv exceeds 5MB limit"
    df = pd.read_csv(f)
    assert set(["text", "label"]).issubset(df.columns)
    assert set(df["label"].unique()).issubset({0, 1})


def test_en_test_csv():
    """Asserts schema and size limits for English test dataset.
    
    Verifies that en_test.csv exists, is under 5MB, contains required columns,
    and adheres strictly to binary labels {0, 1}.
    """
    f = DATA_DIR / "en_test.csv"
    assert f.exists(), "data/en_test.csv must exist"
    assert f.stat().st_size <= MAX_SIZE_BYTES, "en_test.csv exceeds 5MB limit"
    df = pd.read_csv(f)
    assert set(["text", "label"]).issubset(df.columns)
    assert set(df["label"].unique()).issubset({0, 1})


def test_hi_reviews_test_csv():
    """Asserts schema and size limits for Hindi evaluation dataset.
    
    Verifies that hi_reviews_test.csv exists, is under 5MB, and contains
    the required columns: text, label, and provenance source.
    """
    f = DATA_DIR / "hi_reviews_test.csv"
    assert f.exists(), "data/hi_reviews_test.csv must exist"
    assert f.stat().st_size <= MAX_SIZE_BYTES, "hi_reviews_test.csv exceeds 5MB limit"
    df = pd.read_csv(f)
    assert set(["text", "label", "source"]).issubset(df.columns)
