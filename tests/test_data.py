"""Unit tests for dataset structure and size constraints (owner: Swapnil).

Verifies deliverables for SW-02:
- data/en_train.csv: columns [text, label], size <= 5 MB
- data/en_test.csv: columns [text, label], size <= 5 MB
- data/hi_reviews_test.csv: columns [text, label, source], size <= 5 MB
- data/README_data.md exists
"""

# Import Path for resolving cross-platform filesystem paths
from pathlib import Path
# Import pandas for CSV table reading and column validation
import pandas as pd
# Import pytest framework for assertion and test execution
import pytest

# Define path to data directory in project root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Maximum allowable file size threshold (5 Megabytes in bytes) to keep repository lean
MAX_SIZE_BYTES = 5 * 1024 * 1024


def test_readme_data_exists():
    """Asserts that data/README_data.md exists and documents data sources."""
    # Resolve path to data README file
    readme = DATA_DIR / "README_data.md"
    # Assert presence of documentation file
    assert readme.exists(), "data/README_data.md must exist"


def test_en_train_csv():
    """Asserts schema and size limits for English training dataset.
    
    Verifies that en_train.csv exists, is under 5MB, contains required columns,
    and adheres strictly to binary labels {0, 1}.
    """
    # Resolve path to English train CSV
    f = DATA_DIR / "en_train.csv"
    # Check that file exists on disk
    assert f.exists(), "data/en_train.csv must exist"
    # Verify file does not violate repository size constraint
    assert f.stat().st_size <= MAX_SIZE_BYTES, "en_train.csv exceeds 5MB limit"
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(f)
    # Verify presence of text and label columns
    assert set(["text", "label"]).issubset(df.columns)
    # Ensure all labels are binary integers (0 or 1)
    assert set(df["label"].unique()).issubset({0, 1})


def test_en_test_csv():
    """Asserts schema and size limits for English test dataset.
    
    Verifies that en_test.csv exists, is under 5MB, contains required columns,
    and adheres strictly to binary labels {0, 1}.
    """
    # Resolve path to English test CSV
    f = DATA_DIR / "en_test.csv"
    # Check that file exists on disk
    assert f.exists(), "data/en_test.csv must exist"
    # Verify file does not violate repository size constraint
    assert f.stat().st_size <= MAX_SIZE_BYTES, "en_test.csv exceeds 5MB limit"
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(f)
    # Verify presence of text and label columns
    assert set(["text", "label"]).issubset(df.columns)
    # Ensure all labels are binary integers (0 or 1)
    assert set(df["label"].unique()).issubset({0, 1})


def test_hi_reviews_test_csv():
    """Asserts schema and size limits for Hindi evaluation dataset.
    
    Verifies that hi_reviews_test.csv exists, is under 5MB, and contains
    the required columns: text, label, and provenance source.
    """
    # Resolve path to Hindi test CSV
    f = DATA_DIR / "hi_reviews_test.csv"
    # Check that file exists on disk
    assert f.exists(), "data/hi_reviews_test.csv must exist"
    # Verify file does not violate repository size constraint
    assert f.stat().st_size <= MAX_SIZE_BYTES, "hi_reviews_test.csv exceeds 5MB limit"
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(f)
    # Verify presence of text, label, and source columns
    assert set(["text", "label", "source"]).issubset(df.columns)
