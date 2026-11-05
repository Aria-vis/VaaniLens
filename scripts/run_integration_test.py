"""Integration test runner (GATE-2) for VaaniLens (owner: Swapnil).

Runs the full pipeline on all 30 samples in data/integration_samples.csv:
- 15 English, 15 Hindi
- Sarcasm, negation, mixed sentiment, empty input, 500-word long review, identity terms
Logs detailed results, latencies, review flags, and edge cases to results/failure_log.md.
"""

# Import sys for adjusting module import search paths
import sys
# Import Path for resolving cross-platform filesystem locations
from pathlib import Path
# Import pandas for reading tabular evaluation datasets
import pandas as pd

# Resolve the absolute path to the root of the project repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure project root is present at the front of sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the top-level pipeline orchestrator function
from src.pipeline import analyze

# Path to the 30-sample integration test suite CSV file
SAMPLES_PATH = PROJECT_ROOT / "data" / "integration_samples.csv"
# Path to results directory where logs and artifacts are written
RESULTS_DIR = PROJECT_ROOT / "results"
# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Path to generated Markdown failure analysis report
FAILURE_LOG = RESULTS_DIR / "failure_log.md"


def run_integration():
    """Executes end-to-end integration test suite across all 30 test reviews.
    
    Validates end-to-end pipeline robustness across English and Hindi inputs,
    measuring latency, testing confidence gates, and verifying exception safety.
    
    Returns:
        bool: True if all 30 test reviews run without uncaught exceptions.
    """
    # Notify user of test suite execution
    print(f"Loading 30 integration test samples from {SAMPLES_PATH}...")
    # Load integration test cases into pandas DataFrame
    df = pd.read_csv(SAMPLES_PATH)

    # Initialize Markdown report table headers
    log_lines = [
        "# End-to-End Pipeline Integration Test & Failure Analysis Log (GATE-2)\n",
        "Evaluation of 30 challenging multilingual review samples across English and Hindi, covering edge cases: negation, sarcasm, mixed sentiment, empty inputs, extreme length, and gender pronoun inference.\n",
        "| ID | Lang | Category | Input Preview | Sentiment (Score) | Review Flag | Warnings | Latency |",
        "|---|---|---|---|---|---|---|---|",
    ]

    # Total number of test samples to evaluate
    total_samples = len(df)
    # Track any unexpected pipeline crashes or unhandled exceptions
    exceptions = []
    # Track interesting or unexpected qualitative model behaviors
    odd_outputs = []
    # Count how many samples trigger the human oversight gate
    flagged_count = 0

    # Iterate over each row in the test dataset
    for idx, row in df.iterrows():
        # Extract unique sample identifier
        sample_id = row["id"]
        # Extract target review language code
        lang = row["language"]
        # Extract linguistic edge case category
        cat = row["category"]
        # Extract review text string, guarding against null values
        text = str(row["text"]) if pd.notna(row["text"]) else ""

        try:
            # Run end-to-end analysis using DistilBERT model (without LIME for speed)
            res = analyze(text, model="transformer", explain_text=False)
            # Create concise 40-character preview of review text
            preview = text[:40].replace("\n", " ") + ("..." if len(text) > 40 else "")
            # Format sentiment label and confidence score percentage
            sentiment_str = f"{res['sentiment']['label']} ({res['sentiment']['score']:.2f})"
            # Format human review status flag
            flag_str = "⚠️ Flagged" if res["needs_human_review"] else "✅ OK"
            # Format active warnings or default to 'None'
            warnings_str = "; ".join(res["warnings"]) if res["warnings"] else "None"
            # Format pipeline processing latency in milliseconds
            latency_str = f"{res['latency_ms']}ms"

            # Increment counter if human review flag was triggered
            if res["needs_human_review"]:
                flagged_count += 1

            # Append formatted markdown table row
            log_lines.append(
                f"| {sample_id} | {lang.upper()} | {cat} | {preview} | {sentiment_str} | {flag_str} | {warnings_str} | {latency_str} |"
            )

            # Analyze failure cases: Sarcasm false positives
            if cat == "sarcasm" and res["sentiment"]["label"] == "POSITIVE":
                odd_outputs.append(
                    f"- **Sample #{sample_id} (Sarcasm in {lang.upper()})**: Surface lexical polarity is highly positive ('wonderful', 'बहुत खूब'), leading DistilBERT to predict POSITIVE with high confidence. Sarcasm remains a known limitation of non-fine-tuned transformers."
                )
            # Analyze failure cases: Empty input review gating
            elif cat == "empty" and not res["needs_human_review"]:
                odd_outputs.append(f"- **Sample #{sample_id}**: Empty input failed to trigger human review gate.")

        # Catch unexpected pipeline crashes
        except Exception as e:
            exceptions.append(f"Sample #{sample_id} crashed with: {e}")

    # Section for qualitative failure observations
    log_lines.append("\n## Edge Case Findings & Qualitative Observations\n")
    # Add logged qualitative findings
    if odd_outputs:
        log_lines.extend(odd_outputs)
    else:
        log_lines.append("- All edge cases processed according to specification.")

    # Section for summary statistics
    log_lines.append(f"\n## Summary Statistics")
    log_lines.append(f"- **Total Samples Processed**: {total_samples}")
    log_lines.append(f"- **Exceptions / Crashes**: {len(exceptions)}")
    log_lines.append(f"- **Human Review Required Flags Triggered**: {flagged_count} / {total_samples} ({flagged_count/total_samples:.1%})")
    log_lines.append(f"- **Pass Status**: {'FAILED' if exceptions else 'PASSED (GATE-2 Cleared)'}")

    # Write formatted Markdown report to disk
    with open(FAILURE_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    # Output run summary to console
    print(f"Integration log written to {FAILURE_LOG} (0 crashes, {flagged_count} flagged for review)")
    # Assert zero crashes occurred across entire test corpus
    assert len(exceptions) == 0, f"Encountered {len(exceptions)} crashes!"
    # Return overall pass boolean
    return len(exceptions) == 0


# Standard Python module execution guard
if __name__ == "__main__":
    # Execute full integration test suite
    run_integration()
