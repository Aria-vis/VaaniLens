"""Mini Statistical Machine Translation using IBM Model 1 (owner: Swapnil).

Linguistic Architecture & Limitations:
This module implements a toy, word-level Statistical Machine Translation (SMT) system based
on Brown et al.'s IBM Model 1 using Expectation-Maximization (EM). It trains a lexical translation
probability distribution t(e|f) over parallel sentence pairs.

Limitations:
- Purely monotone word-by-word decoding without phrase-based translations.
- No distortion / reordering model (unlike IBM Model 2-5 or Moses).
- No target language model (reranking), resulting in disfluent English syntax.
Demonstrates the foundational step of statistical NLP before the advent of Neural MT.
"""

# Pickle library for serializing the learned lexical translation probability table
import pickle
# Regular expressions for tokenizing text on whitespace delimiters
import re
# Pathlib for resolving model persistence filesystem paths
from pathlib import Path
# Defaultdict collection for dynamically building 2D nested probability matrices
from collections import defaultdict

# Path to serialized IBM Model 1 lexical translation lookup table
MODEL_CACHE = Path(__file__).resolve().parent.parent / "models" / "smt_ibm1.pkl"


class MiniSMTTranslator:
    """Implements IBM Model 1 Statistical Machine Translation with Expectation-Maximization training."""

    def __init__(self, model_path: Path = MODEL_CACHE):
        """Initializes empty lexical probability matrix or loads cached model from disk."""
        # Nested dictionary mapping target English tokens -> {source Hindi tokens: probability}
        self.translation_table = defaultdict(lambda: defaultdict(float))
        self.model_path = model_path
        # Load serialized probability distribution if already compiled
        if model_path.exists():
            self.load(model_path)

    def train(self, parallel_pairs: list[tuple[str, str]], iterations: int = 5):
        """Trains IBM Model 1 lexical translation probabilities using EM.
        Args:
            parallel_pairs: List of (hindi_sentence, english_sentence)
            iterations: Number of EM iterations.
        """
        try:
            # Import NLTK statistical machine translation alignment primitives
            from nltk.translate import AlignedSent, IBMModel1
            aligned_corpus = []
            
            # Prepare parallel aligned sentence pairs
            for hi_sent, en_sent in parallel_pairs:
                # Lowercase and tokenize both language sides
                hi_tokens = [w.lower() for w in re.split(r"\s+", hi_sent) if w]
                en_tokens = [w.lower() for w in re.split(r"\s+", en_sent) if w]
                if hi_tokens and en_tokens:
                    # In NLTK IBMModel1: 'words' represents target English, 'mots' represents source Hindi
                    aligned_corpus.append(AlignedSent(words=en_tokens, mots=hi_tokens))

            # Run Expectation-Maximization (EM) algorithm to estimate lexical translation probabilities t(e|f)
            ibm1 = IBMModel1(aligned_corpus, iterations)
            self.translation_table = ibm1.translation_table
            # Persist learned lexical translation table to disk
            self.save(self.model_path)
            print(f"Trained IBM Model 1 on {len(aligned_corpus)} pairs.")
        except Exception as e:
            # Fallback to co-occurrence frequency estimation if NLTK raises errors
            print(f"NLTK IBMModel1 training error: {e}. Using co-occurrence fallback.")
            self._train_cooccurrence(parallel_pairs)

    def _train_cooccurrence(self, pairs):
        """Fallback training method computing normalized joint token co-occurrence frequencies."""
        counts = defaultdict(lambda: defaultdict(float))
        # Accumulate joint co-occurrence counts across all sentence pairs
        for hi_sent, en_sent in pairs:
            hi_tokens = [w.lower() for w in re.split(r"\s+", hi_sent) if w]
            en_tokens = [w.lower() for w in re.split(r"\s+", en_sent) if w]
            for h in hi_tokens:
                for e in en_tokens:
                    counts[h][e] += 1.0

        # Normalize counts into proper probability distributions summing to 1.0
        for h, targets in counts.items():
            total = sum(targets.values())
            for e, count in targets.items():
                self.translation_table[e][h] = count / total
                
        # Persist fallback model to disk
        self.save(self.model_path)

    def translate_sentence(self, sentence: str) -> str:
        """Executes monotone word-level translation using maximum a posteriori lexical lookup."""
        # Return empty string if input sentence is blank
        if not sentence or not sentence.strip():
            return ""
        # Tokenize source Hindi text on whitespace
        tokens = [w.lower() for w in re.split(r"\s+", sentence) if w]
        output = []

        # Find the English word e that maximizes probability t(e|f) for each Hindi word f
        for hi_word in tokens:
            best_en = None
            best_prob = 0.0
            for en_word, hi_dict in self.translation_table.items():
                prob = hi_dict.get(hi_word, 0.0)
                if prob > best_prob:
                    best_prob = prob
                    best_en = en_word

            # Filter out spurious low-probability alignments (< 0.01)
            if best_en and best_prob > 0.01:
                output.append(best_en)
            else:
                # Mark untranslated word as Out-of-Vocabulary in square brackets
                output.append(f"[{hi_word}]")

        # Join tokens into a sentence string with initial capitalization
        res = " ".join(output)
        return res.capitalize() if res else ""

    def translate(self, text: str) -> str:
        """Splits multi-sentence document, translates each sentence, and reassembles with periods."""
        if not text:
            return ""
        # Split document along sentence boundary punctuation marks
        sentences = re.split(r"[।\.\n\?!]+", text)
        translated = [self.translate_sentence(s) for s in sentences if s.strip()]
        # Concatenate translated sentences
        return ". ".join(translated) + ("." if translated else "")

    def save(self, path: Path):
        """Serializes translation probability dictionary to binary pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert defaultdict to normal dict for clean pickling
        plain_dict = {en: dict(sources) for en, sources in self.translation_table.items()}
        with open(path, "wb") as f:
            pickle.dump(plain_dict, f)

    def load(self, path: Path):
        """Deserializes translation probability dictionary from binary pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            # Reconstruct nested defaultdict with zero default float values
            self.translation_table = defaultdict(lambda: defaultdict(float), {
                en: defaultdict(float, sources) for en, sources in data.items()
            })


# Global singleton instance for MiniSMT translator
_smt_model = None


def get_smt_translator():
    """Lazily initializes and returns the singleton MiniSMT translator instance."""
    global _smt_model
    if _smt_model is None:
        _smt_model = MiniSMTTranslator()
    return _smt_model


def translate_smt(text: str) -> str:
    """Convenience functional wrapper to translate arbitrary Hindi text using Mini-SMT."""
    return get_smt_translator().translate(text)

