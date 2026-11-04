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

import pickle
import re
from pathlib import Path
from collections import defaultdict

MODEL_CACHE = Path(__file__).resolve().parent.parent / "models" / "smt_ibm1.pkl"


class MiniSMTTranslator:
    def __init__(self, model_path: Path = MODEL_CACHE):
        self.translation_table = defaultdict(lambda: defaultdict(float))
        self.model_path = model_path
        if model_path.exists():
            self.load(model_path)

    def train(self, parallel_pairs: list[tuple[str, str]], iterations: int = 5):
        """Trains IBM Model 1 lexical translation probabilities using EM.
        Args:
            parallel_pairs: List of (hindi_sentence, english_sentence)
            iterations: Number of EM iterations.
        """
        try:
            from nltk.translate import AlignedSent, IBMModel1
            aligned_corpus = []
            for hi_sent, en_sent in parallel_pairs:
                hi_tokens = [w.lower() for w in re.split(r"\s+", hi_sent) if w]
                en_tokens = [w.lower() for w in re.split(r"\s+", en_sent) if w]
                if hi_tokens and en_tokens:
                    # In NLTK IBMModel1: words = target (en), mots = source (hi)
                    aligned_corpus.append(AlignedSent(words=en_tokens, mots=hi_tokens))

            ibm1 = IBMModel1(aligned_corpus, iterations)
            self.translation_table = ibm1.translation_table
            self.save(self.model_path)
            print(f"Trained IBM Model 1 on {len(aligned_corpus)} pairs.")
        except Exception as e:
            print(f"NLTK IBMModel1 training error: {e}. Using co-occurrence fallback.")
            self._train_cooccurrence(parallel_pairs)

    def _train_cooccurrence(self, pairs):
        counts = defaultdict(lambda: defaultdict(float))
        for hi_sent, en_sent in pairs:
            hi_tokens = [w.lower() for w in re.split(r"\s+", hi_sent) if w]
            en_tokens = [w.lower() for w in re.split(r"\s+", en_sent) if w]
            for h in hi_tokens:
                for e in en_tokens:
                    counts[h][e] += 1.0

        for h, targets in counts.items():
            total = sum(targets.values())
            for e, count in targets.items():
                self.translation_table[e][h] = count / total
        self.save(self.model_path)

    def translate_sentence(self, sentence: str) -> str:
        if not sentence or not sentence.strip():
            return ""
        tokens = [w.lower() for w in re.split(r"\s+", sentence) if w]
        output = []

        for hi_word in tokens:
            # Find English word with max t(en | hi)
            best_en = None
            best_prob = 0.0
            for en_word, hi_dict in self.translation_table.items():
                prob = hi_dict.get(hi_word, 0.0)
                if prob > best_prob:
                    best_prob = prob
                    best_en = en_word

            if best_en and best_prob > 0.01:
                output.append(best_en)
            else:
                output.append(f"[{hi_word}]")

        res = " ".join(output)
        return res.capitalize() if res else ""

    def translate(self, text: str) -> str:
        if not text:
            return ""
        sentences = re.split(r"[।\.\n\?!]+", text)
        translated = [self.translate_sentence(s) for s in sentences if s.strip()]
        return ". ".join(translated) + ("." if translated else "")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert defaultdict to normal dict for pickling
        plain_dict = {en: dict(sources) for en, sources in self.translation_table.items()}
        with open(path, "wb") as f:
            pickle.dump(plain_dict, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.translation_table = defaultdict(lambda: defaultdict(float), {
                en: defaultdict(float, sources) for en, sources in data.items()
            })


_smt_model = None


def get_smt_translator():
    global _smt_model
    if _smt_model is None:
        _smt_model = MiniSMTTranslator()
    return _smt_model


def translate_smt(text: str) -> str:
    return get_smt_translator().translate(text)

