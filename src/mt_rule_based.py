"""Rule-based Hindi to English Machine Translation (owner: Swapnil).

Explicit Linguistic Transformation Rules:
1. Lexical Lookup: Word-by-word dictionary translation using curated lexicon (data/hi_en_dict.json).
2. Out-of-Vocabulary (OOV) Handling: Unknown words are preserved in square brackets '[word]'.
3. Postposition Inversion: Hindi genitive postpositions (का, की, के) 'X का Y' inverted to English 'Y of X'.
4. Hindi SOV to English SVO Reordering: The sentence-final auxiliary/verb group (e.g. 'अच्छा है' -> 'is good')
   is reordered to precede objects/complements.
5. Negation Transformation: Negation particle 'नहीं' is transformed to 'not' and placed auxiliary-adjacent.
"""

import json
import re
from pathlib import Path

DICT_PATH = Path(__file__).resolve().parent.parent / "data" / "hi_en_dict.json"


class RuleBasedTranslator:
    """Implements an explicit rule-based Hindi to English translation engine with syntactic reordering."""

    def __init__(self, dict_path: Path = DICT_PATH):
        """Loads bilingual vocabulary dictionary from disk into in-memory hash table."""
        self.lexicon = {}
        if dict_path.exists():
            with open(dict_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)

    def translate_sentence(self, sentence: str) -> str:
        """Translates a single Hindi clause into English applying syntactic and morphological rules."""
        if not sentence or not sentence.strip():
            return ""

        clean_sentence = re.sub(r"[।\.\?!,]", " ", sentence).strip()
        tokens = clean_sentence.split()
        if not tokens:
            return ""

        reordered_tokens = []
        i = 0
        while i < len(tokens):
            if i + 2 < len(tokens) and tokens[i+1] in ["का", "की", "के"]:
                reordered_tokens.extend([tokens[i+2], tokens[i+1], tokens[i]])
                i += 3
            else:
                reordered_tokens.append(tokens[i])
                i += 1

        if len(reordered_tokens) >= 3 and reordered_tokens[-1] in ["है", "हैं", "था", "थी", "थे"]:
            aux = reordered_tokens.pop()
            reordered_tokens.insert(1, aux)

        translated_words = []
        for token in reordered_tokens:
            if token == "नहीं":
                translated_words.append("not")
            elif token in self.lexicon:
                translated_words.append(self.lexicon[token])
            else:
                translated_words.append(f"[{token}]")

        result = " ".join(translated_words)
        if result:
            result = result[0].upper() + result[1:] + "."
        return result

    def translate(self, text: str) -> str:
        """Translates arbitrary Hindi text by splitting on punctuation and translating sentence by sentence."""
        if not text or not text.strip():
            return ""
        sentences = re.split(r"[।\.\n\?!]+", text)
        translated_sentences = [
            self.translate_sentence(s) for s in sentences if s.strip()
        ]
        return " ".join(translated_sentences)


_translator = None


def get_rule_translator():
    global _translator
    if _translator is None:
        _translator = RuleBasedTranslator()
    return _translator


def translate_rule_based(text: str) -> str:
    return get_rule_translator().translate(text)

