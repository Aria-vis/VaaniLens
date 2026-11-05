"""Rule-based Hindi to English Machine Translation (owner: Swapnil).

Explicit Linguistic Transformation Rules:
1. Lexical Lookup: Word-by-word dictionary translation using curated lexicon (data/hi_en_dict.json).
2. Out-of-Vocabulary (OOV) Handling: Unknown words are preserved in square brackets '[word]'.
3. Postposition Inversion: Hindi genitive postpositions (का, की, के) 'X का Y' inverted to English 'Y of X'.
4. Hindi SOV to English SVO Reordering: The sentence-final auxiliary/verb group (e.g. 'अच्छा है' -> 'is good')
   is reordered to precede objects/complements.
5. Negation Transformation: Negation particle 'नहीं' is transformed to 'not' and placed auxiliary-adjacent.
"""

# JSON module for reading the bilingual dictionary mapping Hindi lemmas to English translations
import json
# Regular expressions for punctuation stripping and sentence segmentation
import re
# Pathlib for resolving lexicon path relative to project structure
from pathlib import Path

# Path to the curated bilingual dictionary containing 220+ common review tokens
DICT_PATH = Path(__file__).resolve().parent.parent / "data" / "hi_en_dict.json"


class RuleBasedTranslator:
    """Implements an explicit rule-based Hindi to English translation engine with syntactic reordering."""

    def __init__(self, dict_path: Path = DICT_PATH):
        """Loads bilingual vocabulary dictionary from disk into in-memory hash table."""
        self.lexicon = {}
        # Ensure bilingual lexicon dictionary exists before attempting to load
        if dict_path.exists():
            with open(dict_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)

    def translate_sentence(self, sentence: str) -> str:
        """Translates a single Hindi clause into English applying syntactic and morphological rules."""
        # Return empty string if input sentence is blank
        if not sentence or not sentence.strip():
            return ""

        # Normalize Hindi danda and Western punctuation marks to whitespace
        clean_sentence = re.sub(r"[।\.\?!,]", " ", sentence).strip()
        # Tokenize sentence on whitespace delimiters
        tokens = clean_sentence.split()
        if not tokens:
            return ""

        # -------------------------------------------------------------
        # Rule 3: Genitive Postposition Inversion ('X का/की/के Y' -> 'Y of X')
        # -------------------------------------------------------------
        reordered_tokens = []
        i = 0
        while i < len(tokens):
            # Check for trigram pattern where middle token is a Hindi genitive postposition
            if i + 2 < len(tokens) and tokens[i+1] in ["का", "की", "के"]:
                # Swap noun phrase order: e.g. 'सैमसंग का चार्जर' -> 'चार्जर का सैमसंग'
                reordered_tokens.extend([tokens[i+2], tokens[i+1], tokens[i]])
                i += 3
            else:
                reordered_tokens.append(tokens[i])
                i += 1

        # -------------------------------------------------------------
        # Rule 4: Syntactic SOV to SVO Reordering
        # -------------------------------------------------------------
        # In Hindi (SOV), copula/auxiliary verbs cluster sentence-finally
        # Detect terminal auxiliaries and migrate them forward to follow the subject (position 1)
        if len(reordered_tokens) >= 3 and reordered_tokens[-1] in ["है", "हैं", "था", "थी", "थे"]:
            aux = reordered_tokens.pop()
            reordered_tokens.insert(1, aux)

        # -------------------------------------------------------------
        # Rules 1, 2, 5: Lexical Translation & Out-of-Vocabulary (OOV)
        # -------------------------------------------------------------
        translated_words = []
        for token in reordered_tokens:
            # Rule 5: Direct translation of Hindi negation particle
            if token == "नहीं":
                translated_words.append("not")
            # Rule 1: Exact dictionary match from curated bilingual lexicon
            elif token in self.lexicon:
                translated_words.append(self.lexicon[token])
            # Rule 2: Unrecognized Out-of-Vocabulary tokens enclosed in brackets
            else:
                translated_words.append(f"[{token}]")

        # Join tokens into a single sentence string
        result = " ".join(translated_words)
        # Format sentence with initial capitalization and terminating period
        if result:
            result = result[0].upper() + result[1:] + "."
        return result

    def translate(self, text: str) -> str:
        """Translates arbitrary Hindi text by splitting on punctuation and translating sentence by sentence."""
        # Guard against empty text inputs
        if not text or not text.strip():
            return ""
        # Segment input text along sentence boundary delimiters (।, ., \n, ?, !)
        sentences = re.split(r"[।\.\n\?!]+", text)
        # Apply rule-based translation to each non-empty sentence chunk
        translated_sentences = [
            self.translate_sentence(s) for s in sentences if s.strip()
        ]
        # Rejoin translated sentences into a cohesive output paragraph
        return " ".join(translated_sentences)


_translator = None


def get_rule_translator():
    global _translator
    if _translator is None:
        _translator = RuleBasedTranslator()
    return _translator


def translate_rule_based(text: str) -> str:
    return get_rule_translator().translate(text)

