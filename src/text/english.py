"""English text processing."""

import re
from typing import List, Tuple

import inflect
from g2p_en import G2p
from num2words import num2words

from .symbols import ENGLISH_SYMBOLS, symbol_to_id

_inflect = inflect.engine()
_g2p = None


def _get_g2p():
    """Get or create G2p instance."""
    global _g2p
    if _g2p is None:
        _g2p = G2p()
    return _g2p


def normalize_numbers(text: str) -> str:
    """Normalize numbers in text to words."""
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)

    def replace_number(match):
        num = match.group(0)
        if "." in num:
            return num2words(float(num), lang="en")
        else:
            return num2words(int(num), lang="en")

    text = re.sub(r"\b\d+\.?\d*\b", replace_number, text)
    return text


def normalize_abbreviations(text: str) -> str:
    """Normalize common abbreviations."""
    abbreviations = {
        "mr.": "mister",
        "mrs.": "misses",
        "ms.": "miss",
        "dr.": "doctor",
        "prof.": "professor",
        "vs.": "versus",
        "etc.": "etcetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "inc.": "incorporated",
        "ltd.": "limited",
        "corp.": "corporation",
        "jr.": "junior",
        "sr.": "senior",
    }

    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)

    return text


def normalize_text(text: str) -> str:
    """Normalize text for processing."""
    text = text.lower()
    text = normalize_numbers(text)
    text = normalize_abbreviations(text)
    text = re.sub(r"[^\w\s\.\!\?\,\-\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_phonemes(text: str) -> List[str]:
    """Convert text to phonemes."""
    g2p = _get_g2p()
    phonemes = g2p(text)
    return phonemes


def phonemes_to_ids(phonemes: List[str]) -> List[int]:
    """Convert phonemes to IDs."""
    sym2id = symbol_to_id("english")
    ids = []

    for phoneme in phonemes:
        phoneme = phoneme.strip()

        if not phoneme:
            continue

        if phoneme in sym2id:
            ids.append(sym2id[phoneme])
        else:
            ids.append(sym2id.get(" ", sym2id["_"]))

    return ids


class EnglishProcessor:
    """English text processor."""

    def __init__(self):
        self.symbols = ENGLISH_SYMBOLS
        self.sym2id = symbol_to_id("english")

    def process(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Process text to phone, tone, and language IDs.

        Returns:
            phones: Phone IDs
            tones: Tone IDs (all 0 for English)
            lang_ids: Language IDs (all 0 for English)
        """
        text = normalize_text(text)
        phonemes = text_to_phonemes(text)
        phones = phonemes_to_ids(phonemes)

        tones = [0] * len(phones)
        lang_ids = [0] * len(phones)

        return phones, tones, lang_ids

    def get_sample_text(self) -> str:
        """Get sample text for testing."""
        return "Hello, this is a test of the text to speech system."
