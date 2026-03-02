"""Phoneme symbols for different languages."""

from typing import List

# English symbols
ENGLISH_SYMBOLS = [
    "_", "-", "!", "?", ".", "'", " ", 
    "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2",
    "AH", "AH0", "AH1", "AH2", "AO", "AO0", "AO1", "AO2",
    "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2",
    "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2",
    "F", "G", "HH", "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
    "OW", "OW0", "OW1", "OW2", "OY", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
]

# Chinese symbols (pinyin + special)
CHINESE_SYMBOLS = [
    "_", "-", "!", "?", ".", "'", " ",
    # Initials
    "b", "p", "m", "f", "d", "t", "n", "l", 
    "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
    # Finals
    "a", "o", "e", "i", "u", "v", "er",
    "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iu", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ue", "van", "vn",
    # Tones
    "1", "2", "3", "4", "5",
]

# Japanese symbols
JAPANESE_SYMBOLS = [
    "_", "-", "!", "?", ".", "'", " ",
    # Hiragana/Katakana sounds represented as romanized
    "a", "i", "u", "e", "o",
    "ka", "ki", "ku", "ke", "ko",
    "sa", "shi", "su", "se", "so",
    "ta", "chi", "tsu", "te", "to",
    "na", "ni", "nu", "ne", "no",
    "ha", "hi", "fu", "he", "ho",
    "ma", "mi", "mu", "me", "mo",
    "ya", "yu", "yo",
    "ra", "ri", "ru", "re", "ro",
    "wa", "wo", "n",
    "ga", "gi", "gu", "ge", "go",
    "za", "ji", "zu", "ze", "zo",
    "da", "de", "do",
    "ba", "bi", "bu", "be", "bo",
    "pa", "pi", "pu", "pe", "po",
]

PAD = "_"
PUNCTUATION = ["!", "?", ".", "'", " ", "-", ",", ";", ":"]


def get_symbols(language: str) -> List[str]:
    """Get symbol list for a language."""
    language = language.lower()
    
    if language in ["en", "english"]:
        return ENGLISH_SYMBOLS
    elif language in ["zh", "chinese"]:
        return CHINESE_SYMBOLS
    elif language in ["jp", "japanese"]:
        return JAPANESE_SYMBOLS
    else:
        raise ValueError(f"Unknown language: {language}")


def get_num_symbols(language: str) -> int:
    """Get number of symbols for a language."""
    return len(get_symbols(language))


def symbol_to_id(language: str) -> dict:
    """Get symbol to ID mapping for a language."""
    return {s: i for i, s in enumerate(get_symbols(language))}
