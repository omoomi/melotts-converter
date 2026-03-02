"""Chinese text processing with mixed English support."""

import logging
import re
from typing import List, Tuple

import jieba
from num2words import num2words
from pypinyin import Style, pinyin

from .symbols import CHINESE_SYMBOLS, ENGLISH_SYMBOLS, symbol_to_id

jieba.setLogLevel(20)
logger = logging.getLogger(__name__)


def normalize_numbers_zh(text: str) -> str:
    """Normalize Chinese numbers to words."""
    def replace_number(match):
        num_str = match.group(0)
        try:
            if "." in num_str:
                return num2words(float(num_str), lang="zh")
            else:
                return num2words(int(num_str), lang="zh")
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert number '{num_str}': {e}")
            return num_str

    text = re.sub(r"\d+\.?\d*", replace_number, text)
    return text


def is_chinese_char(char: str) -> bool:
    """Check if character is Chinese."""
    return "\u4e00" <= char <= "\u9fff"


def is_english_char(char: str) -> bool:
    """Check if character is English."""
    return char.isalpha() and char.isascii()


def segment_mixed_text(text: str) -> List[Tuple[str, str]]:
    """
    Segment mixed Chinese/English text.
    
    Returns:
        List of (segment, language) tuples
    """
    segments = []
    current_segment = ""
    current_lang = None

    for char in text:
        if is_chinese_char(char):
            if current_lang == "zh":
                current_segment += char
            else:
                if current_segment:
                    segments.append((current_segment, current_lang or "zh"))
                current_segment = char
                current_lang = "zh"
        elif is_english_char(char):
            if current_lang == "en":
                current_segment += char
            else:
                if current_segment:
                    segments.append((current_segment, current_lang or "zh"))
                current_segment = char
                current_lang = "en"
        else:
            if current_segment:
                segments.append((current_segment, current_lang or "zh"))
            if char.strip():
                segments.append((char, "punct"))
            current_segment = ""
            current_lang = None

    if current_segment:
        segments.append((current_segment, current_lang or "zh"))

    return segments


def chinese_to_pinyin(text: str) -> List[Tuple[str, int]]:
    """
    Convert Chinese text to pinyin with tones.
    
    Returns:
        List of (pinyin, tone) tuples
    """
    pinyin_list = pinyin(text, style=Style.TONE3)

    results = []
    for py, char in zip(pinyin_list, text):
        if not is_chinese_char(char):
            continue

        py_str = py[0]

        tone = 0
        if py_str and py_str[-1].isdigit():
            tone = int(py_str[-1])
            py_str = py_str[:-1]

        results.append((py_str, tone))

    return results


def text_to_phonemes_zh(text: str) -> List[Tuple[str, int]]:
    """Convert Chinese text to phonemes with tones."""
    text = normalize_numbers_zh(text)
    
    words = list(jieba.cut(text))
    
    phonemes = []
    for word in words:
        if not word.strip():
            continue

        if all(is_chinese_char(c) for c in word):
            pinyin_list = chinese_to_pinyin(word)
            phonemes.extend(pinyin_list)
        else:
            phonemes.append((word, 0))

    return phonemes


class ChineseProcessor:
    """Chinese text processor with mixed English support."""

    def __init__(self):
        self.zh_symbols = CHINESE_SYMBOLS
        self.en_symbols = ENGLISH_SYMBOLS
        self.zh_sym2id = symbol_to_id("chinese")
        self.en_sym2id = symbol_to_id("english")

    def process(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Process mixed Chinese/English text.

        Returns:
            phones: Phone IDs
            tones: Tone IDs
            lang_ids: Language IDs (3=Chinese, 0=English)
        """
        phones = []
        tones = []
        lang_ids = []

        segments = segment_mixed_text(text)

        for segment, lang in segments:
            if lang == "zh":
                phonemes = text_to_phonemes_zh(segment)
                for py, tone in phonemes:
                    if py in self.zh_sym2id:
                        phones.append(self.zh_sym2id[py])
                        tones.append(tone)
                        lang_ids.append(3)
                    else:
                        for char in py:
                            if char in self.zh_sym2id:
                                phones.append(self.zh_sym2id[char])
                                tones.append(0)
                                lang_ids.append(3)
            elif lang == "en":
                from .english import normalize_text, text_to_phonemes

                normalized = normalize_text(segment)
                en_phonemes = text_to_phonemes(normalized)

                for phoneme in en_phonemes:
                    phoneme = phoneme.strip()
                    if phoneme in self.en_sym2id:
                        phones.append(self.en_sym2id[phoneme])
                        tones.append(0)
                        lang_ids.append(0)
            elif lang == "punct":
                if segment in self.zh_sym2id:
                    phones.append(self.zh_sym2id[segment])
                    tones.append(0)
                    lang_ids.append(3)

        return phones, tones, lang_ids

    def get_sample_text(self) -> str:
        """Get sample text for testing."""
        return "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
