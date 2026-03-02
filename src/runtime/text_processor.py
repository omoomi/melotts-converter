"""Text processor for multiple languages."""

from typing import Dict, List, Tuple, Union

from config import Config

from text.chinese import ChineseProcessor
from text.english import EnglishProcessor


class TextProcessor:
    """Unified text processor for multiple languages."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._processors: Dict[str, Union[EnglishProcessor, ChineseProcessor]] = {}

    def _get_processor(self, language: str):
        """Get or create processor for a language."""
        language = language.lower()

        if language not in self._processors:
            if language in ["en", "english"]:
                self._processors[language] = EnglishProcessor()
            elif language in ["zh", "chinese"]:
                self._processors[language] = ChineseProcessor()
            else:
                raise ValueError(f"Unsupported language: {language}")

        return self._processors[language]

    def process(
        self, text: str, language: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Process text to phone, tone, and language IDs.

        Args:
            text: Input text
            language: Language code (en, zh)

        Returns:
            phones: Phone IDs
            tones: Tone IDs
            lang_ids: Language IDs
        """
        processor = self._get_processor(language)
        return processor.process(text)

    def split_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences."""
        from utils import split_sentences

        return split_sentences(text, language)

    def get_sample_text(self, language: str) -> str:
        """Get sample text for a language."""
        processor = self._get_processor(language)
        return processor.get_sample_text()
