"""BERT encoder for extracting BERT embeddings on CPU."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import Config

logger = logging.getLogger(__name__)


class BERTEncoder:
    """Extract BERT embeddings for text."""

    BERT_MODELS = {
        "en": "bert-base-uncased",
        "zh": "bert-base-chinese",
        "jp": "cl-tohoku/bert-base-japanese-v3",
    }

    def __init__(self, language: str, config: Optional[Config] = None, device: str = "cpu"):
        """
        Initialize BERT encoder.

        Args:
            language: Language code (en, zh)
            config: Configuration
            device: Device to use (cpu, cuda)
        """
        self.language = language.lower()
        self.config = config or Config()
        self.device = device

        lang_config = self.config.get_language_config(language)
        self.bert_dim = lang_config.bert_dim

        model_name = self.BERT_MODELS.get(self.language)
        if not model_name:
            raise ValueError(f"No BERT model for language: {language}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """
        Extract BERT embeddings for text.

        Args:
            text: Input text

        Returns:
            BERT embeddings [seq_len, bert_dim]
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state[0]

            return last_hidden.cpu().numpy()

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Extract BERT embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of BERT embeddings
        """
        return [self.encode(text) for text in texts]

    def encode_for_tts(
        self, text: str, seq_len: int
    ) -> np.ndarray:
        """
        Extract and format BERT embeddings for TTS.

        Args:
            text: Input text
            seq_len: Expected sequence length

        Returns:
            BERT embeddings [bert_dim, seq_len]
        """
        embeddings = self.encode(text)

        emb_len = embeddings.shape[0]

        if emb_len < seq_len:
            pad_len = seq_len - emb_len
            embeddings = np.pad(
                embeddings,
                ((0, pad_len), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif emb_len > seq_len:
            embeddings = embeddings[:seq_len]

        embeddings = embeddings.T

        return embeddings.astype(np.float32)

    def __del__(self):
        """Clean up model."""
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
