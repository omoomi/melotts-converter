"""RKNN encoder runtime for NPU inference."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import Config
from utils import load_binary

logger = logging.getLogger(__name__)


class RNNEncoder:
    """RKNN encoder for NPU inference."""

    def __init__(
        self,
        model_path: Path,
        config: Optional[Config] = None,
    ):
        """
        Initialize RKNN encoder.

        Args:
            model_path: Path to encoder.rknn
            config: Configuration
        """
        self.model_path = Path(model_path)
        self.config = config or Config()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {model_path}")

        self._init_rknn()

    def _init_rknn(self):
        """Initialize RKNN runtime."""
        try:
            from rknnlite.api import RKNNLite

            self.rknn = RKNNLite()

            ret = self.rknn.load_rknn(str(self.model_path))
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model: {self.model_path}")

            ret = self.rknn.init_runtime()
            if ret != 0:
                raise RuntimeError("Failed to init RKNN runtime")

            logger.info(f"Loaded encoder RKNN: {self.model_path}")

        except ImportError as e:
            raise ImportError(
                "RKNNLite not installed. "
                "Install with: pip install rknn-toolkit-lite2==2.3.2"
            ) from e

    def infer(
        self,
        phones: np.ndarray,
        tones: np.ndarray,
        lang_ids: np.ndarray,
        bert: np.ndarray,
        ja_bert: np.ndarray,
        g: np.ndarray,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run encoder inference with NCHW-compliant inputs.

        Inputs are automatically reshaped to NCHW format [Batch, Channel, Height]
        for RKNN NPU compatibility.

        Args:
            phones: Phone IDs [1, seq_len] - will be reshaped to [1, 1, seq_len]
            tones: Tone IDs [1, seq_len] - will be reshaped to [1, 1, seq_len]
            lang_ids: Language IDs [1, seq_len] - will be reshaped to [1, 1, seq_len]
            bert: BERT embeddings [1, bert_dim, seq_len] - already NCHW
            ja_bert: Japanese BERT embeddings [1, 768, seq_len] - already NCHW
            g: Speaker embedding [1, 256, 1] - already NCHW
            noise_scale: Noise scale for flow (default: 0.667)
            length_scale: Length scale for duration (default: 1.0)

        Returns:
            z: Latent representation [1, 192, audio_frames]
            y_mask: Mask [1, 1, audio_frames]
            audio_len: Audio length in samples
        """
        seq_len = phones.shape[1]

        noise_scale_arr = np.array([noise_scale], dtype=np.float32).reshape(1, 1, 1)
        length_scale_arr = np.array([length_scale], dtype=np.float32).reshape(1, 1, 1)

        phones_nchw = phones.astype(np.int64).reshape(1, 1, seq_len)
        tones_nchw = tones.astype(np.int64).reshape(1, 1, seq_len)
        lang_ids_nchw = lang_ids.astype(np.int64).reshape(1, 1, seq_len)

        bert_nchw = bert.astype(np.float32)
        ja_bert_nchw = ja_bert.astype(np.float32)
        g_nchw = g.astype(np.float32)

        inputs = [
            phones_nchw,
            tones_nchw,
            lang_ids_nchw,
            bert_nchw,
            ja_bert_nchw,
            g_nchw,
            noise_scale_arr,
            length_scale_arr,
        ]

        outputs = self.rknn.inference(inputs=inputs)

        z = outputs[0]
        y_mask = outputs[1]
        audio_len = int(outputs[2][0])

        logger.debug(f"Encoder inference complete: seq_len={seq_len}, audio_len={audio_len}")

        return z, y_mask, audio_len

    def load_voice_embedding(self, voice_path: Path) -> np.ndarray:
        """Load speaker embedding from file."""
        emb = load_binary(voice_path, dtype=np.float32)
        return emb.reshape(1, 256, 1)

    def release(self):
        """Release RKNN resources."""
        if hasattr(self, "rknn"):
            self.rknn.release()

    def __del__(self):
        """Clean up."""
        self.release()
