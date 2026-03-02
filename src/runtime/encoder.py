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
        g: np.ndarray,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run encoder inference.

        Args:
            phones: Phone IDs [1, seq_len]
            tones: Tone IDs [1, seq_len]
            lang_ids: Language IDs [1, seq_len]
            bert: BERT embeddings [1, bert_dim, seq_len]
            g: Speaker embedding [1, 256, 1]
            noise_scale: Noise scale
            length_scale: Length scale
            noise_scale_w: Duration noise scale
            sdp_ratio: SDP ratio

        Returns:
            z: Latent representation [1, 192, audio_frames]
            y_mask: Mask [1, 1, audio_frames]
            audio_len: Audio length
        """
        noise_scale_arr = np.array([noise_scale], dtype=np.float32)
        length_scale_arr = np.array([length_scale], dtype=np.float32)
        noise_scale_w_arr = np.array([noise_scale_w], dtype=np.float32)
        sdp_ratio_arr = np.array([sdp_ratio], dtype=np.float32)

        inputs = [
            phones.astype(np.int32),
            tones.astype(np.int32),
            lang_ids.astype(np.int32),
            bert.astype(np.float32),
            g.astype(np.float32),
            noise_scale_arr,
            length_scale_arr,
            noise_scale_w_arr,
            sdp_ratio_arr,
        ]

        outputs = self.rknn.inference(inputs=inputs)

        z = outputs[0]
        y_mask = outputs[1]
        audio_len = int(outputs[2][0])

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
