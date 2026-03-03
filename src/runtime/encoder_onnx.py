"""ONNX encoder runtime for CPU inference."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class ONNXEncoder:
    """ONNX encoder for CPU inference."""

    def __init__(
        self,
        model_path: Path,
        config: Optional[Config] = None,
    ):
        """
        Initialize ONNX encoder.

        Args:
            model_path: Path to encoder.onnx
            config: Configuration
        """
        self.model_path = Path(model_path)
        self.config = config or Config()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {model_path}")

        self._init_onnx()

    def _init_onnx(self):
        """Initialize ONNX runtime."""
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )

            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

            logger.info(f"Loaded encoder ONNX: {self.model_path}")
            logger.debug(f"Inputs: {self.input_names}")
            logger.debug(f"Outputs: {self.output_names}")

        except ImportError as e:
            raise ImportError(
                "ONNXRuntime not installed. "
                "Install with: pip install onnxruntime"
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
        Run encoder inference with ONNX runtime.

        Args:
            phones: Phone IDs [1, seq_len]
            tones: Tone IDs [1, seq_len]
            lang_ids: Language IDs [1, seq_len]
            bert: BERT embeddings [1, bert_dim, seq_len]
            ja_bert: Japanese BERT embeddings [1, 768, seq_len]
            g: Speaker embedding [1, 256, 1]
            noise_scale: Noise scale for flow (default: 0.667)
            length_scale: Length scale for duration (default: 1.0)

        Returns:
            z: Latent representation [1, 192, 2048]
            y_mask: Mask [1, 1, 2048]
            audio_len: Audio length in samples (NOTE: may be incorrect)
        """
        seq_len = phones.shape[1]

        noise_scale_arr = np.array([noise_scale], dtype=np.float32)
        length_scale_arr = np.array([length_scale], dtype=np.float32)

        phones_input = phones.astype(np.int64)
        tones_input = tones.astype(np.int64)
        lang_ids_input = lang_ids.astype(np.int64)
        bert_input = bert.astype(np.float32)
        ja_bert_input = ja_bert.astype(np.float32)
        g_input = g.astype(np.float32)

        inputs = {
            'phones': phones_input,
            'tones': tones_input,
            'lang_ids': lang_ids_input,
            'bert': bert_input,
            'ja_bert': ja_bert_input,
            'g': g_input,
            'noise_scale': noise_scale_arr,
            'length_scale': length_scale_arr,
        }

        outputs = self.session.run(self.output_names, inputs)

        z = outputs[0]
        y_mask = outputs[1]
        audio_len = int(outputs[2][0]) if len(outputs) > 2 else 0

        current_frames = z.shape[2]
        target_frames = 2048 if current_frames > 1024 else 1024

        if current_frames < target_frames:
            pad_len = target_frames - current_frames
            z = np.pad(z, ((0, 0), (0, 0), (0, pad_len)), mode='constant', constant_values=0)
            y_mask = np.pad(y_mask, ((0, 0), (0, 0), (0, pad_len)), mode='constant', constant_values=0)
        elif current_frames > target_frames:
            z = z[:, :, :target_frames]
            y_mask = y_mask[:, :, :target_frames]

        logger.debug(f"Encoder inference complete: seq_len={seq_len}, audio_len={audio_len}, z_shape={z.shape}")

        return z, y_mask, audio_len

    def release(self):
        """Release ONNX resources."""
        if hasattr(self, 'session'):
            del self.session

    def __del__(self):
        """Clean up."""
        self.release()
