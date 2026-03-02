"""RKNN decoder runtime for NPU inference."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class RKNNDecoder:
    """RKNN decoder for NPU inference."""

    MAX_FRAMES = 65536 // 512
    HOP_LENGTH = 512

    def __init__(
        self,
        model_path: Path,
        config: Optional[Config] = None,
    ):
        """
        Initialize RKNN decoder.

        Args:
            model_path: Path to decoder.rknn
            config: Configuration
        """
        self.model_path = Path(model_path)
        self.config = config or Config()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Decoder model not found: {model_path}")

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

            logger.info(f"Loaded decoder RKNN: {self.model_path}")

        except ImportError as e:
            raise ImportError(
                "RKNNLite not installed. "
                "Install with: pip install rknn-toolkit-lite2==2.3.2"
            ) from e

    def infer(
        self,
        z: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        """
        Run decoder inference.

        Args:
            z: Latent representation [1, 192, frames]
            g: Speaker embedding [1, 256, 1]

        Returns:
            audio: Generated audio [1, 1, samples]
        """
        inputs = [
            z.astype(np.float32),
            g.astype(np.float32),
        ]

        outputs = self.rknn.inference(inputs=inputs)

        audio = outputs[0]

        return audio

    def infer_long(
        self,
        z: np.ndarray,
        g: np.ndarray,
        audio_len: int,
    ) -> np.ndarray:
        """
        Run decoder inference for long sequences.

        Handles sequences longer than MAX_FRAMES by slicing and merging.

        Args:
            z: Latent representation [1, 192, frames]
            g: Speaker embedding [1, 256, 1]
            audio_len: Expected audio length in samples

        Returns:
            audio: Generated audio [audio_len]
        """
        frames = z.shape[2]

        if frames <= self.MAX_FRAMES:
            audio = self.infer(z, g)
            return audio.flatten()[:audio_len]

        sub_audios = []
        slice_num = int(np.ceil(frames / self.MAX_FRAMES))

        for i in range(slice_num):
            start = i * self.MAX_FRAMES
            end = min((i + 1) * self.MAX_FRAMES, frames)

            z_slice = z[:, :, start:end]

            sub_frames = z_slice.shape[2]
            if sub_frames < self.MAX_FRAMES:
                pad_len = self.MAX_FRAMES - sub_frames
                z_slice = np.pad(
                    z_slice,
                    ((0, 0), (0, 0), (0, pad_len)),
                    mode="constant",
                    constant_values=0,
                )

            sub_audio = self.infer(z_slice, g)
            sub_audio = sub_audio.flatten()

            sub_audio_len = sub_frames * self.HOP_LENGTH
            sub_audio = sub_audio[:sub_audio_len]

            sub_audios.append(sub_audio)

        audio = np.concatenate(sub_audios, axis=0)

        return audio[:audio_len]

    def release(self):
        """Release RKNN resources."""
        if hasattr(self, "rknn"):
            self.rknn.release()

    def __del__(self):
        """Clean up."""
        self.release()
