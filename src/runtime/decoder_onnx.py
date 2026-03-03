"""ONNX decoder runtime for CPU inference."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class ONNXDecoder:
    """ONNX decoder for CPU inference."""

    MAX_FRAMES = 65536 // 512
    HOP_LENGTH = 512

    def __init__(
        self,
        model_path: Path,
        config: Optional[Config] = None,
    ):
        """
        Initialize ONNX decoder.

        Args:
            model_path: Path to decoder.onnx
            config: Configuration
        """
        self.model_path = Path(model_path)
        self.config = config or Config()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Decoder model not found: {model_path}")

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

            logger.info(f"Loaded decoder ONNX: {self.model_path}")
            logger.debug(f"Inputs: {self.input_names}")
            logger.debug(f"Outputs: {self.output_names}")

        except ImportError as e:
            raise ImportError(
                "ONNXRuntime not installed. "
                "Install with: pip install onnxruntime"
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
        inputs = {
            'z': z.astype(np.float32),
            'g': g.astype(np.float32),
        }

        outputs = self.session.run(self.output_names, inputs)

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
        """Release ONNX resources."""
        if hasattr(self, 'session'):
            del self.session

    def __del__(self):
        """Clean up."""
        self.release()
