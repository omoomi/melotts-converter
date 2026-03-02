"""Configuration management for melotts-converter.

This module provides configuration classes for managing model settings,
language configurations, and RKNN conversion parameters.

Example:
    >>> from config import Config
    >>> config = Config()
    >>> lang_config = config.get_language_config('en')
    >>> print(lang_config.name)
    'English'
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from packaging.version import Version


@dataclass
class LanguageConfig:
    """Configuration for a specific language model.

    Attributes:
        name: Human-readable language name.
        huggingface_repo: HuggingFace repository ID for the model.
        voices: List of available voice names for this language.
        sample_rate: Audio sample rate in Hz (default: 44100).
        bert_dim: BERT embedding dimension (default: 1024).
        num_languages: Number of supported languages (default: 1).
        num_tones: Number of tone levels (default: 1).

    Example:
        >>> config = LanguageConfig(
        ...     name="English",
        ...     huggingface_repo="myshell-ai/MeloTTS-English",
        ...     voices=["en-us", "en-br", "en-india"]
        ... )
    """

    name: str
    huggingface_repo: str
    voices: List[str]
    sample_rate: int = 44100
    bert_dim: int = 1024
    num_languages: int = 1
    num_tones: int = 1


@dataclass
class ModelConfig:
    """Configuration for model architecture parameters.

    Attributes:
        inter_channels: Internal channels for the model (default: 192).
        hidden_channels: Hidden layer dimensions (default: 192).
        filter_channels: Filter dimensions for convolutional layers (default: 768).
        n_heads: Number of attention heads (default: 2).
        n_layers: Number of encoder layers (default: 6).
        kernel_size: Kernel size for convolutions (default: 3).
        p_dropout: Dropout probability (default: 0.1).
        resblock: Residual block type ('1' or '2', default: '1').
        resblock_kernel_sizes: Kernel sizes for residual blocks.
        resblock_dilation_sizes: Dilation rates for residual blocks.
        upsample_rates: Upsampling rates for the decoder.
        upsample_initial_channel: Initial channel count for upsampling (default: 512).
        upsample_kernel_sizes: Kernel sizes for upsampling layers.
        n_layers_trans_flow: Number of transformer flow layers (default: 6).
        gin_channels: Speaker conditioning channels (default: 256).
        n_speakers: Number of speakers in the model (default: 256).
    """

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    n_layers_trans_flow: int = 6
    gin_channels: int = 256
    n_speakers: int = 256


@dataclass
class RKNNConfig:
    """Configuration for RKNN conversion and optimization.

    Attributes:
        target_platform: Target NPU platform ('rk3588' or 'rk3576', default: 'rk3588').
        float_dtype: Floating-point data type ('float16' or 'float32', default: 'float16').
        optimization_level: RKNN optimization level (0-3, default: 3).
        do_quantization: Whether to apply quantization (default: False).
        quantized_dtype: Quantization data type ('w8a8' for 8-bit, default: 'w8a8').
        encoder_max_frames: Maximum number of frames for encoder (default: 2048).
        dynamic_input_profiles: List of dynamic input profiles for RKNN conversion.
            Each profile defines input dimensions for different sequence lengths.
    """

    target_platform: str = "rk3588"
    float_dtype: str = "float16"
    optimization_level: int = 3
    do_quantization: bool = False
    quantized_dtype: str = "w8a8"
    encoder_max_frames: int = 2048

    dynamic_input_profiles: List[Dict] = field(default_factory=lambda: [
        {"seq_len": 64, "frames": 1024},
        {"seq_len": 128, "frames": 2048},
    ])


class Config:
    """Main configuration class for melotts-converter.

    This class manages all configuration settings including language models,
    model architecture parameters, and RKNN conversion settings.

    Attributes:
        models_dir: Directory for storing model files (default: ./models).
        cache_dir: Directory for caching downloads (default: ~/.cache/melotts-converter).
        model_config: Model architecture configuration.
        rknn_config: RKNN conversion configuration.
        SUPPORTED_LANGUAGES: Dictionary of supported language configurations.
        REQUIRED_ONNX_VERSION: Required ONNX version (1.18.0).
        REQUIRED_RKNN_VERSION: Required RKNN version (2.3.2).

    Example:
        >>> config = Config()
        >>> config.models_dir
        PosixPath('models')
        >>> config.get_language_config('en').name
        'English'
    """

    SUPPORTED_LANGUAGES = {
        "en": LanguageConfig(
            name="English",
            huggingface_repo="myshell-ai/MeloTTS-English",
            voices=["en-us", "en-br", "en-india", "en-au", "en-default"],
            sample_rate=44100,
            bert_dim=1024,
            num_languages=4,
            num_tones=4,
        ),
        "zh": LanguageConfig(
            name="Chinese",
            huggingface_repo="myshell-ai/MeloTTS-Chinese",
            voices=["zh"],
            sample_rate=44100,
            bert_dim=1024,
            num_languages=3,
            num_tones=5,
        ),
        "jp": LanguageConfig(
            name="Japanese",
            huggingface_repo="myshell-ai/MeloTTS-Japanese",
            voices=["jp"],
            sample_rate=44100,
            bert_dim=768,
            num_languages=1,
            num_tones=1,
        ),
    }

    REQUIRED_ONNX_VERSION = Version("1.18.0")
    REQUIRED_RKNN_VERSION = Version("2.3.2")

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize configuration.

        Args:
            models_dir: Directory for storing model files. If None, uses ./models.
            cache_dir: Directory for caching downloads. If None, uses ~/.cache/melotts-converter.
        """
        self.models_dir = models_dir or Path.cwd() / "models"
        self.cache_dir = cache_dir or Path.home() / ".cache" / "melotts-converter"
        self.model_config = ModelConfig()
        self.rknn_config = RKNNConfig()

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_language_config(self, language: str) -> LanguageConfig:
        """Get configuration for a specific language.

        Args:
            language: Language code ('en', 'zh', 'jp').

        Returns:
            LanguageConfig object for the specified language.

        Raises:
            ValueError: If the language is not supported.

        Example:
            >>> config = Config()
            >>> lang_config = config.get_language_config('en')
            >>> print(lang_config.name)
            'English'
        """
        lang_key = language.lower()
        if lang_key not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {list(self.SUPPORTED_LANGUAGES.keys())}"
            )
        return self.SUPPORTED_LANGUAGES[lang_key]

    def get_model_dir(self, language: str) -> Path:
        """Get the model directory for a language.

        Args:
            language: Language code ('en', 'zh', 'jp').

        Returns:
            Path object pointing to the model directory.

        Example:
            >>> config = Config()
            >>> config.get_model_dir('en')
            PosixPath('models/en')
        """
        return self.models_dir / language.lower()

    def get_voice_path(self, language: str, voice: str) -> Path:
        """Get the path to a voice embedding file.

        Args:
            language: Language code ('en', 'zh', 'jp').
            voice: Voice name (e.g., 'en-us', 'zh').

        Returns:
            Path object pointing to the voice embedding file.

        Example:
            >>> config = Config()
            >>> config.get_voice_path('en', 'en-us')
            PosixPath('models/en/voices/en-us.bin')
        """
        return self.get_model_dir(language) / "voices" / f"{voice}.bin"

    @staticmethod
    def get_default_config() -> "Config":
        """Get default configuration instance.

        Returns:
            A new Config object with default settings.

        Example:
            >>> config = Config.get_default_config()
            >>> config.models_dir
            PosixPath('models')
        """
        return Config()
