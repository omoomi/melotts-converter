"""Configuration management for rknn-converter."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from packaging.version import Version


@dataclass
class LanguageConfig:
    """Configuration for a specific language."""

    name: str
    huggingface_repo: str
    voices: List[str]
    sample_rate: int = 44100
    bert_dim: int = 1024
    num_languages: int = 1
    num_tones: int = 1


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

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
    """Configuration for RKNN conversion."""

    target_platform: str = "rk3588"
    float_dtype: str = "float16"
    optimization_level: int = 3
    do_quantization: bool = False
    quantized_dtype: str = "w8a8"
    encoder_max_frames: int = 2048
    
    dynamic_input_profiles: List[Dict] = field(default_factory=lambda: [
        {"seq_len": 32, "frames": 512},
        {"seq_len": 64, "frames": 1024},
        {"seq_len": 128, "frames": 2048},
    ])


class Config:
    """Main configuration class."""

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
        self.models_dir = models_dir or Path.cwd() / "models"
        self.cache_dir = cache_dir or Path.home() / ".cache" / "rknn-converter"
        self.model_config = ModelConfig()
        self.rknn_config = RKNNConfig()

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_language_config(self, language: str) -> LanguageConfig:
        """Get configuration for a specific language."""
        lang_key = language.lower()
        if lang_key not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {list(self.SUPPORTED_LANGUAGES.keys())}"
            )
        return self.SUPPORTED_LANGUAGES[lang_key]

    def get_model_dir(self, language: str) -> Path:
        """Get the model directory for a language."""
        return self.models_dir / language.lower()

    def get_voice_path(self, language: str, voice: str) -> Path:
        """Get the path to a voice embedding file."""
        return self.get_model_dir(language) / "voices" / f"{voice}.bin"

    @staticmethod
    def get_default_config() -> "Config":
        """Get default configuration."""
        return Config()
