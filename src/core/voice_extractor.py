"""Voice embedding extractor."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from rich.console import Console

from config import Config
from utils import ensure_dir, save_binary, save_json, load_binary

console = Console()
logger = logging.getLogger(__name__)


class VoiceExtractor:
    """Extract speaker embeddings from MeloTTS models."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def extract_voices(
        self,
        language: str,
        checkpoint_path: Optional[Path] = None,
        quiet: bool = False,
    ) -> Dict[str, Path]:
        """
        Extract speaker embeddings from model checkpoint.

        Args:
            language: Language code (en, zh)
            checkpoint_path: Path to checkpoint.pth (optional, will use default if not provided)
            quiet: Suppress output

        Returns:
            Dictionary mapping voice names to embedding file paths
        """
        lang_config = self.config.get_language_config(language)
        model_dir = self.config.get_model_dir(language)

        if checkpoint_path is None:
            checkpoint_path = model_dir / "checkpoint.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if not quiet:
            console.print(f"[bold blue]Extracting voice embeddings for {lang_config.name}...[/bold blue]")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model" not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'model' key")

        model_state = checkpoint["model"]

        emb_g_key = "emb_g.weight"
        if emb_g_key not in model_state:
            raise ValueError(f"Invalid checkpoint: missing '{emb_g_key}' key")

        embeddings = model_state[emb_g_key].numpy()

        if not quiet:
            console.print(f"Found embeddings shape: {embeddings.shape}")

        voices_dir = model_dir / "voices"
        ensure_dir(voices_dir)

        voice_paths = {}

        for idx, voice_name in enumerate(lang_config.voices):
            if idx >= embeddings.shape[0]:
                if not quiet:
                    console.print(f"[yellow]Warning: Voice index {idx} out of range[/yellow]")
                continue

            embedding = embeddings[idx]
            voice_path = voices_dir / f"{voice_name}.bin"

            save_binary(embedding, voice_path)
            voice_paths[voice_name] = voice_path

            if not quiet:
                console.print(f"  [green]✓[/green] Extracted {voice_name} ({embedding.shape})")

        self._save_voice_metadata(language, voice_paths, embeddings.shape)

        if not quiet:
            console.print(f"[bold green]✓ Extracted {len(voice_paths)} voices[/bold green]")

        return voice_paths

    def _save_voice_metadata(
        self,
        language: str,
        voice_paths: Dict[str, Path],
        embedding_shape: tuple,
    ) -> None:
        """Save metadata about extracted voices."""
        model_dir = self.config.get_model_dir(language)
        metadata_path = model_dir / "voices" / "metadata.json"

        metadata = {
            "language": language,
            "voices": list(voice_paths.keys()),
            "embedding_dim": embedding_shape[1],
            "num_speakers": embedding_shape[0],
            "voice_files": {k: str(v.name) for k, v in voice_paths.items()},
        }

        save_json(metadata, metadata_path)

    def load_voice_embedding(self, language: str, voice: str) -> np.ndarray:
        """Load a voice embedding file."""
        voice_path = self.config.get_voice_path(language, voice)

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice embedding not found: {voice_path}")

        return load_binary(voice_path, dtype=np.float32)

    def list_voices(self, language: str) -> list:
        """List available voices for a language."""
        lang_config = self.config.get_language_config(language)
        return lang_config.voices
