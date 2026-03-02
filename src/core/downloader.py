"""Model downloader for HuggingFace models."""

import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import Config, LanguageConfig
from utils import ensure_dir, load_json, save_json

console = Console()
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download models from HuggingFace."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def download_model(
        self,
        language: str,
        force: bool = False,
        quiet: bool = False,
    ) -> Path:
        """
        Download model files for a language.

        Args:
            language: Language code (en, zh, jp)
            force: Force re-download even if files exist
            quiet: Suppress progress output

        Returns:
            Path to the model directory
        """
        lang_config = self.config.get_language_config(language)
        model_dir = self.config.get_model_dir(language)

        if not quiet:
            console.print(
                f"[bold blue]Downloading {lang_config.name} model...[/bold blue]"
            )

        files_to_download = ["checkpoint.pth", "config.json"]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet,
        ) as progress:
            for filename in files_to_download:
                task = progress.add_task(f"Downloading {filename}...", total=None)

                local_path = model_dir / filename

                if local_path.exists() and not force:
                    if not quiet:
                        progress.update(task, description=f"[green]✓[/green] {filename} already exists")
                    continue

                try:
                    downloaded_path = hf_hub_download(
                        repo_id=lang_config.huggingface_repo,
                        filename=filename,
                        cache_dir=self.config.cache_dir,
                        force_download=force,
                    )

                    ensure_dir(model_dir)
                    local_path.write_bytes(Path(downloaded_path).read_bytes())

                    if not quiet:
                        progress.update(task, description=f"[green]✓[/green] Downloaded {filename}")

                except Exception as e:
                    progress.update(task, description=f"[red]✗[/red] Failed to download {filename}")
                    raise RuntimeError(f"Failed to download {filename}: {e}") from e

        self._save_model_metadata(language, lang_config)

        if not quiet:
            console.print(f"[bold green]✓ Model downloaded to {model_dir}[/bold green]")

        return model_dir

    def _save_model_metadata(self, language: str, lang_config: LanguageConfig) -> None:
        """Save metadata about the downloaded model."""
        model_dir = self.config.get_model_dir(language)
        config_path = model_dir / "config.json"

        if config_path.exists():
            config_data = load_json(config_path)
        else:
            config_data = {}

        config_data["_melotts_converter"] = {
            "language": language,
            "huggingface_repo": lang_config.huggingface_repo,
            "voices": lang_config.voices,
            "sample_rate": lang_config.sample_rate,
        }

        save_json(config_data, config_path)

    def is_model_downloaded(self, language: str) -> bool:
        """Check if model files exist for a language."""
        model_dir = self.config.get_model_dir(language)
        required_files = ["checkpoint.pth", "config.json"]

        return all((model_dir / f).exists() for f in required_files)

    def get_model_info(self, language: str) -> dict:
        """Get information about a downloaded model."""
        if not self.is_model_downloaded(language):
            raise ValueError(f"Model for {language} is not downloaded")

        model_dir = self.config.get_model_dir(language)
        config_path = model_dir / "config.json"

        return load_json(config_path)

    def list_downloaded_models(self) -> list:
        """List all downloaded models."""
        downloaded = []

        for lang in self.config.SUPPORTED_LANGUAGES:
            if self.is_model_downloaded(lang):
                downloaded.append(lang)

        return downloaded
