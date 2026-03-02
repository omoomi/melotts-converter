"""CLI main entry point for melotts-converter.

This module provides the command-line interface for downloading, exporting,
converting, and using MeloTTS models in RKNN format for Rockchip NPU hardware.

Example:
    $ melotts-converter list
    $ melotts-converter build --language en --target rk3588
    $ melotts-converter inference --language en --voice en-us --text "Hello" --output test.wav
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="melotts-converter")
def cli():
    """MeloTTS Converter - Download and convert HuggingFace models to RKNN format.

    Convert MeloTTS text-to-speech models for deployment on Rockchip RK3588/RK3576.
    """
    pass


@cli.command()
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "zh", "jp"]),
    required=True,
    help="Language to download (en, zh, jp)",
)
@click.option("--force", "-f", is_flag=True, help="Force re-download")
def download(language: str, force: bool):
    """Download model from HuggingFace.

    Downloads the MeloTTS model checkpoint and config for the specified language
    from HuggingFace and saves them to the models directory.

    Args:
        language: Language code ('en', 'zh', 'jp').
        force: Force re-download even if files exist.

    Example:
        $ melotts-converter download --language en
        $ melotts-converter download -l zh --force
    """
    from config import Config
    from core.downloader import ModelDownloader

    try:
        config = Config()
        downloader = ModelDownloader(config)

        downloader.download_model(language, force=force)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "zh"]),
    required=True,
    help="Language to export",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory",
)
def export(language: str, output_dir: Optional[str]):
    """Export PyTorch model to ONNX format.

    Exports the downloaded MeloTTS model to ONNX format, which is an intermediate
    format before conversion to RKNN.

    Args:
        language: Language code ('en', 'zh').
        output_dir: Custom output directory. If None, uses models/{language}.

    Example:
        $ melotts-converter export --language en
        $ melotts-converter export -l zh -o /path/to/output
    """
    from config import Config
    from core.exporter import ONNXExporter

    try:
        config = Config()
        exporter = ONNXExporter(config)

        output_path = Path(output_dir) if output_dir else None

        exporter.export(language, output_dir=output_path)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "zh"]),
    required=True,
    help="Language to convert",
)
@click.option(
    "--target",
    "-t",
    type=click.Choice(["rk3588", "rk3576"]),
    default="rk3588",
    help="Target platform",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory",
)
def convert(language: str, target: str, output_dir: Optional[str]):
    """Convert ONNX model to RKNN format.

    Converts the ONNX model to RKNN format for NPU acceleration on Rockchip
    hardware. Requires ONNX models to be exported first.

    Args:
        language: Language code ('en', 'zh').
        target: Target platform ('rk3588' or 'rk3576').
        output_dir: Custom output directory. If None, uses models/{language}.

    Example:
        $ melotts-converter convert --language en --target rk3588
        $ melotts-converter convert -l zh -t rk3576 -o /path/to/output
    """
    from config import Config
    from core.converter import RKNNConverter

    try:
        config = Config()
        converter = RKNNConverter(config)

        output_path = Path(output_dir) if output_dir else None

        converter.convert(language, target_platform=target, output_dir=output_path)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "zh"]),
    required=True,
    help="Language to build",
)
@click.option(
    "--target",
    "-t",
    type=click.Choice(["rk3588", "rk3576"]),
    default="rk3588",
    help="Target platform",
)
@click.option("--force", "-f", is_flag=True, help="Force re-download")
def build(language: str, target: str, force: bool):
    """Full pipeline: download, export, and convert.

    Runs the complete pipeline to prepare models for inference:
    1. Download model from HuggingFace
    2. Export to ONNX format
    3. Extract voice embeddings
    4. Convert to RKNN format

    Args:
        language: Language code ('en', 'zh').
        target: Target platform ('rk3588' or 'rk3576').
        force: Force re-download even if files exist.

    Example:
        $ melotts-converter build --language en --target rk3588
        $ melotts-converter build -l zh -t rk3576 --force
    """
    from config import Config
    from core.converter import RKNNConverter
    from core.downloader import ModelDownloader
    from core.exporter import ONNXExporter
    from core.voice_extractor import VoiceExtractor

    try:
        config = Config()

        console.print(f"[bold blue]Building {language.upper()} model...[/bold blue]\n")

        downloader = ModelDownloader(config)
        downloader.download_model(language, force=force)

        exporter = ONNXExporter(config)
        exporter.export(language)

        voice_extractor = VoiceExtractor(config)
        voice_extractor.extract_voices(language)

        converter = RKNNConverter(config)
        converter.convert(language, target_platform=target)

        console.print("\n[bold green]✓ Build complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def list_models():
    """List available and downloaded models.

    Shows all supported languages and their download status.

    Example:
        $ melotts-converter list
    """
    from config import Config
    from core.downloader import ModelDownloader

    try:
        config = Config()
        downloader = ModelDownloader(config)

        table = Table(title="Available Models")
        table.add_column("Language", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Voices", style="yellow")

        for lang, lang_config in config.SUPPORTED_LANGUAGES.items():
            downloaded = downloader.is_model_downloaded(lang)
            status = "[green]Downloaded[/green]" if downloaded else "[dim]Not downloaded[/dim]"
            voices = ", ".join(lang_config.voices)

            table.add_row(lang_config.name, status, voices)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "zh"]),
    required=True,
    help="Language for inference",
)
@click.option(
    "--voice",
    "-v",
    type=str,
    default=None,
    help="Voice to use (default: first available)",
)
@click.option(
    "--text",
    "-t",
    type=str,
    required=True,
    help="Text to synthesize",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="output.wav",
    help="Output audio file",
)
@click.option(
    "--speed",
    "-s",
    type=float,
    default=1.0,
    help="Speech speed multiplier",
)
def inference(language: str, voice: Optional[str], text: str, output: str, speed: float):
    """Run TTS inference on RKNN models.

    Synthesizes speech from text using the converted RKNN models.
    Requires the build pipeline to be completed first.

    Args:
        language: Language code ('en', 'zh').
        voice: Voice name to use. If None, uses the first available voice.
        text: Text to synthesize (required).
        output: Output audio file path (default: 'output.wav').
        speed: Speech speed multiplier (default: 1.0).

    Example:
        $ melotts-converter inference --language en --voice en-us --text "Hello world!" --output test.wav
        $ melotts-converter inference -l zh -t "你好世界" -o test_zh.wav --speed 1.2
    """
    from config import Config
    from runtime.tts import TTS

    try:
        config = Config()

        tts = TTS(language, config=config)

        if voice is None:
            voice = tts.list_voices()[0]

        console.print(f"[bold blue]Synthesizing with voice: {voice}[/bold blue]")

        tts.tts_to_file(
            text=text,
            voice=voice,
            output_path=output,
            speed=speed,
        )

        tts.release()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "zh"]),
    required=True,
    help="Language info to show",
)
def info(language: str):
    """Show model information.

    Displays detailed information about a downloaded model including
    configuration parameters and file sizes.

    Args:
        language: Language code ('en', 'zh').

    Example:
        $ melotts-converter info --language en
        $ melotts-converter info -l zh
    """
    from config import Config
    from core.downloader import ModelDownloader

    try:
        config = Config()
        downloader = ModelDownloader(config)

        if not downloader.is_model_downloaded(language):
            console.print(f"[yellow]Model for {language} is not downloaded[/yellow]")
            return

        model_info = downloader.get_model_info(language)
        lang_config = config.get_language_config(language)

        table = Table(title=f"Model Info: {lang_config.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Language", lang_config.name)
        table.add_row("HuggingFace Repo", lang_config.huggingface_repo)
        table.add_row("Sample Rate", str(lang_config.sample_rate))
        table.add_row("Voices", ", ".join(lang_config.voices))
        table.add_row("BERT Dimension", str(lang_config.bert_dim))

        if "_melotts_converter" in model_info:
            table.add_row("Downloaded", "Yes")

        console.print(table)

        model_dir = config.get_model_dir(language)

        console.print("\n[bold]Files:[/bold]")
        for file in model_dir.glob("*"):
            if file.is_file():
                size = file.stat().st_size
                size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
                console.print(f"  {file.name}: {size_str}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
