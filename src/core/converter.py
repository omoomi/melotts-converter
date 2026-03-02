"""RKNN converter for ONNX models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from config import Config
from utils import ensure_dir

console = Console()
logger = logging.getLogger(__name__)


class RKNNConverter:
    """Convert ONNX models to RKNN format."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._check_rknn_installation()

    def _check_rknn_installation(self) -> None:
        """Check if RKNN-Toolkit2 is installed."""
        try:
            from rknn.api import RKNN
        except ImportError as e:
            raise ImportError(
                "RKNN-Toolkit2 is not installed. "
                "Install it with: pip install rknn-toolkit2==2.3.2 "
                "--extra-index-url https://repo.rock-chips.com/pypi/simple"
            ) from e

    def convert(
        self,
        language: str,
        target_platform: str = "rk3588",
        output_dir: Optional[Path] = None,
        quiet: bool = False,
    ) -> Dict[str, Path]:
        """
        Convert ONNX models to RKNN format.

        Args:
            language: Language code (en, zh)
            target_platform: Target platform (rk3588, rk3576)
            output_dir: Output directory (default: models/{language})
            quiet: Suppress output

        Returns:
            Dictionary with paths to converted models
        """
        if target_platform not in ["rk3588", "rk3576"]:
            raise ValueError(f"Unsupported platform: {target_platform}")

        lang_config = self.config.get_language_config(language)
        model_dir = self.config.get_model_dir(language)

        if output_dir is None:
            output_dir = model_dir

        ensure_dir(output_dir)

        if not quiet:
            console.print(
                f"[bold blue]Converting {lang_config.name} model to RKNN "
                f"for {target_platform}...[/bold blue]"
            )

        converted_paths = {}

        encoder_onnx = model_dir / "encoder.onnx"
        if encoder_onnx.exists():
            encoder_rknn = self._convert_encoder(
                encoder_onnx, output_dir, target_platform, quiet
            )
            converted_paths["encoder"] = encoder_rknn
        else:
            if not quiet:
                console.print("  [yellow]⚠[/yellow] encoder.onnx not found, skipping")

        decoder_onnx = model_dir / "decoder.onnx"
        if decoder_onnx.exists():
            decoder_rknn = self._convert_decoder(
                decoder_onnx, output_dir, target_platform, quiet
            )
            converted_paths["decoder"] = decoder_rknn
        else:
            if not quiet:
                console.print("  [yellow]⚠[/yellow] decoder.onnx not found, skipping")

        if not quiet:
            console.print("[bold green]✓ RKNN conversion complete[/bold green]")

        return converted_paths

    def _convert_encoder(
        self,
        onnx_path: Path,
        output_dir: Path,
        target_platform: str,
        quiet: bool,
    ) -> Path:
        """Convert encoder ONNX to RKNN."""
        from rknn.api import RKNN

        rknn_path = output_dir / "encoder.rknn"

        if not quiet:
            console.print(f"  Converting encoder ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)...")

        rknn = RKNN(verbose=False)

        try:
            rknn.config(
                target_platform=target_platform,
                float_dtype="float16",
                optimization_level=self.config.rknn_config.optimization_level,
                single_core_mode=False,
                dynamic_input=self._get_encoder_dynamic_inputs(),
            )

            ret = rknn.load_onnx(model=str(onnx_path))
            if ret != 0:
                raise RuntimeError(f"Failed to load ONNX model: {onnx_path} (error code: {ret})")

            ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f"Failed to build RKNN model (error code: {ret})")

            ret = rknn.export_rknn(str(rknn_path))
            if ret != 0:
                raise RuntimeError(f"Failed to export RKNN model (error code: {ret})")

            if not quiet:
                console.print(f"    [green]✓[/green] Saved to {rknn_path} ({rknn_path.stat().st_size / 1024 / 1024:.1f} MB)")

            return rknn_path

        finally:
            rknn.release()

    def _convert_decoder(
        self,
        onnx_path: Path,
        output_dir: Path,
        target_platform: str,
        quiet: bool,
    ) -> Path:
        """Convert decoder ONNX to RKNN."""
        from rknn.api import RKNN

        rknn_path = output_dir / "decoder.rknn"

        if not quiet:
            console.print(f"  Converting decoder ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)...")

        rknn = RKNN(verbose=False)

        try:
            rknn.config(
                target_platform=target_platform,
                float_dtype="float16",
                optimization_level=self.config.rknn_config.optimization_level,
                single_core_mode=False,
                dynamic_input=self._get_decoder_dynamic_inputs(),
            )

            ret = rknn.load_onnx(model=str(onnx_path))
            if ret != 0:
                raise RuntimeError(f"Failed to load ONNX model: {onnx_path} (error code: {ret})")

            ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f"Failed to build RKNN model (error code: {ret})")

            ret = rknn.export_rknn(str(rknn_path))
            if ret != 0:
                raise RuntimeError(f"Failed to export RKNN model (error code: {ret})")

            if not quiet:
                console.print(f"    [green]✓[/green] Saved to {rknn_path} ({rknn_path.stat().st_size / 1024 / 1024:.1f} MB)")

            return rknn_path

        finally:
            rknn.release()

    def _get_encoder_dynamic_inputs(self) -> List[List[List[int]]]:
        """Get dynamic input profiles for encoder."""
        profiles = []
        for profile in self.config.rknn_config.dynamic_input_profiles:
            seq_len = profile["seq_len"]

            profiles.append([
                [1, seq_len],
                [1, seq_len],
                [1, seq_len],
                [1, 1024, seq_len],
                [1, 768, seq_len],
                [1, 256, 1],
                [1],
                [1],
            ])
        return profiles

    def _get_decoder_dynamic_inputs(self) -> List[List[List[int]]]:
        """Get dynamic input profiles for decoder."""
        profiles = []
        for profile in self.config.rknn_config.dynamic_input_profiles:
            frames = profile["frames"]

            profiles.append([
                [1, 192, frames],
                [1, 256, 1],
            ])
        return profiles
