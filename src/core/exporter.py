"""ONNX exporter for MeloTTS models."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from config import Config, LanguageConfig
from utils import ensure_dir, load_json
from models import SynthesizerTrn

console = Console()
logger = logging.getLogger(__name__)


class EncoderWrapper(nn.Module):
    """
    Wrapper combining TextEncoder + DurationPredictor + Flow for ONNX export.
    This is the NPU-accelerated encoder path.
    
    Note: SDP (Stochastic Duration Predictor) is excluded for RKNN compatibility.
    Only the deterministic DurationPredictor is used.
    """

    def __init__(self, model, max_frames: int = 2048):
        super().__init__()
        self.enc_p = model.enc_p
        self.dp = model.dp
        self.flow = model.flow
        self.max_frames = max_frames

    def forward(
        self,
        phones: torch.Tensor,
        tones: torch.Tensor,
        lang_ids: torch.Tensor,
        bert: torch.Tensor,
        ja_bert: torch.Tensor,
        g: torch.Tensor,
        noise_scale: torch.Tensor,
        length_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            phones: Phone IDs [batch, seq_len]
            tones: Tone IDs [batch, seq_len]
            lang_ids: Language IDs [batch, seq_len]
            bert: BERT embeddings [batch, 1024, seq_len]
            ja_bert: Japanese BERT embeddings [batch, 768, seq_len]
            g: Speaker embedding [batch, 256, 1]
            noise_scale: Noise scale for flow
            length_scale: Length scale for duration

        Returns:
            z: Latent representation [batch, 192, max_frames] (padded)
            y_mask: Mask [batch, 1, max_frames]
            audio_len: Audio length [batch]
        """
        batch_size = phones.shape[0]
        phone_len = phones.shape[1]
        device = phones.device

        x_lengths = torch.full((batch_size,), phone_len, dtype=torch.long, device=device)

        x, m_p, logs_p, x_mask = self.enc_p(
            phones, x_lengths, tones, lang_ids, bert, ja_bert, g=g
        )

        logw = self.dp(x, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_lengths = torch.clamp_max(y_lengths, self.max_frames)
        
        y_mask = torch.unsqueeze(self._sequence_mask_fixed(y_lengths), 1).to(x_mask.dtype)

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

        attn = self._generate_path_fixed(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        audio_len = y_lengths.float()

        return z, y_mask, audio_len

    def _sequence_mask_fixed(self, length: torch.Tensor) -> torch.Tensor:
        """Sequence mask with fixed max length for RKNN compatibility."""
        max_length = self.max_frames
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)

    def _generate_path_fixed(self, duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate path with fixed max length for RKNN compatibility."""
        b, _, t_y, t_x = mask.shape
        cum_duration = torch.cumsum(duration, -1)

        cum_duration_flat = cum_duration.view(b * t_x)
        path = self._sequence_mask_fixed(cum_duration_flat).float().view(b, t_x, t_y)
        path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
        path = path.unsqueeze(1).transpose(2, 3) * mask
        return path


class DecoderWrapper(nn.Module):
    """
    Wrapper for Generator/HiFi-GAN decoder for ONNX export.
    This is the NPU-accelerated decoder path.
    """

    def __init__(self, model):
        super().__init__()
        self.dec = model.dec

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent representation [batch, 192, frames]
            g: Speaker embedding [batch, 256, 1]

        Returns:
            audio: Generated audio [batch, 1, samples]
        """
        return self.dec(z, g=g)


class ONNXExporter:
    """Export MeloTTS models to ONNX format."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def export(
        self,
        language: str,
        output_dir: Optional[Path] = None,
        opset_version: int = 14,
        quiet: bool = False,
    ) -> Dict[str, Path]:
        """
        Export model to ONNX format.

        Args:
            language: Language code (en, zh)
            output_dir: Output directory (default: models/{language})
            opset_version: ONNX opset version
            quiet: Suppress output

        Returns:
            Dictionary with paths to exported models
        """
        lang_config = self.config.get_language_config(language)
        model_dir = self.config.get_model_dir(language)

        if output_dir is None:
            output_dir = model_dir

        ensure_dir(output_dir)

        if not quiet:
            console.print(f"[bold blue]Exporting {lang_config.name} model to ONNX...[/bold blue]")

        checkpoint_path = model_dir / "checkpoint.pth"
        config_path = model_dir / "config.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = self._load_model(checkpoint_path, config_path, lang_config)

        exported_paths = {}

        encoder_path = self._export_encoder(
            model, output_dir, lang_config, opset_version, quiet
        )
        exported_paths["encoder"] = encoder_path

        decoder_path = self._export_decoder(
            model, output_dir, lang_config, opset_version, quiet
        )
        exported_paths["decoder"] = decoder_path

        if not quiet:
            console.print("[bold green]✓ ONNX export complete[/bold green]")

        return exported_paths

    def _load_model(self, checkpoint_path: Path, config_path: Path, lang_config: LanguageConfig):
        """Load MeloTTS model from checkpoint."""
        hps = load_json(config_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model"]

        n_vocab = model_state["enc_p.emb.weight"].shape[0]
        num_tones = model_state["enc_p.tone_emb.weight"].shape[0]
        num_languages = model_state["enc_p.language_emb.weight"].shape[0]

        model_config = hps.get("model", {}).copy()
        if not model_config.get("use_spk_conditioned_encoder", True):
            model_config["gin_channels"] = 0

        model = SynthesizerTrn(
            n_vocab,
            hps.get("data", {}).get("filter_length", 2048) // 2 + 1,
            hps.get("train", {}).get("segment_size", 8192) // hps.get("data", {}).get("hop_length", 512),
            n_speakers=hps.get("data", {}).get("n_speakers", 256),
            num_tones=num_tones,
            num_languages=num_languages,
            **model_config,
        )

        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        return model

    def _export_encoder(
        self,
        model,
        output_dir: Path,
        lang_config: LanguageConfig,
        opset_version: int,
        quiet: bool,
    ) -> Path:
        """Export encoder to ONNX."""
        encoder_path = output_dir / "encoder.onnx"

        if not quiet:
            console.print("  Exporting encoder...")

        max_frames = self.config.rknn_config.encoder_max_frames
        wrapper = EncoderWrapper(model, max_frames=max_frames)

        seq_len = 64
        batch_size = 1

        phones = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
        tones = torch.zeros((batch_size, seq_len), dtype=torch.long)
        lang_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        bert = torch.randn((batch_size, lang_config.bert_dim, seq_len), dtype=torch.float32)
        ja_bert = torch.randn((batch_size, 768, seq_len), dtype=torch.float32)
        g = torch.randn((batch_size, 256, 1), dtype=torch.float32)
        noise_scale = torch.tensor([0.667], dtype=torch.float32)
        length_scale = torch.tensor([1.0], dtype=torch.float32)

        dynamic_axes = {
            "phones": {0: "batch_size", 1: "seq_len"},
            "tones": {0: "batch_size", 1: "seq_len"},
            "lang_ids": {0: "batch_size", 1: "seq_len"},
            "bert": {0: "batch_size", 2: "seq_len"},
            "ja_bert": {0: "batch_size", 2: "seq_len"},
            "z": {0: "batch_size", 2: "audio_frames"},
            "y_mask": {0: "batch_size", 2: "audio_frames"},
        }

        torch.onnx.export(
            wrapper,
            (phones, tones, lang_ids, bert, ja_bert, g, noise_scale, length_scale),
            encoder_path,
            input_names=[
                "phones", "tones", "lang_ids", "bert", "ja_bert", "g",
                "noise_scale", "length_scale"
            ],
            output_names=["z", "y_mask", "audio_len"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

        if not quiet:
            console.print(f"    [green]✓[/green] Saved to {encoder_path}")

        return encoder_path

    def _export_decoder(
        self,
        model,
        output_dir: Path,
        lang_config: LanguageConfig,
        opset_version: int,
        quiet: bool,
    ) -> Path:
        """Export decoder to ONNX."""
        decoder_path = output_dir / "decoder.onnx"

        if not quiet:
            console.print("  Exporting decoder...")

        wrapper = DecoderWrapper(model)

        batch_size = 1
        frames = 4096

        z = torch.randn((batch_size, 192, frames), dtype=torch.float32)
        g = torch.randn((batch_size, 256, 1), dtype=torch.float32)

        dynamic_axes = {
            "z": {0: "batch_size", 2: "frames"},
            "audio": {0: "batch_size", 2: "samples"},
        }

        torch.onnx.export(
            wrapper,
            (z, g),
            decoder_path,
            input_names=["z", "g"],
            output_names=["audio"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

        if not quiet:
            console.print(f"    [green]✓[/green] Saved to {decoder_path}")

        return decoder_path
