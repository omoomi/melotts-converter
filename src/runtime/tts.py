"""High-level TTS interface for RKNN inference."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import Config
from runtime.bert_encoder import BERTEncoder
from runtime.decoder import RKNNDecoder
from runtime.encoder import RNNEncoder
from runtime.text_processor import TextProcessor
from utils import intersperse

console = Console()
logger = logging.getLogger(__name__)


class TTS:
    """High-level TTS interface for RKNN inference."""

    def __init__(
        self,
        language: str,
        model_dir: Optional[Path] = None,
        config: Optional[Config] = None,
        device: str = "cpu",
    ):
        """
        Initialize TTS system.

        Args:
            language: Language code (en, zh)
            model_dir: Directory containing models
            config: Configuration
            device: Device for BERT (cpu, cuda)
        """
        self.language = language.lower()
        self.config = config or Config()
        self.device = device

        lang_config = self.config.get_language_config(language)
        self.sample_rate = lang_config.sample_rate
        self.voices = lang_config.voices

        if model_dir is None:
            model_dir = self.config.get_model_dir(language)
        self.model_dir = Path(model_dir)

        self._text_processor = None
        self._bert_encoder = None
        self._encoder = None
        self._decoder = None

    def _init_text_processor(self):
        """Initialize text processor."""
        if self._text_processor is None:
            self._text_processor = TextProcessor(self.config)

    def _init_bert_encoder(self):
        """Initialize BERT encoder."""
        if self._bert_encoder is None:
            self._bert_encoder = BERTEncoder(self.language, self.config, self.device)

    def _init_encoder(self):
        """Initialize RKNN encoder."""
        if self._encoder is None:
            encoder_path = self.model_dir / "encoder.rknn"
            self._encoder = RNNEncoder(encoder_path, self.config)

    def _init_decoder(self):
        """Initialize RKNN decoder."""
        if self._decoder is None:
            decoder_path = self.model_dir / "decoder.rknn"
            self._decoder = RKNNDecoder(decoder_path, self.config)

    def _load_voice_embedding(self, voice: str) -> np.ndarray:
        """Load speaker embedding."""
        voice_path = self.model_dir / "voices" / f"{voice}.bin"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice embedding not found: {voice_path}")

        emb = np.fromfile(voice_path, dtype=np.float32)
        return emb.reshape(1, 256, 1)

    def infer(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        noise_scale: float = 0.667,
    ) -> np.ndarray:
        """
        Generate audio from text.

        Args:
            text: Input text
            voice: Voice name (default: first available voice)
            speed: Speech speed multiplier
            noise_scale: Noise scale for audio quality

        Returns:
            Audio samples as numpy array
        """
        self._init_text_processor()
        self._init_bert_encoder()
        self._init_encoder()
        self._init_decoder()

        if voice is None:
            voice = self.voices[0]

        if voice not in self.voices:
            raise ValueError(f"Unknown voice: {voice}. Available: {self.voices}")

        g = self._load_voice_embedding(voice)

        sentences = self._text_processor.split_sentences(text, self.language)

        audio_segments = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating audio...", total=len(sentences))

            for sentence in sentences:
                if not sentence.strip():
                    continue

                phones, tones, lang_ids = self._text_processor.process(
                    sentence, self.language
                )

                phones = intersperse(phones, 0)
                tones = intersperse(tones, 0)
                lang_ids = intersperse(lang_ids, 0)

                seq_len = len(phones)

                phones_arr = np.array(phones, dtype=np.int32).reshape(1, -1)
                tones_arr = np.array(tones, dtype=np.int32).reshape(1, -1)
                lang_ids_arr = np.array(lang_ids, dtype=np.int32).reshape(1, -1)

                bert = self._bert_encoder.encode_for_tts(sentence, seq_len)
                bert = bert.reshape(1, -1, seq_len)
                
                ja_bert = np.zeros((1, 768, seq_len), dtype=np.float32)

                z, y_mask, audio_len = self._encoder.infer(
                    phones=phones_arr,
                    tones=tones_arr,
                    lang_ids=lang_ids_arr,
                    bert=bert,
                    ja_bert=ja_bert,
                    g=g,
                    noise_scale=noise_scale,
                    length_scale=1.0 / speed,
                )

                audio = self._decoder.infer_long(z, g, audio_len)

                audio_segments.append(audio)

                silence = np.zeros(int(self.sample_rate * 0.05), dtype=np.float32)
                audio_segments.append(silence)

                progress.advance(task)

        audio = np.concatenate(audio_segments, axis=0)

        audio = audio / (np.abs(audio).max() + 1e-8) * 0.95

        return audio.astype(np.float32)

    def tts_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        format: Optional[str] = None,
    ) -> Path:
        """
        Generate audio from text and save to file.

        Args:
            text: Input text
            voice: Voice name
            output_path: Output file path
            speed: Speech speed multiplier
            noise_scale: Noise scale for audio quality
            format: Audio format (wav, flac, etc.)

        Returns:
            Path to output file
        """
        output_path = Path(output_path)

        audio = self.infer(
            text=text,
            voice=voice,
            speed=speed,
            noise_scale=noise_scale,
        )

        sf.write(output_path, audio, self.sample_rate, format=format)

        console.print(f"[green]✓[/green] Saved audio to {output_path}")

        return output_path

    def list_voices(self) -> List[str]:
        """List available voices."""
        return self.voices

    def release(self):
        """Release all resources."""
        if self._encoder is not None:
            self._encoder.release()
            self._encoder = None

        if self._decoder is not None:
            self._decoder.release()
            self._decoder = None

        if self._bert_encoder is not None:
            del self._bert_encoder
            self._bert_encoder = None

    def __del__(self):
        """Clean up."""
        self.release()
