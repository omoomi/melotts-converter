# MeloTTS Converter - Documentation

This document provides comprehensive documentation for the MeloTTS Converter project.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Development](#development)
8. [Troubleshooting](#troubleshooting)

---

## Overview

MeloTTS Converter downloads and converts HuggingFace MeloTTS text-to-speech models to RKNN format for deployment on Rockchip RK3588/RK3576 chipsets with NPU acceleration.

### Features

- **Model Download**: Download MeloTTS models from HuggingFace
- **ONNX Export**: Convert PyTorch models to ONNX format
- **RKNN Conversion**: Optimize for Rockchip NPU hardware
- **Voice Extraction**: Extract speaker embeddings for multiple voices
- **High-level API**: Python API for easy integration
- **CLI Tools**: Command-line interface for all operations

### Supported Languages

| Language | Model | Voices | Status |
|----------|--------|---------|---------|
| English | `myshell-ai/MeloTTS-English` | 5 (EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default) | ✅ Working |
| Chinese | `myshell-ai/MeloTTS-Chinese` | 1 (ZH with mixed EN support) | ✅ Working |
| Japanese | `myshell-ai/MeloTTS-Japanese` | 1 (JP) | ✅ Working |

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Text Input                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Text Processor (CPU)                              │
│  - Sentence segmentation                                  │
│  - Phoneme/tonalization                                 │
│  - Mixed language detection                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         BERT Encoder (CPU)                               │
│  - Text embedding extraction                             │
│  - Language-specific BERT models                         │
│  - Fast CPU inference                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         RKNN Encoder (NPU)                               │
│  - TextEncoder: Phone/tone/language embedding            │
│  - DurationPredictor: Predict phoneme durations          │
│  - Flow: Latent transformation                          │
│  - Output: Latent representation z                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         RKNN Decoder (NPU)                               │
│  - Generator/HiFi-GAN: Audio generation                │
│  - Upsampling layers                                   │
│  - Output: Audio waveform                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
              Audio Output (WAV)
```

### Component Responsibilities

| Component | Device | Purpose | Notes |
|-----------|--------|---------|--------|
| TextProcessor | CPU | Text normalization, phonemization | Language-specific |
| BERTEncoder | CPU | BERT embeddings for text | Fast, no NPU needed |
| Encoder | NPU | Phone → Latent transformation | RKNN optimized |
| Decoder | NPU | Latent → Audio generation | RKNN optimized |

---

## Installation

### Requirements

- Python 3.9-3.12
- RKNN-Toolkit2 2.3.2
- ONNX 1.18.0
- PyTorch 2.0+
- Rockchip RK3588/RK3576 hardware (for inference)

### Installation Steps

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd melotts-converter
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -e .
   ```

4. **Install RKNN-Toolkit2** (from Rockchip repository)
   ```bash
   pip install rknn-toolkit2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple
   ```

---

## Usage

### Command-Line Interface

#### List Available Models
```bash
melotts-converter list
```

#### Download Model
```bash
# English model
melotts-converter download --language en

# Chinese model with force re-download
melotts-converter download --language zh --force
```

#### Export to ONNX
```bash
melotts-converter export --language en
```

#### Convert to RKNN
```bash
# For RK3588
melotts-converter convert --language en --target rk3588

# For RK3576
melotts-converter convert --language en --target rk3576
```

#### Full Pipeline (Recommended)
```bash
melotts-converter build --language en --target rk3588
```

This runs:
1. Download model from HuggingFace
2. Export to ONNX format
3. Extract voice embeddings
4. Convert to RKNN

#### Run Inference
```bash
# English
melotts-converter inference \
    --language en \
    --voice en-us \
    --text "Hello, this is a test." \
    --output test.wav \
    --speed 1.0

# Chinese with mixed English
melotts-converter inference \
    --language zh \
    --voice zh \
    --text "我最近在学习machine learning。" \
    --output test_zh.wav
```

#### Show Model Info
```bash
melotts-converter info --language en
```

### Python API

#### Basic Usage
```python
from runtime.tts import TTS

# Initialize TTS
tts = TTS(language='en')

# List available voices
print(tts.list_voices())
# ['en-us', 'en-br', 'en-india', 'en-au', 'en-default']

# Generate audio
tts.tts_to_file(
    text="Hello, this is a test of the text to speech system.",
    voice='en-us',
    output_path='output.wav',
    speed=1.0
)

# Clean up
tts.release()
```

#### Advanced Usage
```python
from config import Config
from runtime.tts import TTS

# Custom configuration
config = Config(
    models_dir=Path('/custom/models'),
    cache_dir=Path('/custom/cache')
)

tts = TTS(language='en', config=config, device='cpu')

# Generate audio with custom parameters
audio = tts.infer(
    text="Advanced usage example.",
    voice='en-us',
    speed=1.2,
    noise_scale=0.667
)

# Save manually
import soundfile as sf
sf.write('output.wav', audio, tts.sample_rate)

tts.release()
```

#### Chinese with Mixed English
```python
tts_zh = TTS(language='zh')
tts_zh.tts_to_file(
    text="我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。",
    voice='zh',
    output_path='output_zh.wav'
)
tts_zh.release()
```

---

## API Reference

### Configuration Classes

#### `Config`
Main configuration class for managing all settings.

```python
from config import Config

config = Config(
    models_dir=Path('models'),      # Model storage directory
    cache_dir=Path('.cache')         # Download cache directory
)

# Get language configuration
lang_config = config.get_language_config('en')
print(lang_config.name)  # "English"

# Get model directory
model_dir = config.get_model_dir('en')

# Get voice path
voice_path = config.get_voice_path('en', 'en-us')
```

#### `LanguageConfig`
Configuration for a specific language.

```python
@dataclass
class LanguageConfig:
    name: str                    # Human-readable name
    huggingface_repo: str        # HF repository ID
    voices: List[str]            # Available voices
    sample_rate: int = 44100     # Audio sample rate
    bert_dim: int = 1024         # BERT dimension
    num_languages: int = 1       # Number of languages
    num_tones: int = 1           # Number of tones
```

#### `ModelConfig`
Configuration for model architecture.

```python
@dataclass
class ModelConfig:
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    # ... and more architecture parameters
```

#### `RKNNConfig`
Configuration for RKNN conversion.

```python
@dataclass
class RKNNConfig:
    target_platform: str = "rk3588"
    float_dtype: str = "float16"
    optimization_level: int = 3
    do_quantization: bool = False
    encoder_max_frames: int = 2048
    dynamic_input_profiles: List[Dict] = [...]  # Dynamic input shapes
```

### TTS Class

#### `__init__(language, model_dir=None, config=None, device='cpu')`
Initialize TTS system.

**Parameters:**
- `language` (str): Language code ('en', 'zh', 'jp')
- `model_dir` (Path, optional): Directory containing models
- `config` (Config, optional): Configuration object
- `device` (str): Device for BERT encoding ('cpu', 'cuda')

**Raises:**
- `ValueError`: If language is not supported

**Example:**
```python
tts = TTS(language='en', model_dir=Path('models/en'))
```

#### `infer(text, voice=None, speed=1.0, noise_scale=0.667)`
Generate audio from text.

**Parameters:**
- `text` (str): Input text to synthesize
- `voice` (str, optional): Voice name (default: first available)
- `speed` (float): Speech speed multiplier (default: 1.0)
- `noise_scale` (float): Noise scale for flow model (default: 0.667)

**Returns:**
- `np.ndarray`: Audio samples (float32, normalized to [-1, 1])

**Raises:**
- `ValueError`: If voice name is not recognized
- `FileNotFoundError`: If model files are missing

**Example:**
```python
audio = tts.infer("Hello world!", voice='en-us', speed=1.2)
```

#### `tts_to_file(text, output_path, voice=None, speed=1.0, noise_scale=0.667, format=None)`
Generate audio from text and save to file.

**Parameters:**
- `text` (str): Input text to synthesize
- `output_path` (str/Path): Output file path
- `voice` (str, optional): Voice name
- `speed` (float): Speech speed multiplier (default: 1.0)
- `noise_scale` (float): Noise scale for flow model (default: 0.667)
- `format` (str, optional): Audio format ('wav', 'flac', etc.)

**Returns:**
- `Path`: Path to saved audio file

**Example:**
```python
tts.tts_to_file("Hello world!", "output.wav", voice='en-us')
```

#### `list_voices()`
List available voices for current language.

**Returns:**
- `List[str]`: List of voice names

**Example:**
```python
voices = tts.list_voices()
print(voices)  # ['en-us', 'en-br', ...]
```

#### `release()`
Release all resources and cleanup.

**Example:**
```python
tts.release()
```

### Utility Functions

#### `ensure_dir(path)`
Ensure a directory exists, creating it if necessary.

```python
from utils import ensure_dir
dir_path = ensure_dir('models/en/voices')
```

#### `load_json(path)`, `save_json(data, path)`
Load/save JSON data.

```python
from utils import load_json, save_json

data = load_json('config.json')
save_json(data, 'output.json')
```

#### `load_binary(path, dtype=np.float32)`, `save_binary(data, path)`
Load/save binary data as numpy arrays.

```python
from utils import load_binary, save_binary
import numpy as np

data = load_binary('voice.bin', dtype=np.float32)
save_binary(data, 'voice_copy.bin')
```

#### `intersperse(lst, item)`
Intersperse an item between list elements.

```python
from utils import intersperse
result = intersperse([1, 2, 3], 0)
# [0, 1, 0, 2, 0, 3, 0]
```

#### `split_sentences(text, language)`
Split text into sentences based on language.

```python
from utils import split_sentences

sentences = split_sentences("Hello. World!", "en")
# ['Hello.', 'World!']

sentences = split_sentences("你好。你好吗？", "zh")
# ['你好。', '你好吗？']
```

#### `clean_text(text)`
Clean and normalize text.

```python
from utils import clean_text
clean = clean_text("  Hello    world  ")
# 'Hello world'
```

#### `format_time(seconds)`, `format_size(bytes_size)`
Format values to human-readable strings.

```python
from utils import format_time, format_size

time_str = format_time(90.5)  # '1m 30.5s'
size_str = format_size(2500000000)  # '2.3GB'
```

#### `validate_rknn_installation()`, `validate_onnx_version()`
Validate installation of dependencies.

```python
from utils import validate_rknn_installation, validate_onnx_version

has_rknn = validate_rknn_installation()
onnx_ok = validate_onnx_version()
```

#### `get_device_info()`
Get information about available compute devices.

```python
from utils import get_device_info

info = get_device_info()
# {'cpu': True, 'cuda': False, 'mps': False, 'rknn': True}
```

---

## Configuration

### Model Configuration

Model architecture parameters are defined in `ModelConfig`:

```python
@dataclass
class ModelConfig:
    inter_channels: int = 192          # Internal channels
    hidden_channels: int = 192          # Hidden layer dimensions
    filter_channels: int = 768          # Filter dimensions
    n_heads: int = 2                   # Attention heads
    n_layers: int = 6                   # Encoder layers
    kernel_size: int = 3                 # Convolution kernel size
    p_dropout: float = 0.1              # Dropout probability
    resblock: str = "1"                 # Residual block type
    resblock_kernel_sizes: List[int]      # Kernel sizes
    resblock_dilation_sizes: List[List[int]]  # Dilation rates
    upsample_rates: List[int]            # Upsampling factors
    upsample_initial_channel: int = 512  # Initial channel count
    upsample_kernel_sizes: List[int]     # Upsampling kernel sizes
    n_layers_trans_flow: int = 6        # Transformer flow layers
    gin_channels: int = 256              # Speaker conditioning channels
    n_speakers: int = 256              # Number of speakers
```

### RKNN Configuration

RKNN conversion parameters are defined in `RKNNConfig`:

```python
@dataclass
class RKNNConfig:
    target_platform: str = "rk3588"      # Target NPU platform
    float_dtype: str = "float16"           # Floating-point data type
    optimization_level: int = 3             # Optimization level (0-3)
    do_quantization: bool = False           # Apply quantization
    quantized_dtype: str = "w8a8"        # Quantization data type
    encoder_max_frames: int = 2048         # Max frames for encoder
    dynamic_input_profiles: List[Dict] = [  # Dynamic input shapes
        {"seq_len": 64, "frames": 1024},
        {"seq_len": 128, "frames": 2048},
    ]
```

### Language Configuration

Language-specific settings are defined in `Config.SUPPORTED_LANGUAGES`:

```python
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
    # ... and more languages
}
```

---

## Development

### Project Structure

```
melotts-converter/
├── src/
│   ├── cli/                    # Command-line interface
│   │   └── main.py           # CLI entry point
│   ├── config.py              # Configuration classes
│   ├── utils.py              # Utility functions
│   ├── models.py             # Model architecture
│   ├── modules.py            # Neural network modules
│   ├── attentions.py         # Attention mechanisms
│   ├── transforms.py         # Flow transformations
│   ├── commons.py           # Common functions
│   ├── text/                # Text processing
│   │   ├── english.py       # English text processor
│   │   ├── chinese.py       # Chinese text processor
│   │   ├── symbols.py       # Phone symbols
│   │   └── __init__.py
│   ├── core/                # Core operations
│   │   ├── downloader.py    # HuggingFace downloader
│   │   ├── exporter.py      # ONNX exporter
│   │   ├── converter.py     # RKNN converter
│   │   └── voice_extractor.py  # Voice embedding extractor
│   └── runtime/             # Runtime inference
│       ├── tts.py            # High-level TTS interface
│       ├── text_processor.py  # Text processor wrapper
│       ├── bert_encoder.py   # BERT encoder
│       ├── encoder.py        # RKNN encoder
│       └── decoder.py        # RKNN decoder
├── tests/                   # Test suite
├── models/                  # Model storage (gitignored)
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── README.md               # Project overview
```

### Adding a New Language

1. **Add language configuration** in `src/config.py`:
   ```python
   "fr": LanguageConfig(
       name="French",
       huggingface_repo="myshell-ai/MeloTTS-French",
       voices=["fr"],
       sample_rate=44100,
       bert_dim=1024,
       num_languages=1,
       num_tones=1,
   )
   ```

2. **Add BERT model** in `src/runtime/bert_encoder.py`:
   ```python
   BERT_MODELS = {
       "en": "bert-base-uncased",
       "zh": "bert-base-chinese",
       "fr": "camembert-base",  # Add this
       # ...
   }
   ```

3. **Add text processor** if needed in `src/text/`

### Testing

Run tests with pytest:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Code Style

Follow PEP 8 style guide. Use the provided tools:

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type check
mypy src/
```

---

## Troubleshooting

### Common Issues

#### RKNN Installation Fails
```bash
# Try with specific Python version
python3.10 -m venv venv
source venv/bin/activate
pip install rknn-toolkit2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple
```

#### ONNX Version Conflict
Ensure you have the correct ONNX version:
```bash
pip install onnx==1.18.0
```

#### Model Download Fails
Check internet connection and HuggingFace access:
```bash
huggingface-cli login
```

#### Inference Fails on Device
1. Verify RKNN models are built correctly
2. Check model files exist in correct directory
3. Verify voice embedding files are present
4. Check NPU is available on hardware

### Error Messages

| Error | Cause | Solution |
|--------|--------|-----------|
| `ImportError: RKNN-Toolkit2 is not installed` | RKNN not installed | Install RKNN-Toolkit2 |
| `FileNotFoundError: Model not found` | Model files missing | Run `melotts-converter build` |
| `ValueError: Unsupported language` | Language code invalid | Use supported language code |
| `RuntimeError: Failed to load RKNN model` | RKNN model corrupted | Rebuild with `--force` |
| `ValueError: Unknown voice` | Voice name invalid | Use `list_voices()` to see available voices |

### Performance Optimization

To reduce RKNN model size:
1. **Use fewer dynamic profiles**:
   ```python
   rknn_config.dynamic_input_profiles = [
       {"seq_len": 64, "frames": 1024},
   ]
   ```

2. **Enable quantization**:
   ```python
   rknn_config.do_quantization = True
   rknn_config.quantized_dtype = "w8a8"
   ```

3. **Use static profile**:
   ```python
   # Edit dynamic profiles to use single size
   ```

### Getting Help

- Check the [README.md](../README.md) for quick start
- See [docs/](./) directory for detailed documentation
- Review [docs/CRITICAL_FIXES_APPLIED.md](./CRITICAL_FIXES_APPLIED.md) for recent fixes
- Open an issue on GitHub with detailed error information

---

## License

MIT License

## Acknowledgments

- [MeloTTS](https://github.com/myshell-ai/MeloTTS) - Original TTS model
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) - Rockchip NPU SDK
- [happyme531/MeloTTS-RKNN2](https://huggingface.co/happyme531/MeloTTS-RKNN2) - Reference implementation
