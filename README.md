# RKNN Converter

Download and convert MeloTTS text-to-speech models from HuggingFace to RKNN format for deployment on Rockchip RK3588/RK3576 chipsets.

## Status

| Feature | Status |
|---------|--------|
| Model download | ✅ Working |
| ONNX export | ✅ Working |
| RKNN conversion | ✅ Working |
| CLI commands | ✅ Working |
| Python runtime | 🔄 In progress |
| Inference testing | ⬜ Not tested on device |

## Features

- Download MeloTTS models from HuggingFace
- Export PyTorch models to ONNX format
- Convert ONNX to RKNN with NPU optimization
- Extract speaker embeddings for multiple voices
- High-level TTS inference API
- Full pipeline automation

## Supported Languages

| Language | Model | Voices |
|----------|-------|--------|
| English | `myshell-ai/MeloTTS-English` | 5 (EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default) |
| Chinese | `myshell-ai/MeloTTS-Chinese` | 1 (ZH with mixed EN support) |

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -e .
```

**Note:** `setuptools<71` is required for `pkg_resources` used by RKNN-Toolkit2.

### 3. Install RKNN-Toolkit2

RKNN-Toolkit2 is not available on PyPI. Install it separately:

```bash
pip install rknn-toolkit2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple
```

**Note:** RKNN-Toolkit2 v2.3.2 requires `onnx==1.18.0` (already included in dependencies).

## Usage

### CLI Commands

#### List Available Models

```bash
rknn-converter list
```

#### Download Model

```bash
# Download English model
rknn-converter download --language en

# Download Chinese model
rknn-converter download --language zh
```

#### Export to ONNX

```bash
rknn-converter export --language en
```

#### Convert to RKNN

```bash
# For RK3588
rknn-converter convert --language en --target rk3588

# For RK3576
rknn-converter convert --language en --target rk3576
```

#### Full Pipeline (Recommended)

```bash
# Build complete model pipeline
rknn-converter build --language en --target rk3588
```

This will:
1. Download model from HuggingFace
2. Export to ONNX
3. Extract voice embeddings
4. Convert to RKNN

#### Run Inference

```bash
# English
rknn-converter inference \
    --language en \
    --voice en-us \
    --text "Hello, this is a test." \
    --output test.wav

# Chinese with mixed English
rknn-converter inference \
    --language zh \
    --voice zh \
    --text "我最近在学习machine learning。" \
    --output test_zh.wav
```

#### Show Model Info

```bash
rknn-converter info --language en
```

### Python API

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

# Chinese with mixed English
tts_zh = TTS(language='zh')
tts_zh.tts_to_file(
    text="我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。",
    voice='zh',
    output_path='output_zh.wav'
)

# Clean up
tts.release()
```

## Architecture

### Inference Pipeline

```
Text Input
    │
    ▼
[CPU] Text Processor ──► phones, tones, lang_ids
    │
    ▼
[CPU] BERT Encoder ──► bert_embeddings
    │
    ▼
[NPU] encoder.rknn ──► z (latent)
    │
    ▼
[NPU] decoder.rknn ──► audio
    │
    ▼
Audio Output
```

### NPU Utilization

| Component | Device | Purpose |
|-----------|--------|---------|
| TextEncoder | NPU | Process text to latent |
| DurationPredictor | NPU | Predict phoneme durations (deterministic) |
| Flow | NPU | Latent transformation |
| Generator | NPU | Audio generation (HiFi-GAN) |
| BERT | CPU | Text embeddings (fast) |

**Note:** Stochastic Duration Predictor (SDP) is excluded for RKNN compatibility. Only the deterministic DurationPredictor is used.

## Project Structure

```
rknn-converter/
├── src/
│   ├── cli/           # CLI commands
│   ├── core/          # Core conversion modules
│   ├── runtime/       # Inference runtime
│   ├── text/          # Text processing
│   ├── config.py      # Configuration
│   └── utils.py       # Utilities
├── models/            # Downloaded/converted models
│   ├── en/
│   ├── zh/
│   └── jp/
├── tests/             # Unit tests
├── pyproject.toml     # Package configuration
└── README.md
```

## Model Files

After building, each language model directory contains:

```
models/en/
├── config.json        # Model configuration
├── checkpoint.pth     # PyTorch checkpoint (~200MB)
├── encoder.onnx       # ONNX encoder (~110MB)
├── decoder.onnx       # ONNX decoder (~58MB)
├── encoder.rknn       # RKNN encoder (~1.3GB, NPU)
├── decoder.rknn       # RKNN decoder (~180MB, NPU)
└── voices/
    ├── en-us.bin      # Speaker embeddings (256-dim)
    ├── en-br.bin
    ├── en-india.bin
    ├── en-au.bin
    └── en-default.bin
```

### Model Sizes (English)

| File | Size | Description |
|------|------|-------------|
| checkpoint.pth | ~200 MB | PyTorch checkpoint |
| encoder.onnx | ~110 MB | ONNX encoder |
| decoder.onnx | ~58 MB | ONNX decoder |
| encoder.rknn | ~1.3 GB | RKNN encoder (NPU) |
| decoder.rknn | ~180 MB | RKNN decoder (NPU) |

## Requirements

- Python 3.9-3.12
- RKNN-Toolkit2 2.3.2
- ONNX 1.18.0
- PyTorch 2.0+
- Transformers 4.27+

## Target Devices

- RK3588 (6 TOPS NPU)
- RK3576 (6 TOPS NPU)

## Performance

Expected performance on RK3588:
- Inference speed: ~3-5x real-time
- Memory usage: ~200MB
- Latency: ~100-200ms per sentence

## Limitations

1. **No quantization**: Models use float16 for full quality
2. **Fixed output length**: Encoder uses max_frames=2048 for RKNN compatibility
3. **No SDP**: Stochastic Duration Predictor excluded (deterministic only)
4. **BERT on CPU**: BERT inference runs on CPU (fast enough)
5. **Large encoder**: RKNN encoder is ~1.3GB due to float16

## Technical Notes

### RKNN Compatibility

The encoder was modified for RKNN compatibility:

1. **Removed SDP**: Stochastic Duration Predictor uses dynamic tensor operations incompatible with RKNN's graph optimizer. Only the deterministic DurationPredictor is used.

2. **Fixed max_frames**: The `sequence_mask` and `generate_path` functions use a fixed maximum output length (2048 frames) instead of dynamic computation.

3. **Dynamic input profiles**: 
   - Encoder: seq_len 32, 64, 128
   - Decoder: frames 512, 1024, 2048

### Float16 Considerations

- Decoder frame sizes are limited to avoid float16 overflow in HiFi-GAN upscaling layers
- Large frame sizes (>2048) may cause register overflow errors

## Troubleshooting

### RKNN Installation Fails

```bash
# Try with specific Python version
python3.10 -m venv venv
source venv/bin/activate
pip install rknn-toolkit2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple
```

### ONNX Version Conflict

Ensure you have `onnx==1.18.0`:
```bash
pip install onnx==1.18.0
```

### Model Download Fails

Check internet connection and HuggingFace access:
```bash
huggingface-cli login
```

## License

MIT License

## Acknowledgments

- [MeloTTS](https://github.com/myshell-ai/MeloTTS) - Original TTS model
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) - Rockchip NPU SDK
- [happyme531/MeloTTS-RKNN2](https://huggingface.co/happyme531/MeloTTS-RKNN2) - Reference implementation
