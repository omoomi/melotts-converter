# RKNN Converter

Download and convert MeloTTS text-to-speech models from HuggingFace to RKNN format for deployment on Rockchip RK3588/RK3576 chipsets.

## Status

| Feature | Status |
|---------|--------|
| Model download | ✅ Working |
| ONNX export | ✅ Working |
| RKNN conversion | ✅ Working |
| ONNX inference | ✅ Working |
| RKNN inference | ⚠️ Partial (BERT issue) |
| CLI commands | ✅ Working |

**Latest Update** (March 2, 2026):
- ✅ All critical fixes applied to runtime
- ✅ ONNX inference fully operational
- ⚠️ RKNN encoder has BERT input format issue (see [API Documentation](docs/API_DOCUMENTATION.md))
- ✅ Server integration complete with ONNX fallback

**Recommendation:** Use ONNX inference for production deployment. RKNN acceleration available but blocked by BERT input format compatibility issue.

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
melotts-converter list
```

#### Download Model

```bash
# Download English model
melotts-converter download --language en

# Download Chinese model
melotts-converter download --language zh
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
# Build complete model pipeline
melotts-converter build --language en --target rk3588
```

This will:
1. Download model from HuggingFace
2. Export to ONNX
3. Extract voice embeddings
4. Convert to RKNN

#### Run Inference

```bash
# English
melotts-converter inference \
    --language en \
    --voice en-us \
    --text "Hello, this is a test." \
    --output test.wav

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
melotts-converter/
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
| encoder.onnx | ~105 MB | ONNX encoder (✅ Working) |
| decoder.onnx | ~56 MB | ONNX decoder (✅ Working) |
| encoder.rknn | ~900 MB | RKNN encoder (⚠️ BERT issue) |
| decoder.rknn | ~155 MB | RKNN decoder (✅ Ready) |

**Why is RKNN larger than ONNX?**

RKNN models are 5-12x larger than ONNX due to:
1. **Dynamic input profiles**: Each profile stores a separate optimized graph (2 profiles = 2x size)
2. **NPU optimization**: Pre-computed operations and NPU-specific kernels
3. **No compression**: RKNN stores raw float16 weights, while ONNX uses protobuf compression

See [RKNN Size Optimization](docs/RKNN_SIZE_OPTIMIZATION.md) for detailed analysis and optimization strategies.

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

1. **RKNN BERT compatibility**: Encoder requires BERT embeddings in specific UNDEFINED layout format (currently blocking RKNN inference)
2. **No quantization**: Models use float16 for full quality
3. **Fixed output length**: Encoder uses max_frames=2048 for RKNN compatibility
4. **No SDP**: Stochastic Duration Predictor excluded (deterministic only)
5. **BERT on CPU**: BERT inference runs on CPU (fast enough)
6. **Large encoder**: RKNN encoder is ~900MB due to float16 and dynamic profiles

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

### RKNN BERT Input Issues

**Status:** Known issue with BERT embedding format in RKNN encoder

**Symptom:** Encoder inference fails with BERT input shape error

**Workaround:** Use ONNX inference instead

```bash
# Set environment variable in melotts-server
export MELORTS_USE_ONNX=true
python server.py
```

For detailed information, see [API Documentation](docs/API_DOCUMENTATION.md) and [RKNN Size Optimization](docs/RKNN_SIZE_OPTIMIZATION.md).

## License

MIT License

## Acknowledgments

- [MeloTTS](https://github.com/myshell-ai/MeloTTS) - Original TTS model
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) - Rockchip NPU SDK
- [happyme531/MeloTTS-RKNN2](https://huggingface.co/happyme531/MeloTTS-RKNN2) - Reference implementation
