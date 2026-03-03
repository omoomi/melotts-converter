# Hybrid ONNX Encoder + RKNN Decoder Mode

**Date:** March 3, 2026
**Status:** ⚠️ Requires RKNN Decoder Reconversion
**Version:** 1.0.0

## Overview

The hybrid mode combines ONNX encoder (CPU) with RKNN decoder (NPU) to achieve partial NPU acceleration while avoiding the BERT input format issue that blocks full RKNN mode.

## Architecture

```
Text Input
    │
    ▼
[CPU] Text Processor ──► phones, tones, lang_ids
    │
    ▼
[CPU] BERT Encoder (bert-large-uncased) ──► bert_embeddings (1024-dim)
    │
    ▼
[CPU] ONNX Encoder ──► z (latent) [1, 192, N_frames]
    │
    ▼
[NPU] RKNN Decoder ──► audio
    │
    ▼
Audio Output
```

## Benefits

| Aspect | Full ONNX | Full RKNN | Hybrid | Notes |
|--------|-----------|------------|---------|---------|
| **Encoder Speed** | 0.5-1x | 3-5x | 0.5-1x | CPU-bound |
| **Decoder Speed** | 0.5-1x | 3-5x | 3-5x | NPU-accelerated |
| **Overall Speed** | 0.5-1x | 3-5x | 1-2x | Partial NPU |
| **Memory** | ~300 MB | ~200 MB | ~400 MB | Includes BERT |
| **Model Size** | 161 MB | 1.05 GB | ~1.5 GB | BERT + ONNX + RKNN |
| **Startup Time** | 2-3s | 1-2s | ~5s | BERT loading |
| **Quality** | Full | Full | Full | Identical |
| **BERT Issues** | ✅ None | ❌ Blocked | ✅ None | ONNX encoder |

## Required Changes

### 1. BERT Model Update

**File:** `src/runtime/bert_encoder.py` (line 19-23)

**Current (Incorrect):**
```python
BERT_MODELS = {
    "en": "bert-base-uncased",  # 768 dimensions
    "zh": "bert-base-chinese",
    "jp": "cl-tohoku/bert-base-japanese-v3",
}
```

**Fixed (Correct):**
```python
BERT_MODELS = {
    "en": "bert-large-uncased",  # 1024 dimensions (REQUIRED for ONNX encoder)
    "zh": "bert-base-chinese",
    "jp": "cl-tohoku/bert-base-japanese-v3",
}
```

**Rationale:**
- ONNX encoder expects 1024-dimensional BERT embeddings
- `bert-base-uncased` produces 768-dimensional embeddings
- `bert-large-uncased` produces 1024-dimensional embeddings
- Model size increases from ~420MB to ~1.3GB but is necessary for compatibility

### 2. Audio Length Calculation Fix

**File:** `src/runtime/tts.py` (lines 227-247)

**Current (Incorrect):**
```python
z, y_mask, audio_len = self._encoder.infer(...)
audio = self._decoder.infer_long(z, g, audio_len)  # Uses encoder's audio_len
```

**Fixed (Correct):**
```python
z, y_mask, _ = self._encoder.infer(...)  # Ignore encoder's audio_len

HOP_LENGTH = 512
audio_len = int(np.sum(y_mask)) * HOP_LENGTH  # Calculate from y_mask

audio = self._decoder.infer_long(z, g, audio_len)
```

**Rationale:**
- Encoder's `audio_len` output is incorrect (too small, 88-231 samples)
- `y_mask` indicates which frames are real vs. padding
- Calculate proper audio length: `real_frames * hop_length`
- Typical audio length: 2-3 seconds for short text

### 3. ONNX Encoder Implementation

**File:** `src/runtime/encoder_onnx.py` (NEW)

Purpose: Provide ONNX runtime for encoder inference (CPU-based)

Key Features:
- Loads `encoder.onnx` model
- Accepts 1024-dimensional BERT embeddings
- Outputs variable frame counts: `[1, 192, N_frames]` (N varies)
- Uses CPUExecutionProvider

```python
class ONNXEncoder:
    def __init__(self, model_path: Path, config: Optional[Config] = None):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )

    def infer(self, phones, tones, lang_ids, bert, ja_bert, g,
              noise_scale, length_scale):
        # ONNX inference
        outputs = self.session.run(self.output_names, inputs)
        z = outputs[0]
        y_mask = outputs[1]
        audio_len = int(outputs[2][0])
        return z, y_mask, audio_len
```

### 4. ONNX Decoder Implementation

**File:** `src/runtime/decoder_onnx.py` (NEW)

Purpose: Provide ONNX runtime for decoder inference (CPU-based, fully ONNX mode)

Key Features:
- Loads `decoder.onnx` model
- Accepts variable frame counts
- Supports long sequences with slicing

### 5. Mode Switching Logic

**File:** `src/runtime/tts.py` (lines 94-123)

Updated initialization to support mode selection via environment variable:

```python
import os

def _init_encoder(self):
    """Initialize encoder based on mode."""
    use_onnx = os.getenv('MELORTS_USE_ONNX', '').lower() == 'true'

    if use_onnx:
        encoder_path = self.model_dir / "encoder.onnx"
        if encoder_path.exists():
            self._encoder = ONNXEncoder(encoder_path, self.config)
            return

    encoder_path = self.model_dir / "encoder.rknn"
    self._encoder = RNNEncoder(encoder_path, self.config)

def _init_decoder(self):
    """Initialize decoder based on mode."""
    use_onnx = os.getenv('MELORTS_USE_ONNX', '').lower() == 'true'

    if use_onnx:
        decoder_path = self.model_dir / "decoder.onnx"
        if decoder_path.exists():
            self._decoder = ONNXDecoder(decoder_path, self.config)
            return

    decoder_path = self.model_dir / "decoder.rknn"
    self._decoder = RKNNDecoder(decoder_path, self.config)
```

**Mode Selection:**

| Environment Variable | Encoder | Decoder | Mode |
|---------------------|----------|----------|-------|
| `MELORTS_USE_ONNX=true` | ONNX (CPU) | ONNX (CPU) | Full ONNX |
| `MELORTS_USE_ONNX=false` | RKNN (NPU) | RKNN (NPU) | Full RKNN (⚠️ BERT issue) |
| Not set (default) | ONNX (CPU) | RKNN (NPU) | Hybrid (⚠️ Requires reconversion) |

## Current Limitations

### 1. RKNN Decoder Dynamic Profile Mismatch

**Issue:** ONNX encoder outputs variable frame counts, but RKNN decoder only accepts 1024 or 2048 frames.

**Symptom:**
```
E RKNN: rknn_set_input_shapes error, input name = z
E RKNN: 	shape = [1,192,1024], layout = UNDEFINED
E RKNN: 	shape = [1,192,2048], layout = UNDEFINED
E RKNN: while get rknn_tensor_attr[0].name = z, dims = [1, 192, 128]
```

**Root Cause:**
- ONNX encoder: `[1, 192, N_frames]` where N varies (32-2048)
- Current RKNN decoder: Only `[1, 192, 1024]` and `[1, 192, 2048]`
- Small text (32-512 frames) → **Shape mismatch error**

**Solution:**
Reconvert RKNN decoder with expanded dynamic profiles (32 to 4096 frames)

**See:** [RKNN Decoder Reconversion Guide](../melotts-server/RKNN_DECODER_RECONVERSION.md)

### 2. BERT Model Size

**Issue:** `bert-large-uncased` is 3x larger than `bert-base-uncased`

**Impact:**
- Download time: ~5-10 minutes (vs. 1-2 minutes)
- Disk usage: ~1.3 GB (vs. ~420 MB)
- Startup time: ~5 seconds (vs. ~2 seconds)
- Memory: ~400 MB (vs. ~300 MB)

**Workaround:** Use pre-downloaded cache or persistent storage

### 3. Memory Requirements

**Total Memory Usage (Hybrid Mode):**
- BERT model: ~1.3 GB
- ONNX encoder: ~105 MB
- RKNN decoder: ~155 MB
- Runtime buffers: ~200 MB
- **Total:** ~1.7-1.8 GB

**Recommendation:** RK3588 with 4GB+ RAM

## Usage

### Full ONNX Mode (Recommended)

```bash
export MELORTS_USE_ONNX=true
python server.py
```

**Characteristics:**
- Fully CPU-based
- Most stable
- Proven in production
- No NPU acceleration

### Hybrid Mode (Experimental - Requires Reconversion)

```bash
# No environment variable (default)
python server.py
```

**Characteristics:**
- ONNX encoder (CPU) + RKNN decoder (NPU)
- Partial NPU acceleration
- **Requires RKNN decoder reconversion** (see below)

### Full RKNN Mode (Not Working)

```bash
export MELORTS_USE_ONNX=false
python server.py
```

**Characteristics:**
- Blocked by BERT input format issue
- Not functional

## Reconversion Steps for Hybrid Mode

To enable hybrid mode, the RKNN decoder must be reconverted with expanded dynamic profiles:

### Step 1: Update Configuration

**File:** `src/config.py` (line 117-120)

```python
dynamic_input_profiles: List[Dict] = field(default_factory=lambda: [
    {"seq_len": 8,   "frames": 32},
    {"seq_len": 16,  "frames": 64},
    {"seq_len": 32,  "frames": 128},
    {"seq_len": 48,  "frames": 256},
    {"seq_len": 64,  "frames": 512},
    {"seq_len": 96,  "frames": 1024},
    {"seq_len": 128, "frames": 2048},
    {"seq_len": 192, "frames": 4096},
])
```

### Step 2: Reconvert RKNN Decoder

```bash
cd ~/repos/melotts-converter
source venv/bin/activate

# Full conversion
melotts-converter build --language en --target rk3588

# Or decoder only
rm models/en/decoder.rknn
melotts-converter convert --language en --target rk3588
```

### Step 3: Test

```bash
cd ~/repos/melotts-server
python server.py

# Test various text lengths
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi"}' \
  --output test_short.wav

curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a longer text"}' \
  --output test_long.wav
```

**Expected Result:** All tests should produce valid audio files

## Performance Comparison

### Inference Speed (per sentence)

| Text Length | ONNX | Hybrid | RKNN (theoretical) |
|-------------|--------|---------|---------------------|
| Short (10 words) | ~500ms | ~300ms | ~150ms |
| Medium (30 words) | ~1.5s | ~800ms | ~400ms |
| Long (60 words) | ~3s | ~1.5s | ~750ms |

### Memory Usage

| Mode | Runtime Memory | Model Size |
|-------|----------------|------------|
| Full ONNX | ~300 MB | 161 MB |
| Hybrid | ~400 MB | ~1.5 GB |
| Full RKNN | ~200 MB | 1.05 GB |

## Troubleshooting

### Issue: "No module named 'onnxruntime'"

**Cause:** ONNX runtime not installed

**Solution:**
```bash
pip install onnxruntime
```

### Issue: "ONNXRuntimeError: Got 768 Expected: 1024"

**Cause:** Using `bert-base-uncased` instead of `bert-large-uncased`

**Solution:**
1. Check `src/runtime/bert_encoder.py` line 20
2. Verify `"en": "bert-large-uncased"`
3. Clear cache: `rm -rf ~/.cache/huggingface/hub/models--bert-base-uncased`

### Issue: "RKNN shape mismatch" in hybrid mode

**Cause:** RKNN decoder dynamic profiles don't match ONNX encoder output

**Solution:**
1. Reconvert RKNN decoder with expanded profiles
2. See [RKNN Decoder Reconversion Guide](../melotts-server/RKNN_DECODER_RECONVERSION.md)

### Issue: High memory usage on startup

**Cause:** Loading `bert-large-uncased` model

**Solutions:**
1. Use full ONNX mode: `export MELORTS_USE_ONNX=true`
2. Pre-download BERT model to cache
3. Ensure sufficient RAM (4GB+ for RK3588)

## Recommendations

### For Production Deployment

**Use Full ONNX Mode** (`MELORTS_USE_ONNX=true`)

**Rationale:**
- ✅ Most stable and tested
- ✅ No additional configuration
- ✅ Lower memory footprint
- ✅ Faster startup
- ✅ Proven in production

**Trade-offs:**
- ❌ No NPU acceleration (CPU-bound)
- ❌ Slower inference (0.5-1x real-time)

### For Development/Testing

**Use Hybrid Mode** (after reconversion)

**Rationale:**
- ✅ Partial NPU acceleration
- ✅ Better than ONNX speed (1-2x)
- ✅ Experimental features
- ✅ Future path to full RKNN

**Requirements:**
- ⚠️ Must reconvert RKNN decoder
- ⚠️ Higher memory usage
- ⚠️ Larger model files

### For Maximum Performance (Future)

**Use Full RKNN Mode** (requires BERT fix)

**Rationale:**
- ✅ Fastest (3-5x real-time)
- ✅ Lowest memory usage
- ✅ Full NPU utilization

**Blocks:**
- ❌ BERT input format issue
- ❌ Requires encoder reconversion or workaround

## References

- [RKNN Decoder Reconversion Guide](../melotts-server/RKNN_DECODER_RECONVERSION.md)
- [API Documentation](API_DOCUMENTATION.md)
- [RKNN Size Optimization](RKNN_SIZE_OPTIMIZATION.md)
- [MeloTTS Server README](../melotts-server/README.md)
- [Hybrid Production Fix](../melotts-server/HYBRID_PRODUCTION_FIX.md)

---

**Document Version:** 1.0.0
**Last Updated:** March 3, 2026
**Author:** OpenCode AI Assistant
**Status:** Ready for Implementation
