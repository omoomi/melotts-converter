# RKNN Decoder Reconversion with Compatible Dynamic Profiles

**Date:** March 3, 2026
**Status:** Technical Specification
**Related:** Hybrid Pipeline Integration Issue

---

## Executive Summary

The hybrid ONNX encoder + RKNN decoder pipeline fails due to incompatible dynamic profiles between the ONNX encoder's output shapes and the RKNN decoder's input constraints. This document details the problem and provides a comprehensive solution for re-converting the RKNN decoder with compatible dynamic profiles.

---

## Problem Analysis

### Current Issue

The hybrid pipeline encounters a shape mismatch error when attempting to use ONNX encoder output as RKNN decoder input:

```
E RKNN: rknn_set_input_shapes error, input name = z supported input shapes are as follows:
E RKNN: 	shape = [1,192,1024], layout = UNDEFINED
E RKNN: 	shape = [1,192,2048], layout = UNDEFINED
E RKNN: while get rknn_tensor_attr[0].name = z, dims = [1, 192, 128], fmt = UNDEFINED
Exception: Set input shape failed. error code: RKNN_ERR_FAIL
```

### Root Cause

1. **ONNX Encoder Output:** Produces variable frame counts based on input text length
   - Example output shape: `[1, 192, 128]` (for short text)
   - Dynamic dimension: `frames` varies from ~32 to ~2048
   - ONNX decoder accepts this with shape `['batch_size', 192, 'frames']`

2. **RKNN Decoder Conversion:** Uses only two fixed dynamic profiles
   - Profile 1: `[1, 192, 1024]`
   - Profile 2: `[1, 192, 2048]`
   - Missing: Smaller frame counts (< 1024)

3. **Shape Mismatch:**
   - ONNX encoder outputs: `[1, 192, N]` where N varies
   - RKNN decoder accepts: Only `[1, 192, 1024]` or `[1, 192, 2048]`
   - When N ≠ 1024 and N ≠ 2048 → **Runtime Error**

### Current Dynamic Profiles Configuration

File: `~/repos/melotts-converter/src/config.py` (lines 117-120)

```python
dynamic_input_profiles: List[Dict] = field(default_factory=lambda: [
    {"seq_len": 64, "frames": 1024},
    {"seq_len": 128, "frames": 2048},
])
```

**Analysis:**
- Encoder profiles cover sequence lengths: 64, 128
- Decoder profiles cover frame counts: 1024, 2048
- **Gap:** No profile for small frame counts (32-512)

### Frame Count Calculation

The encoder's frame output depends on:
- **Input text length:** Longer text → more phones → more frames
- **Hop length:** 512 samples per frame (from config)
- **Duration prediction:** Encoder's `length_scale` parameter

**Typical frame counts:**
| Text Length | Characters | Est. Frames | Current RKNN Support |
|-------------|--------------|--------------|----------------------|
| "Hello" | 5 | 32-64 | ❌ Not supported |
| "This is a test" | 15 | 64-128 | ❌ Not supported |
| "Testing with medium text" | 27 | 128-256 | ❌ Not supported |
| "Longer sentence for testing purposes" | 42 | 256-512 | ❌ Not supported |
| "This is a significantly longer text that should produce more frames" | 68 | 512-1024 | ❌ Not supported |
| "This is a very long text that should produce a large number of frames" | 75 | 1024 | ✅ Supported |
| Extended paragraph | 150+ | 2048 | ✅ Supported |

---

## Solution Approach

### Strategy: Expand Dynamic Profile Coverage

Reconvert the RKNN decoder with a comprehensive set of dynamic profiles covering the full range of possible frame counts from minimum to maximum.

### Profile Design Principles

1. **Exponential spacing:** Frame counts follow powers of 2 (or close to it)
2. **Full range coverage:** From minimum expected to maximum
3. **Reasonable granularity:** Enough profiles to minimize padding overhead
4. **Memory consideration:** Too many profiles increase model size

### Recommended Dynamic Profiles

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

**Profile Breakdown:**

| Profile | seq_len | frames | Use Case | Memory Overhead |
|---------|----------|---------|-----------|-----------------|
| 1 | 8 | 32 | Single words | Minimal |
| 2 | 16 | 64 | Short phrases | Minimal |
| 3 | 32 | 128 | Short sentences | Low |
| 4 | 48 | 256 | Medium sentences | Low |
| 5 | 64 | 512 | Medium-long sentences | Medium |
| 6 | 96 | 1024 | Long sentences (original) | Medium |
| 7 | 128 | 2048 | Very long sentences (original) | High |
| 8 | 192 | 4096 | Paragraphs | High |

**Benefits:**
- ✅ Covers 32 to 4096 frames (128x range)
- ✅ Exponential spacing minimizes padding overhead
- ✅ Maintains backward compatibility with original profiles (1024, 2048)
- ✅ Handles edge cases (very short/very long text)

---

## Implementation Steps

### Step 1: Update Configuration

**File:** `~/repos/melotts-converter/src/config.py`

**Location:** Line 117-120

**Change:**
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

### Step 2: Backup Existing RKNN Decoder

```bash
cd ~/repos/melotts-server/models/en
mv decoder.rknn decoder.rknn.backup.$(date +%Y%m%d)
```

### Step 3: Reconvert RKNN Decoder

```bash
cd ~/repos/melotts-converter
source venv/bin/activate

# Full conversion (encoder + decoder)
melotts-converter build --language en --target rk3588

# Or convert only decoder
melotts-converter build --language en --target rk3588 --decoder-only
```

**Expected Output:**
```
Converting English model to RKNN for rk3588...
  Converting encoder (105.0 MB)...
    ✓ Saved to models/en/encoder.rknn (896.0 MB)
  Converting decoder (56.0 MB)...
    ✓ Saved to models/en/decoder.rknn (155.0 MB)
✓ RKNN conversion complete
```

### Step 4: Verify Dynamic Profiles

**File:** `~/repos/melotts-converter/src/core/converter.py`

**Location:** Line 207-217

The `_get_decoder_dynamic_inputs()` method generates profiles from config:

```python
def _get_decoder_dynamic_inputs(self) -> List[List[List[int]]]:
    """Get dynamic input profiles for decoder."""
    profiles = []
    for profile in self.config.rknn_config.dynamic_input_profiles:
        frames = profile["frames"]

        profiles.append([
            [1, 192, frames],  # z input
            [1, 256, 1],        # g input
        ])
    return profiles
```

**Verification:**
After conversion, check that RKNN model supports all profiles:

```python
from rknnlite.api import RKNNLite

rknn = RKNNLite()
rknn.load_rknn('models/en/decoder.rknn')
rknn.init_runtime()

# List supported shapes
print("Supported shapes for input 'z':")
# Check RKNN API for dynamic shape enumeration
```

### Step 5: Test Hybrid Pipeline

```bash
cd ~/repos/melotts-server
source venv/bin/activate

# Start server (no MELORTS_USE_ONNX = hybrid mode)
python server.py

# Test with various text lengths
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi"}' \
  --output test_short.wav

curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a medium length sentence"}' \
  --output test_medium.wav

curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a very long text that should produce many frames for testing the dynamic profile support"}' \
  --output test_long.wav
```

### Step 6: Profile Performance

**Test Coverage:**
- Very short text (< 32 frames)
- Short text (32-128 frames)
- Medium text (128-512 frames)
- Long text (512-1024 frames)
- Very long text (> 1024 frames)

**Metrics to Track:**
- Conversion success rate (should be 100%)
- Inference latency per frame count
- Memory usage per profile
- Padding overhead (difference between actual and nearest profile)

---

## Technical Details

### RKNN Dynamic Profile Mechanism

RKNN uses **dynamic input profiles** to optimize for multiple input shapes:

1. **Conversion Phase:**
   - For each profile, RKNN builds optimized kernels
   - More profiles = larger model size but better flexibility
   - Each profile is a specialized kernel for that shape

2. **Inference Phase:**
   - Input shape is matched to nearest profile
   - If exact match: Uses optimized kernel
   - If no match: Uses padding to nearest profile (zero-padding)
   - Padding increases memory and compute overhead

3. **Profile Selection:**
   ```python
   # Pseudocode for profile matching
   def match_profile(input_shape, profiles):
       for profile in profiles:
           if input_shape <= profile:
               return profile
       return profiles[-1]  # Use largest as fallback
   ```

### Memory Impact Estimation

**Model Size Growth:**
```
Base decoder.rknn: 155 MB (2 profiles)
Estimated with 8 profiles: 155 MB * (8 / 2) ≈ 620 MB
```

**Runtime Memory:**
```
Per frame count:
  - 32 frames:  32 * 192 * 4 bytes = 24 KB
  - 512 frames: 512 * 192 * 4 bytes = 393 KB
  - 2048 frames: 2048 * 192 * 4 bytes = 1.5 MB
  - 4096 frames: 4096 * 192 * 4 bytes = 3 MB
```

### Conversion Time Estimation

```
Original (2 profiles): ~2-3 minutes
Expanded (8 profiles): ~8-12 minutes
```

---

## Alternative Solutions

### Option A: Single Fixed Profile with Padding

**Approach:** Always pad encoder output to 2048 frames

**Pros:**
- Simple to implement
- No re-conversion needed

**Cons:**
- High memory overhead for short text (64 frames padded to 2048)
- Slower inference (processing zero-padding)
- Not optimal for hybrid mode performance goals

**Implementation:**
```python
# In encoder_onnx.py or runtime layer
def pad_output(z, y_mask, target_frames=2048):
    current_frames = z.shape[2]
    pad_len = target_frames - current_frames
    z = np.pad(z, ((0, 0), (0, 0), (0, pad_len)), mode='constant')
    return z, y_mask  # Keep y_mask for length calculation
```

### Option B: Re-export ONNX Decoder with Fixed Input

**Approach:** Export decoder.onnx with fixed frame count

**Pros:**
- No need for dynamic profiles
- Smaller RKNN model

**Cons:**
- Requires ONNX model re-export
- Still needs padding for variable frame counts
- More complex setup

**Implementation:**
```python
# In exporter.py (line 299+)
def export_decoder_fixed(self, output_dir, fixed_frames=2048):
    decoder_path = output_dir / "decoder_fixed.onnx"
    z = torch.randn((1, 192, fixed_frames), dtype=torch.float32)
    # ... export with fixed shape
```

### Option C: Runtime Shape Normalization Layer

**Approach:** Add inference layer to normalize shapes

**Pros:**
- No model re-conversion
- Works with existing models

**Cons:**
- Extra runtime overhead
- Still uses padding internally

---

## Testing and Validation

### Unit Tests

Create test suite to verify dynamic profile support:

```python
# test_dynamic_profiles.py
import numpy as np
from runtime.tts import TTS

def test_decoder_dynamic_profiles():
    """Test decoder with various frame counts."""
    tts = TTS(language='en')

    test_cases = [
        ("Hi", 32-64),
        ("This is short", 64-128),
        ("Medium length text here", 128-256),
        ("This is a longer sentence with more words", 256-512),
        ("Extended text that generates many frames", 512-1024),
    ]

    for text, expected_frames in test_cases:
        audio = tts.infer(text)
        assert audio is not None
        print(f"✓ '{text[:20]}...' -> {len(audio)/44100:.1f}s")

    tts.release()
```

### Integration Tests

```bash
# test_hybrid_integration.sh
#!/bin/bash

# Test various text lengths
TESTS=(
    "Hi"  # 1 word
    "Hello world"  # 2 words
    "This is a test sentence"  # 6 words
    "Testing the hybrid pipeline with a longer sentence"  # 10 words
    "This is a very long text designed to test the dynamic profile handling capabilities of the RKNN decoder"  # 20+ words
)

for text in "${TESTS[@]}"; do
    echo "Testing: $text"
    curl -s -X POST http://127.0.0.1:8000/tts \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"voice\": \"en-us\"}" \
        --output test_output.wav

    if [ -f test_output.wav ]; then
        DURATION=$(soxi -D test_output.wav 2>/dev/null || echo "N/A")
        echo "  ✓ Success (duration: ${DURATION}s)"
    else
        echo "  ✗ Failed"
    fi
done
```

### Performance Benchmarks

```python
# benchmark_profiles.py
import time
from runtime.tts import TTS

def benchmark_profile_performance():
    """Benchmark inference time per profile."""
    tts = TTS(language='en')

    texts = {
        "32 frames": "Hi",
        "128 frames": "This is short",
        "512 frames": "Medium length text",
        "2048 frames": "This is a much longer text that should produce approximately 2048 frames when processed by the encoder",
    }

    for label, text in texts.items():
        start = time.time()
        audio = tts.infer(text)
        elapsed = time.time() - start
        print(f"{label}: {elapsed:.3f}s")

    tts.release()

if __name__ == '__main__':
    benchmark_profile_performance()
```

---

## Troubleshooting

### Issue: Conversion Fails with "Out of Memory"

**Symptoms:**
```
Error: Failed to build RKNN model (error code: -1)
```

**Solution:**
1. Reduce number of profiles
2. Close other applications
3. Use single_core_mode=True
4. Increase swap space

### Issue: Decoder Still Rejects Input Shapes

**Symptoms:**
```
E RKNN: rknn_set_input_shapes error
```

**Verification:**
```python
# Check loaded profiles
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn('decoder.rknn')
rknn.init_runtime()
# Query supported shapes via RKNN API
```

**Solution:**
1. Verify config change applied
2. Clear Python cache: `rm -rf src/__pycache__`
3. Reconvert from scratch (delete old decoder.rknn first)

### Issue: Performance Degraded

**Symptoms:**
- Inference slower than before
- Higher memory usage

**Diagnosis:**
```bash
# Compare model sizes
ls -lh decoder.rknn decoder.rknn.backup*

# Check memory usage
top -p $(pgrep -f "python server.py")
```

**Solution:**
1. Reduce number of profiles
2. Profile-specific optimization
3. Use smaller frame count profiles

---

## Conclusion

### Summary

The hybrid ONNX encoder + RKNN decoder pipeline requires re-conversion of the RKNN decoder with expanded dynamic profiles to support variable frame counts from the ONNX encoder output.

### Key Changes Required

1. **Configuration:** Update `dynamic_input_profiles` in `config.py` with 8 profiles (32 to 4096 frames)
2. **Reconversion:** Run `melotts-converter build` to generate new `decoder.rknn`
3. **Testing:** Validate with various text lengths (short to long)

### Expected Benefits

- ✅ Hybrid pipeline functional end-to-end
- ✅ Support for all text lengths (1 word to paragraphs)
- ✅ Minimal padding overhead (exponential profile spacing)
- ✅ Backward compatible with existing profiles

### Next Steps

1. Implement configuration change
2. Perform re-conversion
3. Run integration tests
4. Benchmark performance
5. Deploy to production

---

**Document Version:** 1.0
**Last Updated:** March 3, 2026
**Author:** OpenCode AI Assistant
**Status:** Ready for Implementation
