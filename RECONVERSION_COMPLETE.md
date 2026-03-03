# RKNN Decoder Reconversion - Complete

**Date:** March 3, 2026  
**Status:** ✅ Reconversion Complete, Pending Hardware Testing

---

## ✅ Completed Tasks

### 1. Configuration Update
- **File:** `src/config.py:117-126`
- **Change:** Expanded `dynamic_input_profiles` from 2 to 8 profiles
- **Coverage:** 32 to 4096 frames (128x range)

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

### 2. Dependencies Installed
- ✅ RKNN-Toolkit2 2.3.2 (conversion toolkit)
- ✅ melotts-converter 0.1.0 (development mode)

### 3. RKNN Decoder Reconverted
- **Old decoder:** 154.6 MB (2 profiles: 1024, 2048 frames)
- **New decoder:** 367.7 MB (8 profiles: 32, 64, 128, 256, 512, 1024, 2048, 4096 frames)
- **Size increase:** 213.1 MB (137.9%)

### 4. Backup Created
- **File:** `models/en/decoder.rknn.backup.20260303_134251`
- **Size:** 154.6 MB

---

## 📊 Reconversion Results

### Model Size Comparison

| Component | Old Size | New Size | Change |
|-----------|----------|----------|--------|
| encoder.rknn | 896 MB | 896 MB | No change |
| decoder.rknn | 154.6 MB | 367.7 MB | +213.1 MB (+137.9%) |
| **Total** | **1.05 GB** | **1.26 GB** | **+213.1 MB (+20.2%)** |

### Dynamic Profiles Coverage

| Profile | seq_len | frames | Use Case |
|---------|---------|--------|-----------|
| 1 | 8 | 32 | Single words |
| 2 | 16 | 64 | Short phrases |
| 3 | 32 | 128 | Short sentences |
| 4 | 48 | 256 | Medium sentences |
| 5 | 64 | 512 | Medium-long sentences |
| 6 | 96 | 1024 | Long sentences |
| 7 | 128 | 2048 | Very long sentences |
| 8 | 192 | 4096 | Paragraphs |

---

## 🎯 Expected Benefits

### Hybrid Mode Performance

| Text Length | Frames | Expected Inference Time |
|-------------|--------|------------------------|
| Very short | 32-64 | 200-300ms |
| Short | 64-128 | 250-400ms |
| Medium | 128-512 | 400-700ms |
| Long | 512-2048 | 700-1500ms |

### Mode Comparison

| Mode | Encoder | Decoder | Speed | Memory | Status |
|------|---------|---------|-------|--------|--------|
| Full ONNX | CPU | CPU | 1x | 300 MB | ✅ Working |
| **Hybrid** | CPU | NPU | **1.5-2x** | 400 MB | ✅ Ready |
| Full RKNN | NPU | NPU | 3-5x | 200 MB | ⚠️ BERT issue |

---

## ⏳ Pending Tasks (Require RK3588 Hardware)

### 1. Test Hybrid Pipeline on RK3588

**Prerequisites:**
- RK3588/RK3576 device
- Install RKNNLite on target device:
  ```bash
  pip install rknn-toolkit-lite2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple
  ```

**Test Procedure:**
```python
from runtime.tts import TTS

tts = TTS('en')

# Test various text lengths
test_cases = [
    ("Hi", "Single word"),
    ("This is short", "Short sentence"),
    ("Medium length text here", "Medium sentence"),
    ("This is a longer sentence with more words", "Long sentence"),
]

for text, description in test_cases:
    audio = tts.infer(text)
    print(f"{description}: Success")

tts.release()
```

**Expected Result:** All test cases should produce valid audio without shape mismatch errors.

### 2. Verify Dynamic Profile Support

```python
from rknnlite.api import RKNNLite

rknn = RKNNLite()
rknn.load_rknn('models/en/decoder.rknn')
rknn.init_runtime()

# Test dynamic shapes
test_shapes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

for frames in test_shapes:
    try:
        z = np.random.randn(1, 192, frames).astype(np.float16)
        g = np.random.randn(1, 256, 1).astype(np.float16)
        output = rknn.inference(inputs=[z, g])
        print(f"✅ {frames} frames: Accepted")
    except Exception as e:
        print(f"❌ {frames} frames: {e}")
```

**Expected Result:** All 8 profile shapes should be accepted.

### 3. Benchmark Performance

```python
import time
from runtime.tts import TTS

tts = TTS('en')

tests = {
    "32 frames": "Hi",
    "128 frames": "This is short",
    "512 frames": "Medium length text with enough words",
    "2048 frames": "Very long text to test largest profile",
}

for label, text in tests.items():
    start = time.time()
    audio = tts.infer(text)
    elapsed = time.time() - start
    print(f"{label}: {elapsed:.3f}s")

tts.release()
```

---

## 📝 Files Modified/Created

### Modified
1. `src/config.py:117-126` - Updated dynamic profiles

### Created
1. `reconvert_decoder.py` - Reconversion script
2. `reconvert_decoder.sh` - Bash automation script
3. `verify_decoder_profiles.py` - Profile verification tool
4. `test_hybrid_pipeline.py` - Hybrid pipeline testing tool
5. `verify_simple.py` - Simple verification tool
6. `RECONVERSION_SUMMARY.md` - Quick reference
7. `RKNN_DECODER_RECONVERSION_STEPS.md` - Detailed guide
8. `RKNN_DECODER_RECONVERSION_STATUS.txt` - Status report
9. `RECONVERSION_COMPLETE.md` - This document

### Generated
1. `models/en/decoder.rknn` - New decoder (367.7 MB, 8 profiles)

### Backed Up
1. `models/en/decoder.rknn.backup.20260303_134251` - Original decoder (154.6 MB)

---

## 🚀 Deployment Steps

### On Development Machine (Complete ✅)
1. ✅ Updated configuration with 8 dynamic profiles
2. ✅ Installed RKNN-Toolkit2 2.3.2
3. ✅ Installed melotts-converter
4. ✅ Reconverted decoder.rknn
5. ✅ Verified file size and structure

### On RK3588 Target Device (Pending)
1. Transfer models:
   ```bash
   rsync -av models/en/ user@rk3588:/path/to/models/en/
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install rknn-toolkit-lite2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple
   ```

3. Test hybrid pipeline:
   ```python
   from runtime.tts import TTS
   tts = TTS('en')
   tts.tts_to_file("Hello world", 'en-us', 'output.wav')
   ```

4. Verify all text lengths work (from single words to paragraphs)

---

## 🐛 Troubleshooting

### Issue: Shape Mismatch on RK3588

**Error:**
```
E RKNN: rknn_set_input_shapes error, input name = z
E RKNN: while get rknn_tensor_attr[0].name = z, dims = [1, 192, 128]
```

**Solution:**
- Verify decoder.rknn was reconverted with new profiles
- Check file size: should be ~367 MB (not 154 MB)
- Reconvert if needed: `melotts-converter convert --language en --target rk3588`

### Issue: RKNNLite Not Found

**Error:**
```
ModuleNotFoundError: No module named 'rknnlite'
```

**Solution:**
```bash
pip install rknn-toolkit-lite2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple
```

### Issue: Poor Performance

**Symptoms:**
- Inference slower than expected
- High CPU usage

**Diagnosis:**
- Check if RKNN mode is actually being used
- Verify decoder is loaded from .rknn file (not .onnx)
- Monitor NPU usage: `rknpu-smi` (if available)

---

## 📚 References

- [RKNN Decoder Reconversion Specification](./RKNN_DECODER_RECONVERSION.md)
- [RKNN Decoder Reconversion Steps](./RKNN_DECODER_RECONVERSION_STEPS.md)
- [Hybrid Mode Guide](./docs/HYBRID_MODE.md)
- [README](./README.md)

---

**Status:** Reconversion Complete, Awaiting Hardware Testing  
**Last Updated:** March 3, 2026  
**Next Step:** Test on RK3588/RK3576 device
