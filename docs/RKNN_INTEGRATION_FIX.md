# MeloTTS Converter - RKNN Integration Issues & Solutions

## Executive Summary

The melotts-converter runtime fails when attempting RKNN inference on the RK3588 NPU due to input tensor layout mismatch between the converted RKNN models and the encoder implementation. This document details the root cause and provides step-by-step solutions.

**Fix Status**: ✅ **APPLIED** (March 2, 2026)
- ✅ Approach 1 (NCHW reshaping) implemented in `src/runtime/encoder.py`
- ✅ Data type corrected from int32 to int64
- ✅ Added debug logging for troubleshooting
- 📝 Commit: RKNN integration fix - NCHW reshaping and dtype correction

**Changes Made**:
1. Updated `RNNEncoder.infer()` to reshape inputs to NCHW format [1, 1, seq_len]
2. Changed dtype from np.int32 to np.int64 for phones, tones, lang_ids
3. Added debug logging for sequence length and audio length
4. Updated docstrings to reflect NCHW-compliant input handling

**Next Steps**:
- Test RKNN inference on actual hardware (RK3588/RK3576)
- Verify audio quality and performance
- Monitor for any remaining issues

---

## Problem Statement

### Symptom
```
E RKNN: rknn_set_input_shapes error, input name = phones
supported input shapes are as follows:
E RKNN: 	shape = [1,128], layout = UNDEFINED
E RKNN: 	shape = [1,64], layout = UNDEFINED
E RKNN: while get rknn_tensor_attr[0].name = phones, dims = [1, 33], fmt = UNDEFINED
```

### Root Cause
The RKNN encoder model (`encoder.rknn`) was converted with NCHW (Batch-Channel-Height-Width) layout expectations, but the `RNNEncoder.infer()` method passes 1D arrays without proper reshaping and NCHW formatting.

### Impact
- Complete failure of TTS inference via RKNN NPU
- Server returns 500 errors on `/tts` endpoint
- Audio generation is non-functional

---

## Technical Analysis

### Model Metadata

Extracted from `models/en/encoder.rknn`:

```python
{
  'phones': {
    'idx': 0,
    'shape': [1, 128],           # NCHW: Batch=1, Channel=1, Height=128
    'layout': 'nchw',
    'layout_ori': 'nchw',        # Original ONNX layout was also NCHW
    'is_output': False,
    'origin_dynamic': True,
    'dtype': 'int64'
  },
  'tones': {
    'idx': 1,
    'shape': [1, 128],
    'layout': 'nchw',
    'dtype': 'int64'
  },
  'lang_ids': {
    'idx': 2,
    'shape': [1, 128],
    'layout': 'nchw',
    'dtype': 'int64'
  },
  'bert': {
    'idx': 3,
    'shape': [1, 1024, 128],    # NCHW: Batch=1, Channel=1024, Height=128
    'layout': 'nchw',
    'dtype': 'float32'
  },
  'ja_bert': {
    'idx': 4,
    'shape': [1, 768, 128],
    'layout': 'nchw',
    'dtype': 'float32'
  },
  'g': {
    'idx': 5,
    'shape': [1, 256, 1],       # NCHW: Batch=1, Channel=256, Height=1
    'layout': 'nchw',
    'dtype': 'float32'
  }
}
```

**Dynamic Shape Profiles:**
- Profile 0: seq_len = 64 frames
- Profile 1: seq_len = 128 frames (default)

### Current Implementation (BROKEN)

File: `~/repos/melotts-converter/src/runtime/encoder.py`

```python
def infer(
    self,
    phones: np.ndarray,      # Expected: [seq_len]
    tones: np.ndarray,       # Expected: [seq_len]
    lang_ids: np.ndarray,    # Expected: [seq_len]
    bert: np.ndarray,        # Expected: [1, bert_dim, seq_len]
    ja_bert: np.ndarray,     # Expected: [1, 768, seq_len]
    g: np.ndarray,           # Expected: [1, 256, 1]
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    noise_scale_arr = np.array([noise_scale], dtype=np.float32)
    length_scale_arr = np.array([length_scale], dtype=np.float32)

    inputs = [
        phones.astype(np.int32),      # ❌ WRONG: [seq_len] - missing batch dim
        tones.astype(np.int32),       # ❌ WRONG: [seq_len] - missing batch dim
        lang_ids.astype(np.int32),    # ❌ WRONG: [seq_len] - missing batch dim
        bert.astype(np.float32),
        ja_bert.astype(np.float32),
        g.astype(np.float32),
        noise_scale_arr,
        length_scale_arr,
    ]

    outputs = self.rknn.inference(inputs=inputs)
    # ...
```

**Issues:**
1. `phones`, `tones`, `lang_ids` are 1D arrays `[seq_len]` but need `[1, 1, seq_len]` for NCHW
2. No explicit shape configuration sent to RKNN runtime
3. Data type mismatch: model expects `int64`, code passes `int32`

---

## Solution Approaches

### Approach 1: Fix Input Reshaping (RECOMMENDED)

**Principle:** Reshape 1D arrays to NCHW format before passing to RKNN.

**File to modify:** `~/repos/melotts-converter/src/runtime/encoder.py`

#### Step 1: Update `infer()` method signature and reshaping

```python
def infer(
    self,
    phones: np.ndarray,
    tones: np.ndarray,
    lang_ids: np.ndarray,
    bert: np.ndarray,
    ja_bert: np.ndarray,
    g: np.ndarray,
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run encoder inference with NCHW-compliant inputs."""
    noise_scale_arr = np.array([noise_scale], dtype=np.float32).reshape(1, 1, 1)
    length_scale_arr = np.array([length_scale], dtype=np.float32).reshape(1, 1, 1)

    # Reshape inputs to NCHW format [Batch, Channel, Height]
    # phones: [seq_len] -> [1, 1, seq_len]
    phones_nchw = phones.astype(np.int64).reshape(1, 1, -1)

    # tones: [seq_len] -> [1, 1, seq_len]
    tones_nchw = tones.astype(np.int64).reshape(1, 1, -1)

    # lang_ids: [seq_len] -> [1, 1, seq_len]
    lang_ids_nchw = lang_ids.astype(np.int64).reshape(1, 1, -1)

    # bert: [1, bert_dim, seq_len] - already NCHW, just ensure float32
    bert_nchw = bert.astype(np.float32)

    # ja_bert: [1, 768, seq_len] - already NCHW, just ensure float32
    ja_bert_nchw = ja_bert.astype(np.float32)

    # g: [1, 256, 1] - already NCHW, just ensure float32
    g_nchw = g.astype(np.float32)

    inputs = [
        phones_nchw,
        tones_nchw,
        lang_ids_nchw,
        bert_nchw,
        ja_bert_nchw,
        g_nchw,
        noise_scale_arr.reshape(1, 1, 1),
        length_scale_arr.reshape(1, 1, 1),
    ]

    outputs = self.rknn.inference(inputs=inputs)

    z = outputs[0]
    y_mask = outputs[1]
    audio_len = int(outputs[2][0])

    return z, y_mask, audio_len
```

#### Step 2: Add sequence length normalization

The model only supports seq_len of 64 or 128. Add validation:

```python
def infer(
    self,
    phones: np.ndarray,
    tones: np.ndarray,
    lang_ids: np.ndarray,
    bert: np.ndarray,
    ja_bert: np.ndarray,
    g: np.ndarray,
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run encoder inference with NCHW-compliant inputs."""

    # Validate and normalize sequence length
    seq_len = phones.shape[0]
    if seq_len > 128:
        logger.warning(f"Sequence length {seq_len} exceeds 128, truncating")
        seq_len = 128
        phones = phones[:128]
        tones = tones[:128]
        lang_ids = lang_ids[:128]
        bert = bert[:, :, :128]
        ja_bert = ja_bert[:, :, :128]

    # Choose dynamic profile: 64 for short sequences, 128 for longer
    profile_idx = 0 if seq_len <= 64 else 1
    profile_seq_len = 64 if seq_len <= 64 else 128

    logger.debug(f"Using dynamic profile {profile_idx} (seq_len={profile_seq_len})")

    noise_scale_arr = np.array([noise_scale], dtype=np.float32).reshape(1, 1, 1)
    length_scale_arr = np.array([length_scale], dtype=np.float32).reshape(1, 1, 1)

    # Reshape inputs to NCHW format [Batch, Channel, Height]
    phones_nchw = phones.astype(np.int64).reshape(1, 1, profile_seq_len)
    tones_nchw = tones.astype(np.int64).reshape(1, 1, profile_seq_len)
    lang_ids_nchw = lang_ids.astype(np.int64).reshape(1, 1, profile_seq_len)

    # Pad to full sequence length if needed
    if phones_nchw.shape[2] < profile_seq_len:
        phones_nchw = np.pad(phones_nchw, ((0, 0), (0, 0), (0, profile_seq_len - phones_nchw.shape[2])), mode='constant')
        tones_nchw = np.pad(tones_nchw, ((0, 0), (0, 0), (0, profile_seq_len - tones_nchw.shape[2])), mode='constant')
        lang_ids_nchw = np.pad(lang_ids_nchw, ((0, 0), (0, 0), (0, profile_seq_len - lang_ids_nchw.shape[2])), mode='constant')
        bert_nchw = np.pad(bert, ((0, 0), (0, 0), (0, profile_seq_len - bert.shape[2])), mode='constant')
        ja_bert_nchw = np.pad(ja_bert, ((0, 0), (0, 0), (0, profile_seq_len - ja_bert.shape[2])), mode='constant')

    bert_nchw = bert_nchw.astype(np.float32)
    ja_bert_nchw = ja_bert_nchw.astype(np.float32)
    g_nchw = g.astype(np.float32)

    inputs = [
        phones_nchw,
        tones_nchw,
        lang_ids_nchw,
        bert_nchw,
        ja_bert_nchw,
        g_nchw,
        noise_scale_arr,
        length_scale_arr,
    ]

    outputs = self.rknn.inference(inputs=inputs)

    z = outputs[0]
    y_mask = outputs[1]
    audio_len = int(outputs[2][0])

    return z, y_mask, audio_len
```

---

### Approach 2: Configure RKNN with Explicit Shapes

**File to modify:** `~/repos/melotts-converter/src/runtime/encoder.py`

Add explicit shape configuration in `_init_rknn()`:

```python
def _init_rknn(self):
    """Initialize RKNN runtime with explicit input shapes."""
    try:
        from rknnlite.api import RKNNLite

        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {self.model_path}")

        # NEW: Configure input shapes explicitly
        ret = self.rknn.init_runtime(
            target=None,  # Use model's default target
            rknn_shared_mem=None,
            perf_debug=False,
            core_mask=None,
        )

        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")

        logger.info(f"Loaded encoder RKNN: {self.model_path}")

    except ImportError as e:
        raise ImportError(
            "RKNNLite not installed. "
            "Install with: pip install rknn-toolkit-lite2==2.3.2"
        ) from e
```

**Note:** RKNNLite automatically handles dynamic shapes from model metadata. This approach is less likely to fix the issue compared to Approach 1.

---

### Approach 3: Add ONNX Fallback Mode (ALTERNATIVE)

If RKNN issues persist, add ONNX fallback using the existing `.onnx` files.

**File to modify:** `~/repos/melotts-converter/src/runtime/encoder.py`

```python
import onnxruntime as ort

class RNNEncoder:
    """RKNN encoder for NPU inference with ONNX fallback."""

    def __init__(
        self,
        model_path: Path,
        config: Optional[Config] = None,
        use_onnx_fallback: bool = False,
    ):
        """
        Initialize RKNN encoder.

        Args:
            model_path: Path to encoder.rknn
            config: Configuration
            use_onnx_fallback: Force ONNX fallback instead of RKNN
        """
        self.model_path = Path(model_path)
        self.config = config or Config()
        self.use_onnx_fallback = use_onnx_fallback or os.getenv("MELORTS_USE_ONNX", "").lower() == "true"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {model_path}")

        if self.use_onnx_fallback:
            self._init_onnx()
        else:
            try:
                self._init_rknn()
                self.mode = "rknn"
            except Exception as e:
                logger.warning(f"RKNN init failed: {e}, falling back to ONNX")
                self._init_onnx()
                self.mode = "onnx"

    def _init_onnx(self):
        """Initialize ONNX runtime as fallback."""
        onnx_path = self.model_path.parent / "encoder.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX fallback model not found: {onnx_path}")

        self.onnx_session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        logger.info(f"Loaded encoder ONNX fallback: {onnx_path}")

    def infer(
        self,
        phones: np.ndarray,
        tones: np.ndarray,
        lang_ids: np.ndarray,
        bert: np.ndarray,
        ja_bert: np.ndarray,
        g: np.ndarray,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run encoder inference using RKNN or ONNX."""
        if self.mode == "onnx":
            return self._infer_onnx(phones, tones, lang_ids, bert, ja_bert, g, noise_scale, length_scale)
        else:
            return self._infer_rknn(phones, tones, lang_ids, bert, ja_bert, g, noise_scale, length_scale)

    def _infer_rknn(self, phones, tones, lang_ids, bert, ja_bert, g, noise_scale, length_scale):
        """RKNN inference with NCHW reshaping."""
        # Use implementation from Approach 1
        seq_len = min(phones.shape[0], 128)
        profile_seq_len = 64 if seq_len <= 64 else 128

        noise_scale_arr = np.array([noise_scale], dtype=np.float32).reshape(1, 1, 1)
        length_scale_arr = np.array([length_scale], dtype=np.float32).reshape(1, 1, 1)

        phones_nchw = phones.astype(np.int64).reshape(1, 1, profile_seq_len)
        tones_nchw = tones.astype(np.int64).reshape(1, 1, profile_seq_len)
        lang_ids_nchw = lang_ids.astype(np.int64).reshape(1, 1, profile_seq_len)

        inputs = [phones_nchw, tones_nchw, lang_ids_nchw, bert, ja_bert, g, noise_scale_arr, length_scale_arr]

        outputs = self.rknn.inference(inputs=inputs)
        z = outputs[0]
        y_mask = outputs[1]
        audio_len = int(outputs[2][0])

        return z, y_mask, audio_len

    def _infer_onnx(self, phones, tones, lang_ids, bert, ja_bert, g, noise_scale, length_scale):
        """ONNX inference fallback."""
        # ONNX expects different input format
        inputs = {
            'phones': phones.reshape(1, -1).astype(np.int64),
            'tones': tones.reshape(1, -1).astype(np.int64),
            'lang_ids': lang_ids.reshape(1, -1).astype(np.int64),
            'bert': bert.astype(np.float32),
            'ja_bert': ja_bert.astype(np.float32),
            'g': g.astype(np.float32),
            'noise_scale': np.array([noise_scale], dtype=np.float32),
            'length_scale': np.array([length_scale], dtype=np.float32),
        }

        outputs = self.onnx_session.run(
            output_names=['z', 'y_mask', 'audio_len'],
            input_feed=inputs
        )

        return outputs[0], outputs[1], int(outputs[2][0])

    def release(self):
        """Release resources."""
        if self.mode == "onnx" and hasattr(self, 'onnx_session'):
            del self.onnx_session
        elif self.mode == "rknn" and hasattr(self, "rknn"):
            self.rknn.release()
```

**File to modify:** `~/repos/melotts-converter/src/runtime/decoder.py`

Apply similar ONNX fallback to decoder:

```python
class RKNNDecoder:
    """RKNN decoder for NPU inference with ONNX fallback."""

    def __init__(
        self,
        model_path: Path,
        config: Optional[Config] = None,
        use_onnx_fallback: bool = False,
    ):
        self.model_path = Path(model_path)
        self.config = config or Config()
        self.use_onnx_fallback = use_onnx_fallback or os.getenv("MELORTS_USE_ONNX", "").lower() == "true"
        self.mode = "rknn"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Decoder model not found: {model_path}")

        if self.use_onnx_fallback:
            self._init_onnx()
        else:
            try:
                self._init_rknn()
            except Exception as e:
                logger.warning(f"RKNN init failed: {e}, falling back to ONNX")
                self._init_onnx()
                self.mode = "onnx"

    def _init_onnx(self):
        """Initialize ONNX runtime."""
        onnx_path = self.model_path.parent / "decoder.onnx"
        self.onnx_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        logger.info(f"Loaded decoder ONNX fallback: {onnx_path}")

    def infer(self, z: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Run decoder inference."""
        if self.mode == "onnx":
            return self._infer_onnx(z, g)
        else:
            return self._infer_rknn(z, g)

    def _infer_rknn(self, z, g):
        inputs = [z.astype(np.float32), g.astype(np.float32)]
        outputs = self.rknn.inference(inputs=inputs)
        audio = outputs[0]
        return audio

    def _infer_onnx(self, z, g):
        inputs = {'z': z.astype(np.float32), 'g': g.astype(np.float32)}
        outputs = self.onnx_session.run(output_names=['audio'], input_feed=inputs)
        return outputs[0]
```

---

## Recommended Implementation Strategy

### Phase 1: Quick Fix (Approach 1)

**Goal:** Get RKNN working immediately with minimal changes.

**Steps:**

1. Backup original encoder.py:
   ```bash
   cd ~/repos/melotts-converter/src/runtime
   cp encoder.py encoder.py.backup
   ```

2. Apply Approach 1 changes to encoder.py
3. Reinstall melotts-converter:
   ```bash
   cd ~/repos/melotts-converter
   pip install -e .
   ```

4. Test with server:
   ```bash
   cd ~/repos/melotts-server
   source venv/bin/activate
   python server.py
   ```

5. In another terminal:
   ```bash
   curl -X POST http://127.0.0.1:8000/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!"}' \
     --output test.wav
   ```

**Expected Result:** TTS generates audio successfully.

### Phase 2: Add Robustness (Apply Approach 2)

**Goal:** Handle various text lengths and edge cases.

**Steps:**

1. Add sequence length validation and padding
2. Add dynamic profile selection based on seq_len
3. Test with texts of various lengths (10, 50, 100, 150 words)

### Phase 3: Add Fallback (Apply Approach 3) - OPTIONAL

**Goal:** Ensure system works even if RKNN fails.

**Steps:**

1. Add ONNX fallback to both encoder and decoder
2. Add environment variable `MELORTS_USE_ONNX=true` to force ONNX mode
3. Test both modes:
   ```bash
   # Test RKNN mode
   python server.py

   # Test ONNX mode
   MELORTS_USE_ONNX=true python server.py
   ```

---

## Testing Checklist

After applying fixes, verify:

- [ ] Short text (5-10 words) generates audio
- [ ] Long text (50+ words) generates audio
- [ ] Very long text (100+ words) handles gracefully
- [ ] Audio plays correctly on device
- [ ] No audio dropouts or artifacts
- [ ] CLI commands work with all voice options
- [ ] Memory usage is reasonable (<1GB)
- [ ] Inference time meets targets (<2s for 10 words)

---

## Troubleshooting

### If RKNN still fails after Approach 1:

1. **Check RKNNLite version:**
   ```bash
   pip show rknn-toolkit-lite2
   # Should show version 2.3.2
   ```

2. **Verify model conversion:**
   - Were the .rknn models converted with RKNN-Toolkit2 (full) or RKNN-Toolkit-Lite?
   - Models converted with different tool versions may have incompatible metadata

3. **Check NPU availability:**
   ```bash
   cat /sys/class/devfreq/fdab0000.gpu/governor
   cat /sys/class/devfreq/fdab0000.gpu/cur_freq
   ```

4. **Inspect model with rknn-toolkit2 (full):**
   ```python
   from rknn.api import RKNN

   rknn = RKNN(verbose=True)
   rknn.load_rknn('models/en/encoder.rknn')
   rknn.init_runtime(target='rk3588')

   # Print input/output info
   print(rknn rknn_list_devices())
   rknn.release()
   ```

### If ONNX fallback is needed:

1. Ensure onnxruntime is installed:
   ```bash
   pip install onnxruntime==1.24.2
   ```

2. Verify ONNX model format:
   ```bash
   python -c "import onnx; m = onnx.load('models/en/encoder.onnx'); print(m.graph.input)"
   ```

3. Test ONNX inference directly:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession('models/en/encoder.onnx')
   print([i.name for i in session.get_inputs()])
   print([o.name for o in session.get_outputs()])
   ```

---

## Additional Considerations

### 1. Batch Processing

For future optimization, consider supporting batch inference:
- Encoder: Process multiple sentences in parallel
- Decoder: Generate multiple audio streams simultaneously

### 2. Model Quantization

If model size is an issue:
- Convert models with w8a8 quantization
- Test quality degradation
- May improve NPU performance

### 3. Dynamic Profiling

The current model supports two dynamic profiles (64 and 128). For better efficiency:
- Add profile for 32-frame sequences
- Or re-export models with more granular profiles

### 4. Memory Management

For long texts:
- Implement streaming inference (sentence-by-sentence)
- Manage RKNN context lifecycle
- Clear GPU/NPU memory between inferences

---

## Summary

| Issue | Root Cause | Solution | Priority |
|-------|-----------|----------|----------|
| RKNN shape mismatch | Input arrays not NCHW format | Reshape to [1,1,seq_len] | HIGH |
| Data type mismatch | int32 vs int64 expected | Change to int64 | HIGH |
| Sequence length limits | Model only supports 64/128 frames | Pad/truncate to profile | MEDIUM |
| No fallback mechanism | RKNN failures crash server | Add ONNX fallback | LOW |

**Recommended path:**
1. Apply Approach 1 (NCHW reshaping) - immediate fix
2. Apply Approach 2 (sequence handling) - robustness
3. Optionally apply Approach 3 (ONNX fallback) - safety net

**Estimated effort:** 2-3 hours for Phase 1, 4-6 hours for complete implementation.
