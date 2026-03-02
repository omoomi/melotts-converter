# Dynamic Input Profiles Explained

## What Are Dynamic Profiles?

Dynamic profiles allow RKNN models to accept **different input sizes** at runtime. Each profile defines a specific input shape that the model can handle.

## Example: Encoder Profiles

```python
# Profile 1: seq_len=64
[[1, 64], [1, 64], [1, 64], [1, 1024, 64], [1, 768, 64], [1, 256, 1], [1], [1]]

# Profile 2: seq_len=128  
[[1, 128], [1, 128], [1, 128], [1, 1024, 128], [1, 768, 128], [1, 256, 1], [1], [1]]
```

## How They Work

### Single Profile (Static)

```
Model compiled for seq_len=64 ONLY

Runtime:
  Input seq_len=64  → ✓ Works (exact match)
  Input seq_len=50  → ✗ Fails or requires padding
  Input seq_len=100 → ✗ Fails or requires truncation
```

### Multiple Profiles (Dynamic)

```
Model compiled for seq_len=64 AND seq_len=128

Runtime:
  Input seq_len=64  → ✓ Uses profile 1
  Input seq_len=128 → ✓ Uses profile 2
  Input seq_len=50  → ✓ Pads to 64, uses profile 1
  Input seq_len=100 → ✓ Pads to 128, uses profile 2
```

## Practical Differences

| Aspect | Single Profile | Multiple Profiles |
|--------|---------------|-------------------|
| **Size** | ~500 MB | ~500 MB × N profiles |
| **Flexibility** | One input size | Multiple input sizes |
| **Runtime selection** | None | Automatic |
| **Padding waste** | High if mismatch | Lower (picks closest) |
| **Compilation time** | Faster | Slower (N×) |

## Size Impact

Each profile adds ~400 MB to encoder:

```
1 profile (seq_len=64):    ~500 MB
2 profiles (64, 128):      ~900 MB  (+400 MB)
3 profiles (32, 64, 128): ~1300 MB  (+400 MB)
```

## Runtime Behavior

RKNN automatically selects the **smallest profile that fits** your input:

```python
# Example with 2 profiles: [64, 128]

# Input: 50 phonemes
# → Pads to 64, uses profile 1
# → Waste: 14 padding tokens

# Input: 80 phonemes  
# → Pads to 128, uses profile 2
# → Waste: 48 padding tokens

# Input: 64 phonemes
# → Exact match profile 1
# → Waste: 0 padding tokens
```

## When to Use Each

### Single Profile - Use When:
- ✅ Production with known input lengths
- ✅ Processing fixed-length chunks
- ✅ Storage is limited
- ✅ Inputs are uniform (e.g., always ~60 phonemes)

Example:
```python
# TTS for voice assistant - short commands
dynamic_input_profiles = [
    {"seq_len": 32, "frames": 512},  # Short commands only
]
```

### Multiple Profiles - Use When:
- ✅ General-purpose TTS (varied sentence lengths)
- ✅ Don't know input lengths ahead of time
- ✅ Storage is available
- ✅ Want to minimize padding waste

Example:
```python
# General TTS - short and long sentences
dynamic_input_profiles = [
    {"seq_len": 64, "frames": 1024},   # Short: "Hello world"
    {"seq_len": 128, "frames": 2048},  # Long: "The quick brown fox..."
]
```

## Performance Impact

**No runtime performance difference!**

- Inference speed is the same regardless of profile count
- Profile selection happens at model load time
- NPU execution is identical

The only cost is:
- **Storage**: More profiles = larger file
- **Load time**: Slightly longer to load larger file

## Current Configuration

```python
# src/config.py
dynamic_input_profiles = [
    {"seq_len": 64, "frames": 1024},   # Covers 1-64 phonemes
    {"seq_len": 128, "frames": 2048},  # Covers 65-128 phonemes
]
```

**Coverage**: Sentences with 1-128 phonemes (~95% of use cases)
**Size**: 896 MB encoder
**Efficiency**: 
- 1-64 phonemes → 0-63 padding (avg ~20)
- 65-128 phonemes → 0-63 padding (avg ~30)

## Optimization Example

### Before (3 profiles):
```python
[32, 64, 128]  # 1300 MB
```

### After (2 profiles):
```python
[64, 128]      # 896 MB
```

**Trade-off**: 
- Lost: Very short sentences (1-32 phonemes) now pad to 64
- Gained: 404 MB smaller file
- Impact: Minimal - most sentences are >32 phonemes

## Padding Calculation

For a sentence with 50 phonemes:

**With 3 profiles [32, 64, 128]:**
- Picks profile 64
- Pads: 64 - 50 = 14 tokens
- Waste: ~22%

**With 2 profiles [64, 128]:**
- Picks profile 64  
- Pads: 64 - 50 = 14 tokens
- Waste: ~22% (same!)

**With 1 profile [128]:**
- Picks profile 128
- Pads: 128 - 50 = 78 tokens
- Waste: ~61% (much worse!)

## Recommendation

**Default: 2 profiles [64, 128]**
- Good balance of size and flexibility
- Covers most sentence lengths efficiently
- ~900 MB is reasonable for modern devices

**Edge deployment: 1 profile**
- If storage is critical
- If input lengths are predictable
- Can reduce to ~500 MB
