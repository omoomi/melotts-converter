# RKNN Size Optimization Guide

## Problem

RKNN models are significantly larger than ONNX models:
- **encoder.onnx**: 105 MB → **encoder.rknn**: 896 MB (8.5x)
- **decoder.onnx**: 56 MB → **decoder.rknn**: 155 MB (2.8x)

## Root Causes

### 1. Dynamic Input Profiles (Major Factor)

RKNN stores separate optimized graphs for each dynamic input profile:

| Profiles | Encoder Size | Decoder Size |
|----------|--------------|--------------|
| 1 (static) | ~500 MB | ~120 MB |
| 2 profiles | ~900 MB | ~155 MB |
| 3 profiles | ~1300 MB | ~180 MB |

**Each additional profile adds ~400 MB to encoder, ~35 MB to decoder.**

### 2. NPU Optimization Overhead (5x)

Even with a single static profile, RKNN is ~5x larger than ONNX:
- Graph optimization for NPU execution
- Pre-computed intermediate results
- NPU-specific kernel weights

### 3. No Compression

- **ONNX**: Uses protobuf compression
- **RKNN**: Stores raw float16 values (no compression)

## Optimization Strategies

### Option 1: Reduce Dynamic Profiles (Recommended)

Edit `src/config.py`:

\`\`\`python
# 2 profiles - good balance
dynamic_input_profiles: List[Dict] = field(default_factory=lambda: [
    {"seq_len": 64, "frames": 1024},   # Short sentences
    {"seq_len": 128, "frames": 2048},  # Long sentences
])
\`\`\`

**Savings**: 31% smaller encoder (1300 MB → 900 MB)

### Option 2: Single Static Profile

For production with known input lengths:

\`\`\`python
dynamic_input_profiles: List[Dict] = field(default_factory=lambda: [
    {"seq_len": 64, "frames": 1024},
])
\`\`\`

**Savings**: 62% smaller (1300 MB → 500 MB)

**Trade-off**: Less flexibility

### Option 3: Quantization (Future)

Enable int8 quantization for 2-4x additional reduction.

## Current Configuration

Default: 2 dynamic profiles
- **Total size**: ~1.05 GB (encoder + decoder)
- **Flexibility**: 64-128 phonemes input
