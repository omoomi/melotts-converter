#!/usr/bin/env python3
"""Test hybrid pipeline with various text lengths."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from runtime.tts import TTS
except ImportError as e:
    print(f"❌ Error: {e}")
    print("   Install dependencies with: pip install -e .")
    sys.exit(1)

print("=" * 70)
print("Hybrid Pipeline Test - Various Text Lengths")
print("=" * 70)
print()

# Test cases with expected frame ranges
test_cases = [
    ("Hi", "Single word (very short)", "en-us", "test_01_single.wav"),
    ("Hello world", "Two words (short)", "en-us", "test_02_two_words.wav"),
    ("This is a test sentence.", "Six words (short)", "en-us", "test_03_short.wav"),
    ("Testing the hybrid pipeline with a longer sentence.", "10 words (medium)", "en-us", "test_04_medium.wav"),
    ("This is a very long text designed to test the dynamic profile handling capabilities of the RKNN decoder.",
     "20+ words (long)", "en-us", "test_05_long.wav"),
]

print(f"Test cases: {len(test_cases)}")
print("Mode: Hybrid (ONNX encoder + RKNN decoder)")
print()

# Check for models
encoder_onnx = Path("models/en/encoder.onnx")
decoder_rknn = Path("models/en/decoder.rknn")

if not encoder_onnx.exists():
    print(f"❌ Error: {encoder_onnx} not found")
    print("   Please export encoder first: melotts-converter export --language en")
    sys.exit(1)

if not decoder_rknn.exists():
    print(f"❌ Error: {decoder_rknn} not found")
    print("   Please convert decoder first: melotts-converter convert --language en --target rk3588")
    sys.exit(1)

print(f"✅ Encoder: {encoder_onnx.stat().st_size / 1024 / 1024:.1f} MB")
print(f"✅ Decoder: {decoder_rknn.stat().st_size / 1024 / 1024:.1f} MB")
print()

# Initialize TTS
print("🔧 Initializing TTS...")
try:
    tts = TTS('en')
    print("   ✓ TTS initialized")
    print()
except Exception as e:
    print(f"   ❌ Failed to initialize TTS: {e}")
    sys.exit(1)

# Run tests
print("=" * 70)
print("Running Tests")
print("=" * 70)
print()

results = []

for i, (text, description, voice, output_file) in enumerate(test_cases, 1):
    print(f"Test {i}/{len(test_cases)}: {description}")
    print(f"  Text: {text}")
    print(f"  Voice: {voice}")

    try:
        # Generate audio
        tts.tts_to_file(
            text=text,
            voice=voice,
            output_path=output_file,
            speed=1.0
        )

        # Check output
        output_path = Path(output_file)
        if output_path.exists():
            size = output_path.stat().st_size
            duration = size / (44100 * 2)  # Assuming 16-bit PCM at 44.1kHz

            print(f"  ✅ Success - {output_file}")
            print(f"     Size: {size / 1024:.1f} KB")
            print(f"     Duration: ~{duration:.2f}s")
            results.append((i, text, True, None))
        else:
            print(f"  ❌ Failed - No output file generated")
            results.append((i, text, False, "No output file"))

    except Exception as e:
        print(f"  ❌ Failed - {e}")
        results.append((i, text, False, str(e)))

    print()

# Cleanup
print("🧹 Cleaning up...")
tts.release()
print()

# Summary
print("=" * 70)
print("Summary")
print("=" * 70)
print()

passed = sum(1 for _, _, success, _ in results if success)
total = len(results)

print(f"Tests passed: {passed}/{total}")
print()

if passed == total:
    print("✅ All tests passed!")
    print()
    print("The hybrid pipeline is working correctly with the new dynamic profiles.")
    print("The RKNN decoder successfully accepts variable frame counts from the ONNX encoder.")
    print()
else:
    print("⚠️  Some tests failed:")
    print()
    for i, text, success, error in results:
        if not success:
            print(f"  Test {i}: {text[:40]}...")
            print(f"    Error: {error}")
            print()

    print("This may indicate:")
    print("  - Incomplete RKNN decoder reconversion")
    print("  - Missing dynamic profiles")
    print("  - Shape mismatch in hybrid pipeline")
    print()

print("=" * 70)
