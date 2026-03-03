#!/usr/bin/env python3
"""Simple script to reconvert RKNN decoder with new dynamic profiles."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from config import Config
    from core.converter import RKNNConverter

    print("=" * 60)
    print("RKNN Decoder Reconversion with Expanded Dynamic Profiles")
    print("=" * 60)

    # Initialize config and converter
    config = Config()
    converter = RKNNConverter(config)

    # Display current profiles
    print(f"\nCurrent dynamic input profiles:")
    for i, profile in enumerate(config.rknn_config.dynamic_input_profiles, 1):
        print(f"  {i}. seq_len={profile['seq_len']:3}, frames={profile['frames']:4}")

    # Check for decoder.onnx
    model_dir = config.get_model_dir('en')
    decoder_onnx = model_dir / 'decoder.onnx'

    if not decoder_onnx.exists():
        print(f"\n❌ Error: {decoder_onnx} not found")
        print("   Please export the decoder to ONNX first")
        sys.exit(1)

    print(f"\nFound decoder.onnx: {decoder_onnx.stat().st_size / 1024 / 1024:.1f} MB")

    # Perform conversion
    print(f"\nConverting decoder...")
    print("-" * 60)

    decoder_rknn = converter._convert_decoder(
        decoder_onnx,
        model_dir,
        'rk3588',
        quiet=False
    )

    print("-" * 60)
    print(f"\n✅ Conversion complete!")
    print(f"   New decoder: {decoder_rknn}")
    print(f"   Size: {decoder_rknn.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify profiles were used
    print(f"\nExpected decoder dynamic input shapes:")
    decoder_profiles = converter._get_decoder_dynamic_inputs()
    for i, profile in enumerate(decoder_profiles, 1):
        print(f"  {i}. z shape={profile[0]}, g shape={profile[1]}")

except ImportError as e:
    print(f"\n❌ Error: Missing dependency - {e}")
    print("   Install dependencies with: pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
