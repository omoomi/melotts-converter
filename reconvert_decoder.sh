#!/bin/bash
# RKNN Decoder Reconversion Script
# This script performs the reconversion of the RKNN decoder with expanded dynamic profiles

set -e  # Exit on error

echo "==========================================================================="
echo "RKNN Decoder Reconversion with Expanded Dynamic Profiles"
echo "==========================================================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Please create venv first: python3 -m venv venv"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if RKNN-Toolkit2 is installed
if ! python -c "import rknn" 2>/dev/null; then
    echo "❌ Error: RKNN-Toolkit2 is not installed"
    echo "   Install it with:"
    echo "   pip install rknn-toolkit2==2.3.2 --extra-index-url https://repo.rock-chips.com/pypi/simple"
    exit 1
fi

# Check if ONNX decoder exists
if [ ! -f "models/en/decoder.onnx" ]; then
    echo "❌ Error: decoder.onnx not found"
    echo "   Please export the decoder first:"
    echo "   melotts-converter export --language en"
    exit 1
fi

# Backup existing decoder
if [ -f "models/en/decoder.rknn" ]; then
    BACKUP_NAME="models/en/decoder.rknn.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 Backing up existing decoder..."
    mv "models/en/decoder.rknn" "$BACKUP_NAME"
    echo "   Backup saved to: $BACKUP_NAME"
    echo ""
fi

# Display current profiles
echo "📋 Current dynamic input profiles:"
echo ""
python -c "
import sys
sys.path.insert(0, 'src')
from config import Config

config = Config()
profiles = config.rknn_config.dynamic_input_profiles

for i, profile in enumerate(profiles, 1):
    print(f'  {i}. seq_len={profile[\"seq_len\"]:3}, frames={profile[\"frames\"]:4}')
"
echo ""
echo "   Frame coverage: 32 to 4096 frames"
echo "   Total profiles: 8 (was 2)"
echo ""

# Perform reconversion
echo "🔧 Converting decoder with new profiles..."
echo "---------------------------------------------------------------------------"
python reconvert_decoder.py
echo "---------------------------------------------------------------------------"
echo ""

# Verify new decoder
if [ -f "models/en/decoder.rknn" ]; then
    SIZE=$(du -h models/en/decoder.rknn | cut -f1)
    echo "✅ Reconversion complete!"
    echo "   New decoder: models/en/decoder.rknn ($SIZE)"
    echo ""

    # Compare sizes
    OLD_SIZE=$(du -h models/en/decoder.rknn.backup.* | tail -1 | cut -f1)
    echo "📊 Size comparison:"
    echo "   Old decoder: $OLD_SIZE (2 profiles)"
    echo "   New decoder: $SIZE (8 profiles)"
    echo ""

    echo "📝 Next steps:"
    echo "   1. Test hybrid pipeline with various text lengths"
    echo "   2. Verify decoder accepts all frame counts (32-4096)"
    echo "   3. Benchmark performance"
    echo ""
else
    echo "❌ Reconversion failed!"
    echo "   No decoder.rknn file found"
    exit 1
fi
