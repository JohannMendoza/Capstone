#!/bin/bash
set -e

echo "Checking for H5 model to convert..."

if [ -f "media/improved_pest_model.h5" ]; then
    echo "Found improved_pest_model.h5, attempting conversion to .keras format..."
    python << 'EOF'
import os
from pathlib import Path
import sys

h5_path = Path('media/improved_pest_model.h5')
keras_path = Path('media/improved_pest_model.keras')

if h5_path.exists() and not keras_path.exists():
    try:
        import tensorflow as tf
        print("Loading H5 model...")
        model = tf.keras.models.load_model(str(h5_path), compile=False)
        print("Saving to .keras format...")
        model.save(str(keras_path), save_format='keras')
        print(f"✓ Successfully converted to {keras_path}")
    except Exception as e:
        print(f"⚠ Conversion warning (non-fatal): {type(e).__name__}: {e}")
        sys.exit(0)
elif keras_path.exists():
    print(f"✓ {keras_path} already exists, skipping conversion")
else:
    print("✓ No H5 model found, this is fine")
EOF
else
    echo "ℹ No H5 model found in media/ (this is OK for Railway)"
fi

echo "Done"
