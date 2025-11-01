#!/bin/bash

# Model conversion script with better error handling
echo "[v0] Starting model conversion check..."

MEDIA_DIR="/app/media"
H5_PATH="$MEDIA_DIR/improved_pest_model.h5"
KERAS_PATH="$MEDIA_DIR/improved_pest_model.keras"

# Create media directory if it doesn't exist
mkdir -p "$MEDIA_DIR"

# Check if .keras file already exists
if [ -f "$KERAS_PATH" ]; then
    echo "[v0] ✓ .keras model already exists at $KERAS_PATH"
    exit 0
fi

# Check if .h5 file exists
if [ -f "$H5_PATH" ]; then
    echo "[v0] Found .h5 model at $H5_PATH, attempting conversion..."
    
    python3 << 'PYTHON_SCRIPT'
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='[v0] %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    logger.info("TensorFlow version: " + tf.__version__)
    
    h5_path = "/app/media/improved_pest_model.h5"
    keras_path = "/app/media/improved_pest_model.keras"
    
    if os.path.exists(h5_path):
        file_size = os.path.getsize(h5_path) / (1024 * 1024)
        logger.info(f"Loading H5 model ({file_size:.2f} MB)...")
        
        # Load with compatibility fixes
        try:
            model = tf.keras.models.load_model(
                h5_path, 
                custom_objects={'batch_shape': None},
                compile=False
            )
            logger.info("Model loaded successfully with custom_objects handler")
        except Exception as e1:
            logger.warning(f"First attempt failed: {e1}, trying safe mode...")
            model = tf.keras.models.load_model(h5_path, compile=False)
            logger.info("Model loaded with safe mode")
        
        logger.info("Saving to .keras format...")
        model.save(keras_path, save_format='keras')
        logger.info(f"✓ Successfully converted to {keras_path}")
        
        # Verify file was created
        if os.path.exists(keras_path):
            size = os.path.getsize(keras_path) / (1024 * 1024)
            logger.info(f"✓ Verification successful - {keras_path} ({size:.2f} MB)")
        else:
            logger.error("✗ Conversion failed - output file not created")
            sys.exit(1)
    else:
        logger.info(f"No H5 model found at {h5_path} (normal for Railway)")
        
except Exception as e:
    logger.error(f"✗ Conversion error: {type(e).__name__}: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
PYTHON_SCRIPT
    
    if [ $? -eq 0 ]; then
        echo "[v0] Model conversion completed successfully"
    else
        echo "[v0] Model conversion failed but continuing..."
    fi
else
    echo "[v0] No H5 model found - this is expected on Railway if using pre-converted .keras model"
fi

echo "[v0] Conversion check complete"
exit 0
