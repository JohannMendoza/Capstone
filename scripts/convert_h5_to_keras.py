"""
Script to convert .h5 model to .keras format
Run this script to fix batch_shape compatibility issues
Usage: python scripts/convert_h5_to_keras.py
"""

import os
import sys
import tensorflow as tf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_h5_to_keras(h5_path, output_path=None):
    """
    Convert .h5 model to .keras format
    
    Args:
        h5_path: Path to .h5 model file
        output_path: Path to save .keras model (defaults to same dir with .keras extension)
    """
    
    if not os.path.exists(h5_path):
        logger.error(f"Model file not found: {h5_path}")
        return False
    
    if output_path is None:
        output_path = h5_path.replace('.h5', '.keras')
    
    try:
        logger.info(f"Loading .h5 model from: {h5_path}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Load the model
        model = tf.keras.models.load_model(h5_path)
        logger.info("Model loaded successfully!")
        
        # Save in .keras format
        logger.info(f"Saving to .keras format: {output_path}")
        model.save(output_path, save_format='keras')
        
        logger.info(f"Conversion successful!")
        logger.info(f"Original: {h5_path}")
        logger.info(f"Converted: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    # Try to find and convert models in common locations
    media_paths = [
        './media/improved_pest_model.h5',
        './media/model.h5',
        'C:/Users/ASUS/Capstone/media/improved_pest_model.h5',
    ]
    
    converted = False
    for path in media_paths:
        if os.path.exists(path):
            logger.info(f"\nFound model at: {path}")
            if convert_h5_to_keras(path):
                converted = True
            logger.info("-" * 50)
    
    if not converted:
        logger.warning("No .h5 models found in expected locations")
        logger.info("Usage: python scripts/convert_h5_to_keras.py <path_to_model.h5>")
