"""
Model Conversion Script
Converts old .h5 models to new .keras format (TensorFlow 2.14.0)
Run this script to convert existing models
"""

import os
import tensorflow as tf
import logging
from django.conf import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_h5_to_keras(h5_path, keras_path):
    """Convert .h5 model to .keras format"""
    try:
        logger.info(f"Converting {h5_path} to {keras_path}...")
        
        # Load old format
        model = tf.keras.models.load_model(h5_path, safe_mode=False)
        
        # Save new format
        model.save(keras_path)
        
        logger.info(f"✅ Model converted successfully: {keras_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main conversion function"""
    media_root = settings.MEDIA_ROOT if hasattr(settings, 'MEDIA_ROOT') else 'media'
    
    h5_models = [
        'improved_pest_model.h5',
        'model.h5',
        'pest_model.h5'
    ]
    
    for h5_name in h5_models:
        h5_path = os.path.join(media_root, h5_name)
        keras_name = h5_name.replace('.h5', '.keras')
        keras_path = os.path.join(media_root, keras_name)
        
        if os.path.exists(h5_path):
            logger.info(f"Found old model: {h5_path}")
            
            if not os.path.exists(keras_path):
                convert_h5_to_keras(h5_path, keras_path)
            else:
                logger.info(f"New format already exists: {keras_path}")

if __name__ == '__main__':
    main()
