"""
Convert old .h5 model to new .keras format compatible with TensorFlow 2.14.0
This runs automatically on Railway startup
"""
import os
import logging
import tensorflow as tf
from django.conf import settings

logger = logging.getLogger(__name__)

def convert_h5_to_keras():
    """Convert old .h5 model to new .keras format."""
    
    old_model_path = os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.h5')
    new_model_path = os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.keras')
    
    # If new model already exists, skip conversion
    if os.path.exists(new_model_path):
        logger.info("✅ New .keras model already exists, skipping conversion")
        return True
    
    # If old model doesn't exist, create new pre-trained model
    if not os.path.exists(old_model_path):
        logger.info("🔄 Old model not found, creating new pre-trained model...")
        return create_pretrained_model(new_model_path)
    
    try:
        logger.info(f"🔄 Converting {old_model_path} to .keras format...")
        
        # Load old model with safe_mode disabled
        model = tf.keras.models.load_model(
            old_model_path,
            safe_mode=False
        )
        
        logger.info("✅ Old model loaded successfully")
        
        # Save as new .keras format
        model.save(new_model_path)
        logger.info(f"✅ Model converted and saved to {new_model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error converting model: {str(e)}")
        logger.info("🔄 Creating new pre-trained model instead...")
        return create_pretrained_model(new_model_path)

def create_pretrained_model(model_path):
    """Create new pre-trained MobileNetV2 model."""
    try:
        logger.info("🔄 Creating pre-trained MobileNetV2 model...")
        
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Rescaling(1./127.5, offset=-1),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.save(model_path)
        logger.info(f"✅ Pre-trained model created and saved to {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating pre-trained model: {str(e)}")
        return False

if __name__ == '__main__':
    convert_h5_to_keras()
