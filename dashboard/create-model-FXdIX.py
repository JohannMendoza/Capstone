import os
import tensorflow as tf
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def create_pretrained_pest_model(model_path=None):
    """Create pre-trained MobileNetV2 model for pest detection (TensorFlow 2.14.0 compatible)."""
    
    if model_path is None:
        model_path = os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.keras')
    
    logger.info("🔄 Creating pre-trained MobileNetV2 model...")
    
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create model with custom layers
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1./127.5, offset=-1),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(7, activation='softmax')  # 7 pest classes
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("✅ Model created successfully!")
    model.summary()
    
    # Save in new .keras format (TensorFlow 2.14.0 compatible)
    model.save(model_path)
    logger.info(f"✅ Model saved to {model_path}")
    
    return model

if __name__ == '__main__':
    create_pretrained_pest_model()
