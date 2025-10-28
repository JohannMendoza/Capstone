"""
Convert to pre-trained MobileNetV2 model compatible with TensorFlow 2.14.0
Run this once: python convert_to_pretrained.py
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_pretrained_pest_model(model_path='improved_pest_model.keras'):
    """Create a pre-trained MobileNetV2 model for pest detection."""
    
    print("🔄 Creating pre-trained MobileNetV2 model...")
    
    # Load pre-trained MobileNetV2 (trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers for pest classification
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./127.5, offset=-1),  # Normalize for MobileNetV2
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(5, activation='softmax')  # 5 pest classes (adjust if needed)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created successfully!")
    print(f"📊 Model summary:")
    model.summary()
    
    # Save in new .keras format (TensorFlow 2.14.0 compatible)
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    return model

if __name__ == '__main__':
    create_pretrained_pest_model()
