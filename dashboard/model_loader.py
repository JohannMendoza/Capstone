import os
import logging
import tensorflow as tf
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}

def load_pest_model(model_path=None):
    """Load pre-trained MobileNetV2 model for pest detection (TensorFlow 2.14.0 compatible)."""
    global _MODEL_CACHE
    
    if 'pest_model' in _MODEL_CACHE:
        logger.info("✅ Using cached pest detection model")
        return _MODEL_CACHE['pest_model']
    
    if model_path is None:
        model_path = os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.keras')
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None
        
        logger.info(f"🔄 Loading pest model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        logger.info("✅ Pest detection model loaded successfully")
        _MODEL_CACHE['pest_model'] = model
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading pest model: {str(e)}")
        return None

def preprocess_image(image_array, target_size=(224, 224)):
    """Preprocess image for MobileNetV2 model."""
    try:
        image = tf.image.resize(image_array, target_size)
        image = image / 127.5 - 1.0
        image = tf.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_pest(model, image_array):
    """Make prediction on image using pest detection model."""
    try:
        if model is None:
            return {'error': 'Model not loaded'}
        
        processed_image = preprocess_image(image_array)
        if processed_image is None:
            return {'error': 'Image preprocessing failed'}
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = int(tf.argmax(predictions[0]))
        confidence = float(tf.reduce_max(predictions[0]))
        
        pest_classes = ['Adristyrannus', 'Aphids', 'Beetle', 'Bugs', 'Mites', 'Weevil', 'Whitefly']
        
        return {
            'predicted_class': pest_classes[predicted_class] if predicted_class < len(pest_classes) else 'Unknown',
            'confidence': confidence,
            'all_predictions': predictions[0].numpy().tolist()
        }
        
    except Exception as e:
        logger.error(f"❌ Error making prediction: {str(e)}")
        return {'error': str(e)}
