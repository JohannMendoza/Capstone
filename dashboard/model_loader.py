"""
Model Loader for Pest Detection
Compatible with TensorFlow 2.14.0
Handles both .keras (new) and .h5 (old) formats
"""

import os
import logging
import tensorflow as tf
import numpy as np
from django.conf import settings
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}

def load_pest_model(model_path=None):
    """
    Load pre-trained MobileNetV2 model for pest detection.
    Compatible with TensorFlow 2.14.0
    Supports both .keras and .h5 formats
    """
    global _MODEL_CACHE
    
    if 'pest_model' in _MODEL_CACHE:
        logger.info("✅ Using cached pest detection model")
        return _MODEL_CACHE['pest_model']
    
    if model_path is None:
        candidate_paths = [
            os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.keras'),
            os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.h5'),
            os.path.join(settings.MEDIA_ROOT, 'model.keras'),
        ]
        
        model_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found model at: {path}")
                break
        
        if model_path is None:
            logger.error(f"❌ Model file not found. Looked for: {candidate_paths}")
            return None
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None
        
        logger.info(f"🔄 Loading pest model from {model_path}")
        
        if model_path.endswith('.h5'):
            # Old format - use safe_mode=False for compatibility
            logger.info("Loading .h5 format model...")
            model = tf.keras.models.load_model(model_path, safe_mode=False)
        else:
            # New .keras format (native to TF 2.14.0)
            logger.info("Loading .keras format model...")
            model = tf.keras.models.load_model(model_path)
        
        logger.info("✅ Pest detection model loaded successfully")
        _MODEL_CACHE['pest_model'] = model
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading pest model from {model_path}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_image(image_array, target_size=(224, 224)):
    """
    Preprocess image for MobileNetV2 model.
    Applies MobileNetV2 preprocessing (normalize to [-1, 1])
    """
    try:
        image = tf.image.resize(image_array, target_size)
        
        image = image / 127.5 - 1.0
        
        image = tf.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_pest(model, image_array):
    """
    Make prediction on image using pest detection model.
    Returns prediction class and confidence score
    """
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
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def load_image_from_file(image_path):
    """Load image from file path"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image, dtype=np.float32)
        return image_array
    except Exception as e:
        logger.error(f"❌ Error loading image: {str(e)}")
        return None
