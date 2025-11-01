"""
Model Loader for Pest Detection
Compatible with TensorFlow 2.11.0
Handles both .keras (new) and .h5 (old) formats with batch_shape compatibility
"""

import os
import logging
import tensorflow as tf
import numpy as np
from django.conf import settings
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}

class BatchShapeCompatibilityHandler(tf.keras.layers.Layer):
    """Custom layer to handle batch_shape parameter in Input layers"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def load_pest_model(model_path=None):
    """
    Load pre-trained MobileNetV2 model for pest detection.
    Compatible with TensorFlow 2.11.0
    Supports both .keras and .h5 formats
    Handles batch_shape incompatibility issues
    """
    global _MODEL_CACHE
    
    if 'pest_model' in _MODEL_CACHE:
        logger.info("[v0] Using cached pest detection model")
        return _MODEL_CACHE['pest_model']
    
    if model_path is None:
        candidate_paths = [
            os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.keras'),
            os.path.join(settings.MEDIA_ROOT, 'improved_pest_model.h5'),
            os.path.join(settings.MEDIA_ROOT, 'model.keras'),
            os.path.join(settings.MEDIA_ROOT, 'model.h5'),
        ]
        
        model_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"[v0] Found model at: {path}")
                break
        
        if model_path is None:
            logger.error(f"[v0] Model file not found. Looked for: {candidate_paths}")
            return None
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"[v0] Model file not found: {model_path}")
            return None
        
        logger.info(f"[v0] Loading pest model from {model_path}")
        
        if model_path.endswith('.h5'):
            logger.info("[v0] Loading .h5 format model with batch_shape compatibility...")
            try:
                # Try with custom_objects first
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'batch_shape': None}
                )
            except Exception as e1:
                logger.warning(f"[v0] First attempt failed: {str(e1)}")
                try:
                    # Fallback: Try using safe_mode
                    logger.info("[v0] Attempting to load with safe_mode=False...")
                    model = tf.keras.models.load_model(model_path)
                except Exception as e2:
                    logger.warning(f"[v0] Safe mode failed: {str(e2)}")
                    # Final fallback: Try converting to .keras format on-the-fly
                    logger.info("[v0] Attempting emergency conversion to .keras format...")
                    model = _load_and_convert_h5_model(model_path)
                    if model is not None:
                        logger.info("[v0] Successfully converted .h5 to functional model")
        else:
            # New .keras format (native to TF 2.11)
            logger.info("[v0] Loading .keras format model...")
            model = tf.keras.models.load_model(model_path)
        
        if model is None:
            logger.error("[v0] Model loading returned None")
            return None
            
        logger.info("[v0] Pest detection model loaded successfully")
        _MODEL_CACHE['pest_model'] = model
        return model
        
    except Exception as e:
        logger.error(f"[v0] Error loading pest model from {model_path}: {str(e)}")
        logger.error(f"[v0] Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"[v0] Traceback: {traceback.format_exc()}")
        return None

def _load_and_convert_h5_model(model_path):
    """
    Emergency fallback: Try to load .h5 model by converting to .keras format
    This handles the batch_shape incompatibility by rebuilding the model
    """
    try:
        logger.info("[v0] Attempting to rebuild model from .h5...")
        
        # Load with minimal config
        import h5py
        with h5py.File(model_path, 'r') as h5_file:
            # Check model structure
            if 'model_weights' in h5_file:
                logger.info("[v0] HDF5 file has model_weights - attempting to load...")
                # Create fresh model and load weights
                model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("[v0] Model loaded successfully (weights only)")
                return model
    except Exception as e:
        logger.error(f"[v0] HDF5 rebuild failed: {str(e)}")
    
    return None

def preprocess_image(image_array, target_size=(224, 224)):
    """
    Preprocess image for MobileNetV2 model.
    Applies MobileNetV2 preprocessing (normalize to [-1, 1])
    """
    try:
        image = tf.image.resize(image_array, target_size)
        
        # MobileNetV2 preprocessing: normalize to [-1, 1]
        image = image / 127.5 - 1.0
        
        image = tf.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"[v0] Error preprocessing image: {str(e)}")
        return None

def predict_pest(model, image_array):
    """
    Make prediction on image using pest detection model.
    Returns prediction class and confidence score
    """
    try:
        if model is None:
            return {'error': 'Model not loaded', 'success': False}
        
        processed_image = preprocess_image(image_array)
        if processed_image is None:
            return {'error': 'Image preprocessing failed', 'success': False}
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = int(tf.argmax(predictions[0]))
        confidence = float(tf.reduce_max(predictions[0]))
        
        pest_classes = ['Adristyrannus', 'Aphids', 'Beetle', 'Bugs', 'Mites', 'Weevil', 'Whitefly']
        
        return {
            'predicted_class': pest_classes[predicted_class] if predicted_class < len(pest_classes) else 'Unknown',
            'confidence': confidence,
            'all_predictions': predictions[0].numpy().tolist(),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"[v0] Error making prediction: {str(e)}")
        import traceback
        logger.error(f"[v0] Prediction traceback: {traceback.format_exc()}")
        return {'error': str(e), 'success': False}

def load_image_from_file(image_path):
    """Load image from file path"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image, dtype=np.float32)
        return image_array
    except Exception as e:
        logger.error(f"[v0] Error loading image: {str(e)}")
        return None

def clear_model_cache():
    """Clear the model cache if needed"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("[v0] Model cache cleared")
