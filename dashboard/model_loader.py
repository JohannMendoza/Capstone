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

def get_model_path():
    """Get model path with multiple fallbacks for local and Railway deployments"""
    base_paths = [
        os.path.join(settings.MEDIA_ROOT) if hasattr(settings, 'MEDIA_ROOT') else None,
        os.path.join(settings.BASE_DIR, 'media') if hasattr(settings, 'BASE_DIR') else None,
        '/app/media',
        'media',
        './media',
    ]
    
    # Try to find existing media directory
    for base_path in base_paths:
        if base_path and os.path.isdir(base_path):
            logger.info(f"[v0] Using base media path: {base_path}")
            return base_path
    
    # Create and use default
    default_path = os.path.join(settings.MEDIA_ROOT) if hasattr(settings, 'MEDIA_ROOT') else 'media'
    os.makedirs(default_path, exist_ok=True)
    logger.info(f"[v0] Created media directory at: {default_path}")
    return default_path

def find_model_file():
    """Find the best available model file"""
    media_path = get_model_path()
    
    models_to_check = [
        ('improved_pest_model.keras', 'keras'),
        ('improved_pest_model.h5', 'h5'),
        ('model.keras', 'keras'),
        ('model.h5', 'h5'),
    ]
    
    for filename, model_type in models_to_check:
        full_path = os.path.join(media_path, filename)
        if os.path.isfile(full_path) and os.path.getsize(full_path) > 1024 * 1024:
            logger.info(f"[v0] Found valid {model_type} model: {full_path}")
            return full_path, model_type
    
    # Log what we found for debugging
    logger.error(f"[v0] No valid model file found in {media_path}")
    if os.path.isdir(media_path):
        files = os.listdir(media_path)
        logger.error(f"[v0] Files in {media_path}: {files}")
    else:
        logger.error(f"[v0] Media directory does not exist: {media_path}")
    
    return None, None

def load_h5_model(model_path):
    """Load H5 model with batch_shape compatibility"""
    logger.info(f"[v0] Loading H5 model from {model_path}")
    
    try:
        logger.info("[v0] Attempt 1: Loading with custom_objects={'batch_shape': None}...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'batch_shape': None},
            compile=False
        )
        logger.info("[v0] ✓ Success with custom_objects")
        return model
    except Exception as e1:
        logger.warning(f"[v0] Attempt 1 failed: {e1}")
    
    try:
        logger.info("[v0] Attempt 2: Safe mode loading (compile=False)...")
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("[v0] ✓ Success with safe mode")
        return model
    except Exception as e2:
        logger.warning(f"[v0] Attempt 2 failed: {e2}")
    
    try:
        logger.info("[v0] Attempt 3: Direct load...")
        model = tf.keras.models.load_model(model_path)
        logger.info("[v0] ✓ Success with direct load")
        return model
    except Exception as e3:
        logger.error(f"[v0] All H5 loading attempts failed: {e3}")
        return None

def load_pest_model(model_path=None):
    """
    Load pest detection model with automatic fallbacks
    Returns cached model if already loaded
    """
    global _MODEL_CACHE
    
    if 'pest_model' in _MODEL_CACHE and _MODEL_CACHE['pest_model'] is not None:
        logger.info("[v0] Using cached pest detection model")
        return _MODEL_CACHE['pest_model']
    
    # Find model file if not provided
    if model_path is None:
        model_path, model_type = find_model_file()
        if model_path is None:
            logger.error("[v0] Model not found. Please upload improved_pest_model.h5 or .keras to media/ folder")
            return None
    else:
        model_type = 'h5' if model_path.endswith('.h5') else 'keras'
    
    try:
        logger.info(f"[v0] Loading {model_type} model from {model_path}")
        
        if model_type == 'h5':
            model = load_h5_model(model_path)
        else:
            logger.info("[v0] Loading .keras model...")
            model = tf.keras.models.load_model(model_path, compile=False)
        
        if model is None:
            logger.error("[v0] Model loading failed - returned None")
            return None
        
        logger.info("[v0] ✓ Pest detection model loaded successfully")
        _MODEL_CACHE['pest_model'] = model
        return model
        
    except Exception as e:
        logger.error(f"[v0] Error loading model: {e}", exc_info=True)
        return None

def preprocess_image(image_array, target_size=(224, 224)):
    """Preprocess image for MobileNetV2"""
    try:
        image = tf.image.resize(image_array, target_size)
        # MobileNetV2 preprocessing: normalize to [-1, 1]
        image = image / 127.5 - 1.0
        image = tf.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"[v0] Image preprocessing error: {e}")
        return None

def predict_pest(model, image_array):
    """Make pest detection prediction"""
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
        logger.error(f"[v0] Prediction error: {e}", exc_info=True)
        return {'error': str(e), 'success': False}

def load_image_from_file(image_file):
    """Load image from file or file-like object"""
    try:
        image = Image.open(image_file).convert('RGB')
        image_array = np.array(image, dtype=np.float32)
        return image_array
    except Exception as e:
        logger.error(f"[v0] Image loading error: {e}")
        return None

def clear_model_cache():
    """Clear cached model"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("[v0] Model cache cleared")
