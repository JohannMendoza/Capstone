"""
Model Loader for Pest Detection
Optimized for Railway deployment
Auto-downloads model from Supabase if missing
"""

import os
import logging
import tensorflow as tf
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}

# ====================================
# CONFIGURATION
# ====================================
SUPABASE_MODEL_URL = "https://lsevojehwifimphiulfa.supabase.co/storage/v1/object/public/Model/improved_pest_model.h5"
MODEL_FILENAME = "improved_pest_model.h5"
MEDIA_DIR = "/app/media"  # Railway-safe persistent path


# ====================================
# FILE HANDLING
# ====================================
def ensure_media_dir():
    """Ensure /app/media exists"""
    os.makedirs(MEDIA_DIR, exist_ok=True)
    return MEDIA_DIR


def get_local_model_path():
    """Return model path if it exists"""
    media_dir = ensure_media_dir()
    model_path = os.path.join(media_dir, MODEL_FILENAME)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1024 * 1024:
        logger.info(f"[v0] Found local model: {model_path}")
        return model_path
    logger.warning(f"[v0] Model not found in {media_dir}")
    return None


def download_model_from_supabase():
    """Download model file from Supabase storage"""
    try:
        media_dir = ensure_media_dir()
        model_path = os.path.join(media_dir, MODEL_FILENAME)
        logger.info(f"[v0] Downloading model from Supabase → {model_path}")

        response = requests.get(SUPABASE_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"[v0] ✅ Model downloaded and saved to {model_path}")
            return model_path
        else:
            logger.error(f"[v0] ❌ Failed to download model (HTTP {response.status_code})")
            return None
    except Exception as e:
        logger.error(f"[v0] ❌ Error downloading model: {e}", exc_info=True)
        return None


# ====================================
# MODEL LOADING
# ====================================
def load_pest_model(model_path=None):
    """Load pest detection model (with caching)"""
    global _MODEL_CACHE

    if 'pest_model' in _MODEL_CACHE:
        logger.info("[v0] Using cached pest model")
        return _MODEL_CACHE['pest_model']

    # Try local path first
    if model_path is None:
        model_path = get_local_model_path()

    # Download if not found
    if model_path is None:
        model_path = download_model_from_supabase()

    if not model_path or not os.path.exists(model_path):
        logger.error("[v0] ❌ Model not available after download attempt")
        return None

    # Load TensorFlow model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        _MODEL_CACHE['pest_model'] = model
        logger.info(f"[v0] ✅ Pest detection model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"[v0] ❌ Error loading model: {e}", exc_info=True)
        return None


# ====================================
# IMAGE PREPROCESSING & PREDICTION
# ====================================
def preprocess_image(image_array, target_size=(224, 224)):
    """Resize and normalize image"""
    try:
        image = tf.image.resize(image_array, target_size)
        image = image / 127.5 - 1.0
        image = tf.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"[v0] Image preprocessing error: {e}")
        return None


def predict_pest(model, image_array):
    """Run prediction using the pest model"""
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
            'all_predictions': predictions[0].tolist(),
            'success': True
        }

    except Exception as e:
        logger.error(f"[v0] Prediction error: {e}", exc_info=True)
        return {'error': str(e), 'success': False}


def load_image_from_file(image_file):
    """Convert uploaded file to numpy array"""
    try:
        image = Image.open(image_file).convert('RGB')
        return np.array(image, dtype=np.float32)
    except Exception as e:
        logger.error(f"[v0] Image loading error: {e}")
        return None


def clear_model_cache():
    """Clear in-memory model cache"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("[v0] Model cache cleared")
