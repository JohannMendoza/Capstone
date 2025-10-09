import os
import numpy as np
import tensorflow as tf
from django.conf import settings
import traceback
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Global variable to store the model
model = None

def create_model():
    """
    Recreate the exact same model architecture as used during training
    """
    try:
        # This should match your original model architecture exactly
        new_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: healthy, powdery mildew, leaf rust
        ])
        
        # Compile the model with the same settings as during training
        new_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return new_model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        traceback.print_exc()
        return None

# Initialize the model
def initialize_model():
    """
    Initialize the model and return it
    """
    global model
    
    if model is not None:
        logger.info("Model already loaded")
        return model
    
    try:
        # Create a new model with the same architecture
        logger.info("Creating model with the same architecture as training")
        new_model = create_model()
        
        if new_model is None:
            logger.error("Failed to create model")
            return None
        
        # Check if the model file exists
        model_path = getattr(settings, 'MODEL_PATH', os.path.join(settings.BASE_DIR, 'model', 'model.h5'))
        logger.info(f"Looking for model at: {model_path}")
        
        if os.path.exists(model_path):
            logger.info(f"Found model at {model_path}")
            
            try:
                # Try to load weights directly
                logger.info("Attempting to load weights")
                new_model.load_weights(model_path)
                logger.info("Successfully loaded weights")
                
                # Test the model with a dummy input to ensure it works
                logger.info("Testing model with dummy input")
                dummy_input = np.zeros((1, 150, 150, 3))
                predictions = new_model.predict(dummy_input)
                logger.info(f"Prediction shape: {predictions.shape}")
                logger.info(f"Prediction values: {predictions[0]}")
                
                model = new_model  # Set the global model
                return model
            except Exception as e:
                logger.error(f"Error loading weights: {e}")
                
                # Try loading as a full model
                try:
                    logger.info("Attempting to load as full model")
                    full_model = tf.keras.models.load_model(model_path)
                    logger.info("Successfully loaded full model")
                    
                    # Test the model
                    dummy_input = np.zeros((1, 150, 150, 3))
                    predictions = full_model.predict(dummy_input)
                    logger.info(f"Prediction shape: {predictions.shape}")
                    logger.info(f"Prediction values: {predictions[0]}")
                    
                    model = full_model
                    return model
                except Exception as e2:
                    logger.error(f"Error loading full model: {e2}")
                    logger.warning("Using model with random weights - predictions will be inaccurate")
                    model = new_model  # Still use the model even if weights couldn't be loaded
                    return model
        else:
            logger.error(f"Model file not found at {model_path}")
            logger.warning("Using model with random weights - predictions will be inaccurate")
            model = new_model  # Use the model with random weights
            return model
            
    except Exception as e:
        logger.error(f"Error in initialize_model: {e}")
        traceback.print_exc()
        return None

# Load the model - initialize it when this module is imported
model = initialize_model()

def load_model():
    """
    Get the model, initializing it if necessary
    """
    global model
    
    if model is None:
        logger.info("Model not loaded, attempting to initialize")
        model = initialize_model()
        
    return model

# Class names for predictions - EXACT MATCH with training labels
CLASS_NAMES = ['Healthy', 'Leaf Rust', 'Powdery Mildew']