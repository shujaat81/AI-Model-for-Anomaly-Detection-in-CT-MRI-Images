import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    BatchNormalization, LeakyReLU, Flatten, Dense, Reshape
)
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def build_convolutional_autoencoder(input_shape=(128, 128, 3), latent_dim=LATENT_DIM):
    """Build a convolutional autoencoder for anomaly detection."""
    # Encoder
    input_img = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Flatten and encode to latent space
    x = Flatten()(x)
    encoded = Dense(latent_dim)(x)
    
    # Decoder
    x = Dense(16 * 16 * 128)(encoded)
    x = Reshape((16, 16, 128))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    
    # Output layer
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create model
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    
    # Compile model
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                        loss='mean_squared_error')
    
    return autoencoder, encoder

def compute_reconstruction_error(autoencoder, images):
    """Compute reconstruction error for anomaly detection."""
    reconstructions = autoencoder.predict(images)
    mse = np.mean(np.square(images - reconstructions), axis=(1, 2, 3))
    return mse

def compute_reconstruction_error_in_batches(autoencoder, images, batch_size=16):
    """Compute reconstruction error for anomaly detection in batches to save memory."""
    total_samples = len(images)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    
    all_errors = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        
        batch_images = images[start_idx:end_idx]
        batch_reconstructions = autoencoder.predict(batch_images, verbose=0)
        batch_mse = np.mean(np.square(batch_images - batch_reconstructions), axis=(1, 2, 3))
        
        all_errors.extend(batch_mse)
        
        # Print progress
        print(f"Processed batch {i+1}/{num_batches}", end="\r")
    
    print("\nReconstruction error calculation complete.")
    return np.array(all_errors)

def detect_anomalies(autoencoder, images, threshold=None, reference_errors=None):
    """
    Detect anomalies based on reconstruction error.
    
    Args:
        autoencoder: Trained autoencoder model
        images: Images to check for anomalies
        threshold: Anomaly threshold (if None, will be calculated from reference_errors)
        reference_errors: Reconstruction errors from normal data (for threshold calculation)
        
    Returns:
        anomaly_scores: Reconstruction errors for each image
        is_anomaly: Boolean array indicating anomalies
        threshold: The threshold used for detection
    """
    # Compute reconstruction errors
    anomaly_scores = compute_reconstruction_error(autoencoder, images)
    
    # Calculate threshold if not provided
    if threshold is None and reference_errors is not None:
        # Set threshold as mean + 3*std of normal data errors
        threshold = np.mean(reference_errors) + 3 * np.std(reference_errors)
    elif threshold is None:
        threshold = ANOMALY_THRESHOLD
    
    # Detect anomalies
    is_anomaly = anomaly_scores > threshold
    
    return anomaly_scores, is_anomaly, threshold 