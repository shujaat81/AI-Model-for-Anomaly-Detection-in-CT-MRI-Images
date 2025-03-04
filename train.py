import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))
from config import *
from data.data_loader import load_dataset, load_test_data
from models.autoencoder import build_convolutional_autoencoder, compute_reconstruction_error
from models.feature_extractor import build_resnet_feature_extractor, extract_features
from models.one_class_svm import OneClassSVMDetector
from utils.visualization import plot_reconstructions, plot_anomaly_scores
from utils.metrics import find_optimal_threshold

def train_autoencoder(data):
    """Train the autoencoder model."""
    print("Training autoencoder model...")
    
    # Get data
    train_generator = data['ct']['train_generator']
    val_generator = data['ct']['val_generator']
    
    # Build model
    autoencoder, encoder = build_convolutional_autoencoder(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        latent_dim=LATENT_DIM
    )
    
    # Print model summary
    autoencoder.summary()
    
    # Create callbacks
    checkpoint_path = MODELS_DIR / "autoencoder_checkpoint.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
    
    # Train the model
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    history = autoencoder.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Save the final model
    autoencoder.save(MODELS_DIR / "autoencoder_final.h5")
    encoder.save(MODELS_DIR / "encoder_final.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['lr'])
    plt.title('Learning Rate')
    plt.ylabel('LR')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "autoencoder_training_history.png")
    plt.close()
    
    # Visualize reconstructions
    val_images = data['ct']['val_images']
    sample_images = val_images[:10]
    reconstructions = autoencoder.predict(sample_images)
    
    plot_reconstructions(
        sample_images, 
        reconstructions, 
        n=10, 
        save_path=RESULTS_DIR / "autoencoder_reconstructions.png"
    )
    
    return autoencoder, encoder

def train_one_class_svm(data, feature_extractor=None, autoencoder=None):
    """Train a One-Class SVM for anomaly detection."""
    print("Training One-Class SVM model...")
    
    # Get normal data (CT images)
    normal_images = data['ct']['train_images']
    
    # Extract features using either feature extractor or autoencoder
    if feature_extractor is not None:
        print("Extracting features using ResNet...")
        features = extract_features(feature_extractor, normal_images)
    elif autoencoder is not None:
        print("Extracting features using autoencoder...")
        # Use the reconstruction error as a feature
        reconstructions = autoencoder.predict(normal_images)
        features = np.abs(normal_images - reconstructions)
    else:
        # Use the raw images as features
        features = normal_images
    
    # Initialize and train One-Class SVM
    ocsvm = OneClassSVMDetector(nu=0.05)  # nu is the expected proportion of outliers
    ocsvm.fit(features)
    
    # Save the model
    ocsvm.save(MODELS_DIR / "one_class_svm.joblib")
    
    return ocsvm

def main():
    """Main training function."""
    print("Starting training process...")
    
    # Load dataset
    data = load_dataset()
    
    # Train autoencoder
    autoencoder, encoder = train_autoencoder(data)
    
    # Calculate reconstruction errors for normal data
    normal_images = data['ct']['train_images']
    normal_recon_errors = compute_reconstruction_error(autoencoder, normal_images)
    
    # Find optimal threshold
    threshold = find_optimal_threshold(normal_recon_errors, contamination=0.01)
    print(f"Optimal threshold for anomaly detection: {threshold:.6f}")
    
    # Plot anomaly score distribution
    plot_anomaly_scores(
        normal_recon_errors, 
        threshold=threshold,
        save_path=RESULTS_DIR / "anomaly_score_distribution.png"
    )
    
    # Build feature extractor
    feature_extractor = build_resnet_feature_extractor(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    
    # Train One-Class SVM
    ocsvm = train_one_class_svm(data, feature_extractor=feature_extractor)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 