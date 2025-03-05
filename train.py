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
from models.autoencoder import (
    build_convolutional_autoencoder,
    compute_reconstruction_error,
)
from models.feature_extractor import build_resnet_feature_extractor, extract_features
from models.one_class_svm import OneClassSVMDetector
from utils.visualization import plot_reconstructions, plot_anomaly_scores
from utils.metrics import find_optimal_threshold


def train_autoencoder(data):
    """Train the autoencoder model."""
    print("Training autoencoder model...")

    # Get data
    train_generator = data["ct"]["train_generator"]
    val_generator = data["ct"]["val_generator"]

    # Build model
    autoencoder, encoder = build_convolutional_autoencoder(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), latent_dim=LATENT_DIM
    )

    # Print model summary
    autoencoder.summary()

    # Create callbacks
    checkpoint_path = MODELS_DIR / "autoencoder_checkpoint.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        ),
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
        callbacks=callbacks,
    )

    # Save the final model
    autoencoder.save(MODELS_DIR / "autoencoder_final.h5")
    encoder.save(MODELS_DIR / "encoder_final.h5")

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "autoencoder_training_history.png")
    plt.close()

    # Visualize reconstructions
    val_images = data["ct"]["val_images"]
    sample_images = val_images[:10]
    reconstructions = autoencoder.predict(sample_images)

    plot_reconstructions(
        sample_images,
        reconstructions,
        n=10,
        save_path=RESULTS_DIR / "autoencoder_reconstructions.png",
    )

    return autoencoder, encoder


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
        batch_mse = np.mean(
            np.square(batch_images - batch_reconstructions), axis=(1, 2, 3)
        )

        all_errors.extend(batch_mse)

        # Print progress
        print(f"Processed batch {i+1}/{num_batches}", end="\r")

    print("\nReconstruction error calculation complete.")
    return np.array(all_errors)


def main():
    """Main training function."""
    print("Starting training process...")

    # Load dataset
    data = load_dataset()

    # Train autoencoder
    autoencoder, encoder = train_autoencoder(data)

    # Calculate reconstruction errors for normal data in batches
    print("Calculating reconstruction errors...")
    normal_images = data["ct"]["train_images"]
    normal_recon_errors = compute_reconstruction_error_in_batches(
        autoencoder, normal_images
    )

    # Find optimal threshold
    threshold = find_optimal_threshold(normal_recon_errors, contamination=0.01)
    print(f"Optimal threshold for anomaly detection: {threshold:.6f}")

    # Plot anomaly score distribution
    plot_anomaly_scores(
        normal_recon_errors,
        threshold=threshold,
        save_path=RESULTS_DIR / "anomaly_score_distribution.png",
    )

    # Build feature extractor
    print("Building feature extractor...")
    feature_extractor = build_resnet_feature_extractor(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # Train One-Class SVM with batch processing for feature extraction
    ocsvm = train_one_class_svm_with_batches(data, feature_extractor=feature_extractor)

    print("Training completed successfully!")


def train_one_class_svm_with_batches(
    data, feature_extractor=None, autoencoder=None, batch_size=16
):
    """Train a One-Class SVM for anomaly detection with batch processing."""
    print("Training One-Class SVM model...")

    # Get normal data (CT images)
    normal_images = data["ct"]["train_images"]
    total_samples = len(normal_images)
    num_batches = (total_samples + batch_size - 1) // batch_size

    all_features = []

    # Extract features in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)

        batch_images = normal_images[start_idx:end_idx]

        if feature_extractor is not None:
            print(
                f"Extracting features using ResNet... Batch {i+1}/{num_batches}",
                end="\r",
            )
            batch_features = extract_features(feature_extractor, batch_images)

            # Check the shape of features
            if len(batch_features.shape) > 2:
                # If features are 4D (batch_size, height, width, channels)
                # Take mean across spatial dimensions to reduce size
                batch_features = np.mean(batch_features, axis=(1, 2))

            # Ensure features are 2D (batch_size, features)
            batch_features = batch_features.reshape(batch_features.shape[0], -1)

        elif autoencoder is not None:
            print(
                f"Extracting features using autoencoder... Batch {i+1}/{num_batches}",
                end="\r",
            )
            batch_reconstructions = autoencoder.predict(batch_images, verbose=0)
            batch_features = np.abs(batch_images - batch_reconstructions)
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
        else:
            batch_features = batch_images.reshape(batch_images.shape[0], -1)

        all_features.append(batch_features)

    # Combine all batches
    features = np.vstack(all_features)
    print(f"\nExtracted features shape: {features.shape}")

    # Initialize and train One-Class SVM
    print("Training One-Class SVM...")
    ocsvm = OneClassSVMDetector(nu=0.05)
    ocsvm.fit(features)

    # Save the model
    ocsvm.save(MODELS_DIR / "one_class_svm.joblib")

    return ocsvm


if __name__ == "__main__":
    main()
