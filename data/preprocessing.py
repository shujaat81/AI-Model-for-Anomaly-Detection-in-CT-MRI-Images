import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def load_images_from_folder(folder_path):
    """Load all images from a folder into a numpy array."""
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
    
    return np.array(images)

def normalize_images(images):
    """Normalize images to [0, 1] range."""
    return images.astype('float32') / 255.0

def create_data_generators(images, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
    """Create data generators with augmentation for training and validation."""
    # Split data into training and validation
    train_images, val_images = train_test_split(
        images, test_size=validation_split, random_state=RANDOM_SEED
    )
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        train_images, train_images,  # Input = Output for autoencoder
        batch_size=batch_size
    )
    
    val_generator = val_datagen.flow(
        val_images, val_images,  # Input = Output for autoencoder
        batch_size=batch_size
    )
    
    return train_generator, val_generator, train_images, val_images

def visualize_samples(images, n=5):
    """Visualize sample images."""
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_images.png")
    plt.close() 