import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def build_resnet_feature_extractor(input_shape=(128, 128, 3)):
    """Build a ResNet50-based feature extractor."""
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Get the output of a specific layer
    feature_extractor = Model(
        inputs=base_model.input, outputs=base_model.get_layer("conv4_block6_out").output
    )

    return feature_extractor


def build_vgg_feature_extractor(input_shape=(128, 128, 3)):
    """Build a VGG16-based feature extractor."""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Get the output of a specific layer
    feature_extractor = Model(
        inputs=base_model.input, outputs=base_model.get_layer("block4_conv3").output
    )

    return feature_extractor


def extract_features(feature_extractor, images):
    """Extract features from images using the feature extractor."""
    # Check the model type by name instead of using isinstance
    model_name = feature_extractor.name

    # Preprocess images based on model type
    if "resnet" in model_name.lower():
        preprocessed_images = tf.keras.applications.resnet50.preprocess_input(
            images * 255.0
        )
    elif "vgg" in model_name.lower():
        preprocessed_images = tf.keras.applications.vgg16.preprocess_input(
            images * 255.0
        )
    else:
        # Default preprocessing
        preprocessed_images = images

    # Extract features
    features = feature_extractor.predict(preprocessed_images, verbose=0)

    return features
