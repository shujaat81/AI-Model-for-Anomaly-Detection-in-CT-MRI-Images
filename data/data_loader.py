import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from data.preprocessing import (
    load_images_from_folder,
    normalize_images,
    create_data_generators,
    visualize_samples,
)


def load_dataset():
    """Load and prepare the CT-MRI dataset."""
    print("Loading dataset...")

    # Load CT images (trainA)
    ct_train_path = DATA_DIR / "trainA"
    ct_images = load_images_from_folder(ct_train_path)

    # Load MRI images (trainB) - we'll use these as a separate modality
    mri_train_path = DATA_DIR / "trainB"
    mri_images = load_images_from_folder(mri_train_path)

    # Normalize images
    ct_images_normalized = normalize_images(ct_images)
    mri_images_normalized = normalize_images(mri_images)

    # Visualize samples
    visualize_samples(ct_images)
    visualize_samples(mri_images)

    print(f"Loaded {len(ct_images)} CT images and {len(mri_images)} MRI images")

    # Create data generators for CT images (we'll focus on one modality for anomaly detection)
    train_gen, val_gen, train_images, val_images = create_data_generators(
        ct_images_normalized
    )

    return {
        "ct": {
            "train_generator": train_gen,
            "val_generator": val_gen,
            "train_images": train_images,
            "val_images": val_images,
            "all_images": ct_images_normalized,
        },
        "mri": {"all_images": mri_images_normalized},
    }


def load_test_data():
    """Load test data for evaluation."""
    # Load test CT images
    test_ct_path = DATA_DIR / "testA"
    test_ct_images = load_images_from_folder(test_ct_path)
    test_ct_normalized = normalize_images(test_ct_images)

    # Load test MRI images
    test_mri_path = DATA_DIR / "testB"
    test_mri_images = load_images_from_folder(test_mri_path)
    test_mri_normalized = normalize_images(test_mri_images)

    # Load unseen demo images (if available)
    unseen_path = DATA_DIR / "unseen_demo_images"
    if os.path.exists(unseen_path):
        unseen_images = load_images_from_folder(unseen_path)
        unseen_normalized = normalize_images(unseen_images)
    else:
        unseen_normalized = None

    return {
        "test_ct": test_ct_normalized,
        "test_mri": test_mri_normalized,
        "unseen": unseen_normalized,
    }
