import os
from pathlib import Path

# Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "saved_models"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Model parameters
LATENT_DIM = 128
LEARNING_RATE = 1e-4
EPOCHS = 2

# Anomaly detection parameters
ANOMALY_THRESHOLD = 0.5  # This will be tuned based on validation data
