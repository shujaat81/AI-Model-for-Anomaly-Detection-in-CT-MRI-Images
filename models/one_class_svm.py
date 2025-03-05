import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *


class OneClassSVMDetector:
    def __init__(self, nu=0.1, kernel="rbf", gamma="scale"):
        """
        Initialize One-Class SVM for anomaly detection.

        Args:
            nu: An upper bound on the fraction of training errors and a lower bound
                on the fraction of support vectors.
            kernel: Kernel type to be used in the algorithm.
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        """
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.scaler = StandardScaler()

    def preprocess_features(self, features):
        """Flatten and scale features."""
        # Flatten features if they're multi-dimensional
        if len(features.shape) > 2:
            flat_features = features.reshape(features.shape[0], -1)
        else:
            flat_features = features

        return flat_features

    def fit(self, features):
        """Fit the One-Class SVM model to the features."""
        flat_features = self.preprocess_features(features)

        # Scale features
        scaled_features = self.scaler.fit_transform(flat_features)

        # Fit the model
        self.model.fit(scaled_features)

        return self

    def predict(self, features):
        """
        Predict anomalies using the One-Class SVM model.

        Returns:
            predictions: 1 for normal samples, -1 for anomalies
            decision_scores: Decision function values (negative values are anomalies)
        """
        flat_features = self.preprocess_features(features)
        scaled_features = self.scaler.transform(flat_features)

        # Get predictions (1: normal, -1: anomaly)
        predictions = self.model.predict(scaled_features)

        # Get decision scores
        decision_scores = self.model.decision_function(scaled_features)

        return predictions, decision_scores

    def save(self, filepath):
        """Save the model to a file."""
        joblib.dump({"model": self.model, "scaler": self.scaler}, filepath)

    def load(self, filepath):
        """Load the model from a file."""
        saved_data = joblib.load(filepath)
        self.model = saved_data["model"]
        self.scaler = saved_data["scaler"]
        return self
