import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def plot_reconstructions(original_images, reconstructed_images, n=10, save_path=None):
    """Plot original images and their reconstructions."""
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i])
        plt.title("Original")
        plt.axis("off")

        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i])
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_anomaly_scores(
    normal_scores, anomaly_scores=None, threshold=None, save_path=None
):
    """Plot histogram of anomaly scores."""
    plt.figure(figsize=(10, 6))

    plt.hist(normal_scores, bins=50, alpha=0.5, label="Normal")

    if anomaly_scores is not None:
        plt.hist(anomaly_scores, bins=50, alpha=0.5, label="Anomaly")

    if threshold is not None:
        plt.axvline(
            x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold:.4f}"
        )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Distribution of Anomaly Scores")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve for anomaly detection."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return roc_auc


def visualize_anomalies(images, anomaly_scores, threshold, n=10, save_path=None):
    """Visualize images with their anomaly scores."""
    # Sort images by anomaly score
    sorted_indices = np.argsort(anomaly_scores)[::-1]  # Descending order

    plt.figure(figsize=(20, 8))

    for i in range(n):
        idx = sorted_indices[i]
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(images[idx])
        is_anomaly = "ANOMALY" if anomaly_scores[idx] > threshold else "NORMAL"
        plt.title(f"{is_anomaly}\nScore: {anomaly_scores[idx]:.4f}")
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
