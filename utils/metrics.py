import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def calculate_anomaly_metrics(y_true, anomaly_scores):
    """
    Calculate metrics for anomaly detection.
    
    Args:
        y_true: Ground truth labels (1 for anomaly, 0 for normal)
        anomaly_scores: Anomaly scores from the model
        
    Returns:
        Dictionary of metrics
    """
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    
    # Calculate PR AUC (Average Precision)
    pr_auc = average_precision_score(y_true, anomaly_scores)
    
    # Calculate precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    
    # Calculate F1 scores for different thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find the threshold that maximizes F1 score
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0
    best_f1 = f1_scores[best_threshold_idx]
    best_precision = precision[best_threshold_idx]
    best_recall = recall[best_threshold_idx]
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall
    }

def find_optimal_threshold(normal_scores, contamination=0.01):
    """
    Find an optimal threshold for anomaly detection based on normal data.
    
    Args:
        normal_scores: Anomaly scores from normal data
        contamination: Expected proportion of anomalies
        
    Returns:
        Threshold value
    """
    # Sort scores in descending order
    sorted_scores = np.sort(normal_scores)[::-1]
    
    # Calculate the index corresponding to the contamination level
    threshold_idx = int(contamination * len(sorted_scores))
    
    # Get the threshold
    threshold = sorted_scores[threshold_idx]
    
    return threshold 