import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sys
from pathlib import Path
import joblib

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))
from config import *
from data.data_loader import load_dataset, load_test_data
from models.autoencoder import compute_reconstruction_error, detect_anomalies
from models.feature_extractor import build_resnet_feature_extractor, extract_features
from models.one_class_svm import OneClassSVMDetector
from utils.visualization import plot_reconstructions, plot_anomaly_scores, plot_roc_curve, visualize_anomalies
from utils.metrics import calculate_anomaly_metrics, find_optimal_threshold

def load_models():
    """Load trained models for evaluation."""
    print("Loading trained models...")
    
    # Load autoencoder
    autoencoder_path = MODELS_DIR / "autoencoder_final.h5"
    if os.path.exists(autoencoder_path):
        autoencoder = tf.keras.models.load_model(autoencoder_path)
        print("Loaded autoencoder model.")
    else:
        autoencoder = None
        print("Autoencoder model not found.")
    
    # Load encoder
    encoder_path = MODELS_DIR / "encoder_final.h5"
    if os.path.exists(encoder_path):
        encoder = tf.keras.models.load_model(encoder_path)
        print("Loaded encoder model.")
    else:
        encoder = None
        print("Encoder model not found.")
    
    # Load One-Class SVM
    ocsvm_path = MODELS_DIR / "one_class_svm.joblib"
    if os.path.exists(ocsvm_path):
        ocsvm = OneClassSVMDetector()
        ocsvm.load(ocsvm_path)
        print("Loaded One-Class SVM model.")
    else:
        ocsvm = None
        print("One-Class SVM model not found.")
    
    return autoencoder, encoder, ocsvm

def evaluate_autoencoder(autoencoder, normal_data, anomaly_data=None):
    """Evaluate autoencoder-based anomaly detection."""
    print("Evaluating autoencoder-based anomaly detection...")
    
    # Compute reconstruction errors for normal data
    normal_recon_errors = compute_reconstruction_error(autoencoder, normal_data)
    
    # Find optimal threshold
    threshold = find_optimal_threshold(normal_recon_errors, contamination=0.01)
    print(f"Optimal threshold for anomaly detection: {threshold:.6f}")
    
    # If anomaly data is provided, evaluate performance
    if anomaly_data is not None:
        # Compute reconstruction errors for anomaly data
        anomaly_recon_errors = compute_reconstruction_error(autoencoder, anomaly_data)
        
        # Plot anomaly score distribution
        plot_anomaly_scores(
            normal_recon_errors, 
            anomaly_recon_errors,
            threshold=threshold,
            save_path=RESULTS_DIR / "autoencoder_anomaly_scores.png"
        )
        
        # Create labels (0 for normal, 1 for anomaly)
        y_true = np.concatenate([np.zeros(len(normal_recon_errors)), np.ones(len(anomaly_recon_errors))])
        y_scores = np.concatenate([normal_recon_errors, anomaly_recon_errors])
        
        # Calculate metrics
        metrics = calculate_anomaly_metrics(y_true, y_scores)
        
        # Plot ROC curve
        roc_auc = plot_roc_curve(
            y_true, 
            y_scores, 
            save_path=RESULTS_DIR / "autoencoder_roc_curve.png"
        )
        
        print(f"Autoencoder ROC AUC: {roc_auc:.4f}")
        print(f"Autoencoder PR AUC: {metrics['pr_auc']:.4f}")
        print(f"Best F1 Score: {metrics['best_f1']:.4f} at threshold {metrics['best_threshold']:.6f}")
        
        # Visualize some anomalies
        all_images = np.concatenate([normal_data, anomaly_data])
        visualize_anomalies(
            all_images, 
            y_scores, 
            threshold=metrics['best_threshold'],
            n=10,
            save_path=RESULTS_DIR / "autoencoder_detected_anomalies.png"
        )
        
        return metrics
    
    return None

def evaluate_one_class_svm(ocsvm, feature_extractor, normal_data, anomaly_data=None):
    """Evaluate One-Class SVM-based anomaly detection."""
    print("Evaluating One-Class SVM-based anomaly detection...")
    
    # Extract features from normal data
    normal_features = extract_features(feature_extractor, normal_data)
    
    # Get predictions and decision scores for normal data
    normal_preds, normal_scores = ocsvm.predict(normal_features)
    
    # If anomaly data is provided, evaluate performance
    if anomaly_data is not None:
        # Extract features from anomaly data
        anomaly_features = extract_features(feature_extractor, anomaly_data)
        
        # Get predictions and decision scores for anomaly data
        anomaly_preds, anomaly_scores = ocsvm.predict(anomaly_features)
        
        # Plot anomaly score distribution
        # Note: For One-Class SVM, lower scores indicate anomalies, so we negate them
        plot_anomaly_scores(
            -normal_scores, 
            -anomaly_scores,
            save_path=RESULTS_DIR / "ocsvm_anomaly_scores.png"
        )
        
        # Create labels (0 for normal, 1 for anomaly)
        y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
        
        # For metrics, we negate the scores because higher values should indicate anomalies
        y_scores = np.concatenate([-normal_scores, -anomaly_scores])
        
        # Calculate metrics
        metrics = calculate_anomaly_metrics(y_true, y_scores)
        
        # Plot ROC curve
        roc_auc = plot_roc_curve(
            y_true, 
            y_scores, 
            save_path=RESULTS_DIR / "ocsvm_roc_curve.png"
        )
        
        print(f"One-Class SVM ROC AUC: {roc_auc:.4f}")
        print(f"One-Class SVM PR AUC: {metrics['pr_auc']:.4f}")
        print(f"Best F1 Score: {metrics['best_f1']:.4f} at threshold {metrics['best_threshold']:.6f}")
        
        # Visualize some anomalies
        all_images = np.concatenate([normal_data, anomaly_data])
        visualize_anomalies(
            all_images, 
            y_scores, 
            threshold=metrics['best_threshold'],
            n=10,
            save_path=RESULTS_DIR / "ocsvm_detected_anomalies.png"
        )
        
        return metrics
    
    return None

def compare_methods(autoencoder_metrics, ocsvm_metrics):
    """Compare different anomaly detection methods."""
    print("Comparing anomaly detection methods...")
    
    # Create a bar chart to compare ROC AUC scores
    methods = ['Autoencoder', 'One-Class SVM']
    roc_auc_scores = [autoencoder_metrics['roc_auc'], ocsvm_metrics['roc_auc']]
    pr_auc_scores = [autoencoder_metrics['pr_auc'], ocsvm_metrics['pr_auc']]
    f1_scores = [autoencoder_metrics['best_f1'], ocsvm_metrics['best_f1']]
    
    x = np.arange(len(methods))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, roc_auc_scores, width, label='ROC AUC')
    plt.bar(x, pr_auc_scores, width, label='PR AUC')
    plt.bar(x + width, f1_scores, width, label='Best F1')
    
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Comparison of Anomaly Detection Methods')
    plt.xticks(x, methods)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "method_comparison.png")
    plt.close()
    
    # Print comparison summary
    print("\nMethod Comparison Summary:")
    print("-" * 50)
    print(f"{'Method':<15} {'ROC AUC':<10} {'PR AUC':<10} {'Best F1':<10}")
    print("-" * 50)
    print(f"{'Autoencoder':<15} {autoencoder_metrics['roc_auc']:<10.4f} {autoencoder_metrics['pr_auc']:<10.4f} {autoencoder_metrics['best_f1']:<10.4f}")
    print(f"{'One-Class SVM':<15} {ocsvm_metrics['roc_auc']:<10.4f} {ocsvm_metrics['pr_auc']:<10.4f} {ocsvm_metrics['best_f1']:<10.4f}")
    print("-" * 50)
    
    # Determine the best method
    if autoencoder_metrics['roc_auc'] > ocsvm_metrics['roc_auc']:
        print("Based on ROC AUC, Autoencoder performs better.")
    else:
        print("Based on ROC AUC, One-Class SVM performs better.")
    
    if autoencoder_metrics['pr_auc'] > ocsvm_metrics['pr_auc']:
        print("Based on PR AUC, Autoencoder performs better.")
    else:
        print("Based on PR AUC, One-Class SVM performs better.")
    
    if autoencoder_metrics['best_f1'] > ocsvm_metrics['best_f1']:
        print("Based on F1 Score, Autoencoder performs better.")
    else:
        print("Based on F1 Score, One-Class SVM performs better.")

def main():
    """Main evaluation function."""
    print("Starting evaluation process...")
    
    # Load models
    autoencoder, encoder, ocsvm = load_models()
    
    # Load test data
    test_data = load_test_data()
    
    # For evaluation, we'll use CT images as normal and MRI images as anomalies
    normal_data = test_data['test_ct']
    anomaly_data = test_data['test_mri']
    
    print(f"Loaded {len(normal_data)} normal images and {len(anomaly_data)} anomaly images for evaluation.")
    
    # Initialize metrics
    autoencoder_metrics = None
    ocsvm_metrics = None
    
    # Evaluate autoencoder if available
    if autoencoder is not None:
        autoencoder_metrics = evaluate_autoencoder(autoencoder, normal_data, anomaly_data)
    
    # Evaluate One-Class SVM if available
    if ocsvm is not None:
        # Build feature extractor
        feature_extractor = build_resnet_feature_extractor(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )
        ocsvm_metrics = evaluate_one_class_svm(ocsvm, feature_extractor, normal_data, anomaly_data)
    
    # Compare methods if both are available
    if autoencoder_metrics is not None and ocsvm_metrics is not None:
        compare_methods(autoencoder_metrics, ocsvm_metrics)
    
    # If unseen demo images are available, evaluate on them
    if test_data['unseen'] is not None:
        print("\nEvaluating on unseen demo images...")
        unseen_images = test_data['unseen']
        
        if autoencoder is not None:
            # Detect anomalies using autoencoder
            unseen_scores, is_anomaly, threshold = detect_anomalies(
                autoencoder, 
                unseen_images, 
                reference_errors=compute_reconstruction_error(autoencoder, normal_data)
            )
            
            # Visualize results
            visualize_anomalies(
                unseen_images, 
                unseen_scores, 
                threshold=threshold,
                n=min(10, len(unseen_images)),
                save_path=RESULTS_DIR / "autoencoder_unseen_anomalies.png"
            )
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 