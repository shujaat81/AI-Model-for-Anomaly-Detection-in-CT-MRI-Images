import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import sys
from pathlib import Path
import joblib

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))
from config import *
from data.data_loader import load_dataset, load_test_data
from models.autoencoder import compute_reconstruction_error, detect_anomalies, compute_reconstruction_error_in_batches
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

def extract_features_for_evaluation(feature_extractor, images, batch_size=16):
    """Extract features from images in batches for evaluation."""
    total_samples = len(images)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    all_features = []
    print(f"Extracting features from {total_samples} images in {num_batches} batches...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        
        batch_images = images[start_idx:end_idx]
        batch_features = extract_features(feature_extractor, batch_images)
        
        # Apply the same dimensionality reduction as in training
        if len(batch_features.shape) > 2:
            # If features are 4D (batch_size, height, width, channels)
            # Take mean across spatial dimensions to reduce size
            batch_features = np.mean(batch_features, axis=(1, 2))
        
        # Ensure features are 2D (batch_size, features)
        batch_features = batch_features.reshape(batch_features.shape[0], -1)
        
        all_features.append(batch_features)
        
        # Print progress
        if (i+1) % 10 == 0 or (i+1) == num_batches:
            print(f"Processed {i+1}/{num_batches} batches", end="\r")
    
    # Combine all batches
    features = np.vstack(all_features)
    print(f"\nExtracted features shape: {features.shape}")
    
    return features

def evaluate_one_class_svm(ocsvm, feature_extractor, normal_data, anomaly_data):
    """Evaluate One-Class SVM anomaly detection."""
    print("Evaluating One-Class SVM-based anomaly detection...")
    
    # Extract features from normal and anomaly data
    normal_features = extract_features_for_evaluation(feature_extractor, normal_data)
    anomaly_features = extract_features_for_evaluation(feature_extractor, anomaly_data)
    
    # Get predictions and scores
    normal_preds, normal_scores = ocsvm.predict(normal_features)
    anomaly_preds, anomaly_scores = ocsvm.predict(anomaly_features)
    
    # Create labels (1 for normal, -1 for anomaly)
    y_true = np.concatenate([np.ones(len(normal_data)), -np.ones(len(anomaly_data))])
    y_scores = np.concatenate([normal_scores, anomaly_scores])
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true > 0, y_scores)
    pr_auc = average_precision_score(y_true > 0, y_scores)
    
    # Find best F1 score
    precision, recall, thresholds = precision_recall_curve(y_true > 0, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0
    
    # Print results
    print(f"One-Class SVM ROC AUC: {roc_auc:.4f}")
    print(f"One-Class SVM PR AUC: {pr_auc:.4f}")
    print(f"Best F1 Score: {best_f1:.4f} at threshold {best_threshold:.6f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true > 0, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-Class SVM ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / "ocsvm_roc_curve.png")
    plt.close()
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_f1': best_f1,
        'best_threshold': best_threshold
    }

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
    
    # Load trained models
    autoencoder, encoder, ocsvm = load_models()
    
    # Load test data
    test_data = load_test_data()
    
    # For evaluation, we'll use CT images as normal and MRI images as anomalies
    # Use the correct keys from the test_data dictionary
    if 'test_ct' in test_data and 'test_mri' in test_data:
        normal_data = test_data['test_ct']
        anomaly_data = test_data['test_mri']
    else:
        # Try alternative keys if available
        available_keys = list(test_data.keys())
        print(f"Available keys in test_data: {available_keys}")
        
        if len(available_keys) >= 2:
            normal_data = test_data[available_keys[0]]
            anomaly_data = test_data[available_keys[1]]
        else:
            raise ValueError("Could not find appropriate test data for evaluation")
    
    print(f"Loaded {len(normal_data)} normal images and {len(anomaly_data)} anomaly images for evaluation.")
    
    # Evaluate autoencoder
    if autoencoder is not None:
        autoencoder_metrics = evaluate_autoencoder(autoencoder, normal_data, anomaly_data)
    else:
        autoencoder_metrics = None
        print("Skipping autoencoder evaluation.")
    
    # Build feature extractor for One-Class SVM
    feature_extractor = None
    if ocsvm is not None:
        feature_extractor = build_resnet_feature_extractor(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )
        
        # Evaluate One-Class SVM
        ocsvm_metrics = evaluate_one_class_svm(ocsvm, feature_extractor, normal_data, anomaly_data)
    else:
        ocsvm_metrics = None
        print("Skipping One-Class SVM evaluation.")
    
    # Compare methods if both were evaluated
    if autoencoder_metrics is not None and ocsvm_metrics is not None:
        compare_methods(autoencoder_metrics, ocsvm_metrics)
    
    # Optional: Evaluate on some unseen demo images
    try:
        print("Evaluating on unseen demo images...")
        # Check if demo directory exists
        demo_dir = DATA_DIR / "demo"
        if os.path.exists(demo_dir):
            # Load a few demo images
            demo_images = []
            for img_path in os.listdir(demo_dir)[:10]:  # Limit to 10 images
                if img_path.endswith(('.jpg', '.png', '.jpeg')):
                    img = tf.keras.preprocessing.image.load_img(
                        os.path.join(demo_dir, img_path),
                        target_size=IMAGE_SIZE
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    demo_images.append(img_array)
            
            if demo_images:
                demo_images = np.array(demo_images)
                
                # Detect anomalies using autoencoder
                if autoencoder is not None and autoencoder_metrics is not None:
                    # Use batch processing to avoid the progress bar issue
                    unseen_scores = compute_reconstruction_error_in_batches(autoencoder, demo_images)
                    is_anomaly = unseen_scores > autoencoder_metrics['best_threshold']
                    
                    # Visualize results
                    plt.figure(figsize=(15, 10))
                    for i in range(min(len(demo_images), 10)):
                        plt.subplot(2, 5, i+1)
                        plt.imshow(demo_images[i])
                        status = "Anomaly" if is_anomaly[i] else "Normal"
                        plt.title(f"{status} (Score: {unseen_scores[i]:.4f})")
                        plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(RESULTS_DIR / "demo_predictions.png")
                    plt.close()
                    
                    print(f"Evaluated {len(demo_images)} demo images. Results saved to {RESULTS_DIR / 'demo_predictions.png'}")
            else:
                print("No demo images found.")
        else:
            print("Demo directory not found.")
    except Exception as e:
        print(f"Error evaluating demo images: {str(e)}")
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 