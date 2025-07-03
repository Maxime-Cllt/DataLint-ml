import logging
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.path import get_project_root


def setup_logging(model_version: str) -> logging.Logger:
    """
    Setup logging configuration for the evaluation script.
    :param model_version:
    :return:
    """
    log_dir = os.path.join(get_project_root(), "logs", "evaluation")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_{model_version}_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_data_from_database(engine, table_name: str, limit: int = 10000) -> pd.DataFrame:
    """
    Load data from the specified database table with comprehensive logging.
    :param engine:
    :param table_name:
    :param limit:
    :return:
    """
    logger = logging.getLogger(__name__)

    sql_query = f"""
        SELECT label, CAST(note as INTEGER) as note
        FROM {table_name}
        WHERE label IS NOT NULL AND note IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit};
    """

    try:
        start_time = time.time()
        with engine.connect() as connection:
            dataset = pd.read_sql(text(sql_query), connection)
        load_time = time.time() - start_time

        logger.info(f"Data loaded successfully in {load_time:.2f} seconds")

        # Log target distribution (note is the target: 0=safe, 1=unsafe)
        note_distribution = dataset['note'].value_counts().sort_index()
        logger.info(f"Target (note) distribution:")
        for note_value, count in note_distribution.items():
            safety_label = "SAFE" if note_value == 0 else "UNSAFE"
            logger.info(f"  - Note {note_value} ({safety_label}): {count} samples ({count / len(dataset) * 100:.1f}%)")

        # Check if we have balanced classes
        if len(note_distribution) < 2:
            logger.warning("Only one class found in target variable 'note'!")
            logger.warning("This will result in perfect accuracy but meaningless evaluation.")

        return dataset

    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        raise


def preprocess_data(dataset: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Preprocess the dataset to extract input texts and target labels with comprehensive logging.
    :param dataset:
    :return:
    """
    logger = logging.getLogger(__name__)

    initial_size = len(dataset)
    dataset = dataset.dropna(subset=['label', 'note'])
    cleaned_size = len(dataset)

    if cleaned_size < initial_size:
        logger.warning(f"Removed {initial_size - cleaned_size} rows with null values")

    input_texts = dataset['label'].tolist()
    target_labels = dataset['note'].tolist()

    input_texts = [str(text) if text is not None else "" for text in input_texts]

    target_labels = [int(label) for label in target_labels]

    unique_targets = set(target_labels)

    if len(unique_targets) != 2 or not all(t in [0, 1] for t in unique_targets):
        logger.warning(f"Expected binary targets [0, 1], but found: {sorted(unique_targets)}")

    # Text preprocessing
    original_lengths = [len(str(text)) for text in input_texts]
    input_texts = [str(text)[:512] if text else "" for text in input_texts]
    truncated_count = sum(1 for orig, trunc in zip(original_lengths, [len(t) for t in input_texts]) if orig > trunc)

    logger.info(f"Text preprocessing completed:")
    logger.info(f"  - Average original length: {np.mean(original_lengths):.2f}")
    logger.info(f"  - Texts truncated: {truncated_count}")
    logger.info(f"  - Average final length: {np.mean([len(t) for t in input_texts]):.2f}")
    logger.info(f"  - Final dataset size: {len(input_texts)}")

    return input_texts, target_labels


def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any, torch.device]:
    """
    Load the model and tokenizer from the specified path with comprehensive logging.
    :param model_path:
    :return:
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading model from: {model_path}")

        model_dir = os.path.join(model_path, "tiny_model")
        tokenizer_dir = os.path.join(model_path, "tokenizer")

        if not os.path.exists(model_dir):
            logger.error(f"Model directory not found: {model_dir}")
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if not os.path.exists(tokenizer_dir):
            logger.error(f"Tokenizer directory not found: {tokenizer_dir}")
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        model.to(device)
        model.eval()

        logger.info("Model and tokenizer loaded successfully")
        logger.info(f"Model config: {model.config}")
        logger.info(f"Number of labels in model: {model.config.num_labels}")

        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def run_inference(model, tokenizer, device, texts: List[str], threshold: float = 0.5) -> Tuple[
    List[int], List[float], List[List[float]], float]:
    """
    Run inference on the provided texts using the specified model and tokenizer.
    :param model:
    :param tokenizer:
    :param device:
    :param texts:
    :param threshold:
    :return:
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Starting inference on {len(texts)} samples...")
    logger.info(f"Using threshold {threshold} for binary classification")

    predictions = []  # List to store binary predictions (0=SAFE, 1=UNSAFE)
    probabilities = []  # List to store unsafe probabilities
    all_probabilities = []  # List to store all class probabilities (if applicable)
    inference_times = []

    start_time = time.time()

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processing sample {i}/{len(texts)}")

            sample_start = time.time()

            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                outputs = model(**inputs)
                logits = outputs.logits

                if model.config.num_labels == 1:
                    # Single output with sigmoid (binary classification)
                    prob_unsafe = torch.sigmoid(logits).item()
                    prob_safe = 1.0 - prob_unsafe
                    all_probs = [prob_safe, prob_unsafe]
                elif model.config.num_labels == 2:
                    # Two outputs with softmax
                    probs = torch.softmax(logits, dim=-1)
                    all_probs = probs[0].cpu().numpy().tolist()
                    prob_unsafe = all_probs[1]  # Probability of class 1 (unsafe)
                else:
                    logger.error(f"Unexpected number of labels: {model.config.num_labels}")
                    raise ValueError(f"Model has {model.config.num_labels} labels, expected 1 or 2")

                probabilities.append(prob_unsafe)
                all_probabilities.append(all_probs)

                predicted_label = 1 if prob_unsafe >= threshold else 0
                predictions.append(predicted_label)

                inference_times.append(time.time() - sample_start)

            except Exception as e:
                logger.error(f"Error processing sample {i}: {str(e)}")
                predictions.append(0)
                probabilities.append(0.5)
                all_probabilities.append([0.5, 0.5])
                inference_times.append(0)

    total_inference_time = time.time() - start_time
    avg_inference_time = np.mean(inference_times)

    logger.info(f"Inference completed:")
    logger.info(f"  - Total time: {total_inference_time:.2f} seconds")
    logger.info(f"  - Average time per sample: {avg_inference_time * 1000:.2f} ms")
    logger.info(f"  - Samples per second: {len(texts) / total_inference_time:.2f}")

    pred_distribution = pd.Series(predictions).value_counts().sort_index()
    logger.info("Prediction distribution:")
    for pred, count in pred_distribution.items():
        safety_label = "SAFE" if pred == 0 else "UNSAFE"
        logger.info(f"  - Predicted {pred} ({safety_label}): {count} samples ({count / len(predictions) * 100:.1f}%)")

    # Log probability statistics
    logger.info("Probability statistics:")
    logger.info(f"  - Mean unsafe probability: {np.mean(probabilities):.4f}")
    logger.info(f"  - Std unsafe probability: {np.std(probabilities):.4f}")
    logger.info(f"  - Min unsafe probability: {np.min(probabilities):.4f}")
    logger.info(f"  - Max unsafe probability: {np.max(probabilities):.4f}")

    return predictions, probabilities, all_probabilities, total_inference_time


def calculate_comprehensive_metrics(true_labels: List[int], predictions: List[int],
                                    probabilities: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for binary classification with detailed logging.
    :param true_labels:
    :param predictions:
    :param probabilities:
    :param threshold:
    :return:
    """
    logger = logging.getLogger(__name__)

    logger.info("Calculating evaluation metrics...")

    metrics = {}
    metrics['threshold'] = threshold

    # Check class distribution
    unique_true = set(true_labels)
    unique_pred = set(predictions)

    true_dist = pd.Series(true_labels).value_counts().sort_index()
    pred_dist = pd.Series(predictions).value_counts().sort_index()

    logger.info("Class distributions:")
    logger.info(f"  True labels: {true_dist.to_dict()}")
    logger.info(f"  Predictions: {pred_dist.to_dict()}")

    # Basic metrics
    metrics['accuracy'] = accuracy_score(true_labels, predictions)

    # Handle metrics carefully for potential single-class scenarios
    try:
        metrics['precision'] = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(true_labels, predictions, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(true_labels, predictions, average='micro', zero_division=0)

        # Binary-specific metrics
        metrics['precision_unsafe'] = precision_score(true_labels, predictions, pos_label=1, zero_division=0)
        metrics['recall_unsafe'] = recall_score(true_labels, predictions, pos_label=1, zero_division=0)
        metrics['f1_unsafe'] = f1_score(true_labels, predictions, pos_label=1, zero_division=0)

    except Exception as e:
        logger.warning(f"Error calculating precision/recall/f1: {e}")
        metrics.update({'precision': 0.0, 'recall': 0.0, 'f1_weighted': 0.0,
                        'f1_macro': 0.0, 'f1_micro': 0.0, 'precision_unsafe': 0.0,
                        'recall_unsafe': 0.0, 'f1_unsafe': 0.0})

    # ROC AUC for binary classification
    try:
        if len(unique_true) >= 2 and len(set(true_labels + predictions)) >= 2:
            metrics['roc_auc'] = roc_auc_score(true_labels, probabilities)
        else:
            logger.warning("Cannot calculate ROC AUC: insufficient class diversity")
            metrics['roc_auc'] = None
    except ValueError as e:
        logger.warning(f"Could not calculate ROC AUC: {e}")
        metrics['roc_auc'] = None

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    metrics['confusion_matrix'] = cm

    # Extract confusion matrix components for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Calculate additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Classification report
    try:
        metrics['classification_report'] = classification_report(
            true_labels, predictions, labels=[0, 1],
            target_names=['Safe', 'Unsafe'], output_dict=True, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Could not generate classification report: {e}")
        metrics['classification_report'] = {}

    # Log key metrics
    logger.info("=== BINARY CLASSIFICATION RESULTS ===")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision (weighted): {metrics['precision']:.4f}")
    logger.info(f"Recall (weighted): {metrics['recall']:.4f}")
    logger.info(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"F1-score (unsafe class): {metrics['f1_unsafe']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

    if 'specificity' in metrics:
        logger.info(f"Specificity (Safe detection): {metrics['specificity']:.4f}")
        logger.info(f"Sensitivity (Unsafe detection): {metrics['sensitivity']:.4f}")

    # Detailed analysis
    logger.info("=== DETAILED ANALYSIS ===")
    correct_predictions = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    logger.info(f"Correct predictions: {correct_predictions}/{len(true_labels)}")

    if len(unique_true) == 1:
        logger.warning(f"WARNING: All true labels are the same ({list(unique_true)[0]})")
        logger.warning("This suggests a data quality issue or model overfitting.")

    return metrics


def save_results(metrics: Dict[str, Any], model_version: str,
                 inference_time: float, dataset_size: int, probabilities: List[float],
                 true_labels: List[int]):
    """
    Save evaluation results and create visualizations with comprehensive logging.
    :param metrics:
    :param model_version:
    :param inference_time:
    :param dataset_size:
    :param probabilities:
    :param true_labels:
    :return:
    """
    logger = logging.getLogger(__name__)

    results_dir = os.path.join(get_root_path(), "results", "evaluation", model_version)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics as JSON
    import json
    metrics_to_save = {k: v for k, v in metrics.items()
                       if k not in ['confusion_matrix']}
    metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    metrics_to_save['inference_time'] = inference_time
    metrics_to_save['dataset_size'] = dataset_size
    metrics_to_save['timestamp'] = timestamp

    metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    logger.info(f"Metrics saved to: {metrics_file}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Unsafe'], yticklabels=['Safe', 'Unsafe'],
                ax=axes[0, 0])
    axes[0, 0].set_title(f'Confusion Matrix - {model_version}')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 2. Probability Distribution
    axes[0, 1].hist(probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=metrics['threshold'], color='red', linestyle='--',
                       label=f'Threshold ({metrics["threshold"]})')
    axes[0, 1].set_title('Distribution of Unsafe Probabilities')
    axes[0, 1].set_xlabel('Probability of Unsafe')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # 3. ROC Curve (if possible)
    if metrics.get('roc_auc') is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc="lower right")
    else:
        axes[1, 0].text(0.5, 0.5, 'ROC Curve\nNot Available',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('ROC Curve')

    # 4. Precision-Recall Curve
    try:
        precision, recall, _ = precision_recall_curve(true_labels, probabilities)
        axes[1, 1].plot(recall, precision, color='purple', lw=2)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, 'Precision-Recall Curve\nNot Available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plots_file = os.path.join(results_dir, f"evaluation_plots_{timestamp}.png")
    plt.savefig(plots_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Evaluation plots saved to: {plots_file}")


def main():
    """
    Main function to run the evaluation script with comprehensive logging and error handling.
    :return:
    """
    MODEL_VERSION: str = "v1.0"
    CLASSIFICATION_THRESHOLD: float = 0.5  # Threshold for binary classification

    # Setup logging
    logger = setup_logging(MODEL_VERSION)
    logger.info(f"Starting evaluation for model version: {MODEL_VERSION}")
    logger.info(f"Binary classification threshold: {CLASSIFICATION_THRESHOLD}")

    try:
        # Paths and database setup
        PATH = os.path.join(get_project_root(), "script", "v2", "tiny", MODEL_VERSION)
        sqlite_path: str = r'database.db'
        engine = create_engine(f'sqlite:///{sqlite_path}')
        table_name = 'test_model'

        logger.info(f"Model path: {PATH}")
        logger.info(f"Database path: {sqlite_path}")

        # Load data from database
        dataset = load_data_from_database(engine, table_name, limit=5000)

        # Preprocess data (label -> input text, note -> target)
        input_texts, target_labels = preprocess_data(dataset)

        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(PATH)

        # Run inference with proper sigmoid handling
        predictions, probabilities, all_probabilities, inference_time = run_inference(
            model, tokenizer, device, input_texts, threshold=CLASSIFICATION_THRESHOLD
        )

        # Calculate metrics
        metrics = calculate_comprehensive_metrics(target_labels, predictions, probabilities,
                                                  threshold=CLASSIFICATION_THRESHOLD)

        # Save results
        save_results(metrics, MODEL_VERSION, inference_time, len(input_texts),
                     probabilities, target_labels)

        logger.info("=== EVALUATION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Model: {MODEL_VERSION}")
        logger.info(f"Dataset size: {len(input_texts)}")
        logger.info(f"Input: label (text)")
        logger.info(f"Target: note (0=Safe, 1=Unsafe)")
        logger.info(f"Classification threshold: {CLASSIFICATION_THRESHOLD}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (weighted): {metrics['precision']:.4f}")
        logger.info(f"Recall (weighted): {metrics['recall']:.4f}")
        logger.info(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"F1-score (unsafe): {metrics['f1_unsafe']:.4f}")
        if metrics['roc_auc'] is not None:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

        if 'specificity' in metrics:
            logger.info(f"Specificity (Safe detection): {metrics['specificity']:.4f}")
            logger.info(f"Sensitivity (Unsafe detection): {metrics['sensitivity']:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
