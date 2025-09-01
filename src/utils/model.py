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
    precision_recall_curve
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.path import get_project_root


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

    results_dir = os.path.join(get_project_root(), "results", "evaluation", model_version)
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
