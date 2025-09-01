import logging
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    accuracy_score, precision_score, recall_score,
    roc_auc_score
)


def plot_attention(model, tokenizer, inputs):
    """
    Visualize attention weights of the model using a heatmap.
    :param model:
    :param tokenizer:
    :param inputs:
    :return:
    """
    input_ids = inputs['input_ids'][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attention = outputs.attentions[0][0, 0].cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title('Attention Weights')
    plt.xlabel('Attended To')
    plt.ylabel('Attending From')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('attention_visualization_labeled.png')
    plt.close()
    print("Attention visualization saved as attention_visualization_labeled.png")


def print_model_details(model):
    """
    Print details about the model architecture and parameters.
    :param model:
    :return:
    """
    print("\nModel Architecture Details:")
    print(f"Model architecture: {model.config.architectures}")
    print(f"Number of hidden layers: {model.config.num_hidden_layers}")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def print_tokenizer_details(tokenizer):
    """
    Print details about the tokenizer.
    :param tokenizer:
    :return:
    """
    print("\nTokenizer Details:")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")


def print_parameter_sizes(model):
    """
    Print the sizes of each parameter in the model.
    :param model:
    :return:
    """
    print("\nModel Parameter Sizes:")
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Size: {param.size()}, Parameters: {param.numel()}")


def visualize_embeddings_tsne_filtered(model, tokenizer, num_tokens=200):
    """
    Visualize token embeddings using t-SNE, filtering out unused and non-displayable tokens.
    :param model:
    :param tokenizer:
    :param num_tokens:
    :return:
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print("\nVisualizing Filtered Embeddings in 2D with t-SNE...")

    def is_displayable(token):
        try:
            token.encode('latin1')
            return True
        except UnicodeEncodeError:
            return False

    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

    valid_ids = []
    tokens = []
    for idx in range(len(inv_vocab)):
        tok = inv_vocab[idx]
        if not tok.startswith('[unused') and not tok.startswith('[') and is_displayable(tok):
            valid_ids.append(idx)
            tokens.append(tok)
        if len(valid_ids) >= num_tokens:
            break

    embedding_weights = model.get_input_embeddings().weight.detach().cpu().numpy()
    embedding_subset = embedding_weights[valid_ids]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    embeddings_2d = tsne.fit_transform(embedding_subset)

    plt.figure(figsize=(16, 12))
    for i, label in enumerate(tokens):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), fontsize=9)
    plt.title("t-SNE Visualization of Token Embeddings (Filtered)")
    plt.tight_layout()
    plt.savefig("tsne_embeddings_filtered.png")
    plt.close()
    print("Filtered t-SNE embedding visualization saved as tsne_embeddings_filtered.png")


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
