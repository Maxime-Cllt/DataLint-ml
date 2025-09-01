import logging
import os
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from datalint_ml.utils.path import get_project_root
from utils.metrics import calculate_comprehensive_metrics
from utils.model import preprocess_data, load_model_and_tokenizer, run_inference, save_results


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
