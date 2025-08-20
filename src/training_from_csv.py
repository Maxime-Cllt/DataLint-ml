import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from src.path import get_project_root

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 score for the evaluation predictions.
    :param eval_pred:
    :return:
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def tokenize_function(batch):
    """
    Tokenizes the input batch of text data using the provided tokenizer.
    :param batch:
    :return:
    """
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length)


if __name__ == "__main__":
    starting_time = time.time()
    MODEL_NAME: str = "prajjwal1/bert-tiny"
    VERSION = "v1.0"
    PATH_TO_SAVE: str = os.path.join(get_project_root(), "models", VERSION)
    set_seed(42)

    df = pd.read_csv("word_detection.csv", sep=",")
    df = df.drop(columns=["id"], axis=1)
    df = df.rename(columns={"label": "text", "note": "label"})

    df["text"] = df["text"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"],
                                                        random_state=42)

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        dataloader_num_workers=12,
        weight_decay=0.001,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=42
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Eval results -> Accuracy: {eval_results['eval_accuracy']:.2f} | F1: {eval_results['eval_f1']:.2f}")

    # Inference example
    example = "Test example for prediction."
    inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True,
                       max_length=tokenizer.model_max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        print(f"Prediction for '{example}': {predictions.item()}")

    os.makedirs(PATH_TO_SAVE, exist_ok=True)

    try:
        model.save_pretrained(PATH_TO_SAVE)
        print("Model saved with save_pretrained.")
    except Exception as e:
        print(f"Error saving model: {e}")

    try:
        tokenizer.save_pretrained(os.path.join(PATH_TO_SAVE, "tokenizer"))
        print("Tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")

    try:
        trainer.save_model(os.path.join(PATH_TO_SAVE, "trainer"))
        print("Trainer model saved successfully.")
    except Exception as e:
        print(f"Error saving trainer model: {e}")

    try:
        torch.save(model, os.path.join(PATH_TO_SAVE, "model", "complete_model.pt"))
        print("Complete model saved successfully.")
    except Exception as e:
        print(f"Error saving complete model: {e}")

    try:
        class TorchScriptWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


        wrapper = TorchScriptWrapper(model)
        example_input_ids = torch.ones((1, 512), dtype=torch.long).to(device)
        example_attention_mask = torch.ones((1, 512), dtype=torch.long).to(device)
        traced_model = torch.jit.trace(wrapper, (example_input_ids, example_attention_mask))
        traced_model.save(os.path.join(PATH_TO_SAVE, "model", "traced_model.pt"))
        print("Traced model saved successfully.")
    except Exception as e:
        print(f"Error saving traced model: {e}")
    try:
        torch.save(model.state_dict(), os.path.join(PATH_TO_SAVE, "model", "state_dict.pt"))
        print("State dict saved successfully.")
    except Exception as e:
        print(f"Error saving state dict: {e}")

    print("Total execution time:", round(time.time() - starting_time, 2), "seconds")
