import os
from pathlib import Path

import torch
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datalint_ml.utils.path import get_project_root
from utils.metrics import plot_attention, print_tokenizer_details, print_model_details, print_parameter_sizes

if __name__ == '__main__':
    MODEL_VERSION: str = "v1.0"
    PATH = os.path.join(get_project_root(), "models", MODEL_VERSION)

    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(PATH, "model"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PATH, "tokenizer"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    example = "SELECT * FROM users;"
    inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Print the input tensors
    print("\n===== Input Tensors =====")
    for key, value in inputs.items():
        print(f"{key}: {value.shape}")
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Attention Mask: {inputs['attention_mask']}")
    print(f"Token Type IDs: {inputs['token_type_ids']}")
    print(f"Input IDs (decoded): {tokenizer.decode(inputs['input_ids'][0])}")

    print("===== Model Configuration =====")
    print(model.config)

    print_model_details(model)
    print_tokenizer_details(tokenizer)
    print_parameter_sizes(model)

    print("\n===== Model Summary =====")
    summary(model, input_data=(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]), depth=2)

    print("\n===== Model Size =====")
    model_size = sum(f.stat().st_size for f in Path(PATH, 'tiny_model').rglob('*')) / 1e6
    print(f"Model size: {model_size:.2f} MB")

    # Visualize attention weights
    plot_attention(model, tokenizer, inputs)
