import os
from pathlib import Path

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import get_root_path



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


if __name__ == '__main__':
    MODEL_VERSION: str = "v1.0"
    PATH = os.path.join(get_root_path(), "models", MODEL_VERSION)

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
