import matplotlib.pyplot as plt
import torch


def plot_attention(attn_weights, tokens, title="Self-Attention Map"):
    aw = attn_weights[0].detach().cpu().numpy()
    plt.figure(figsize=(1.2 * len(tokens), 5))  # Adjust figure size by sequence length
    plt.imshow(aw, cmap="Blues")
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def make_causal_mask(sz: int):
    mask = torch.full((sz, sz), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return seq == pad_idx
