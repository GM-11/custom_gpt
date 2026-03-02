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


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0:
        return logits

    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    kth_value = values[..., -1, None]
    return torch.where(
        logits < kth_value, torch.full_like(logits, float("-inf")), logits
    )


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(list(xs), batch_first=True, padding_value=0)
    ys = torch.nn.utils.rnn.pad_sequence(list(ys), batch_first=True, padding_value=0)
    return xs, ys
