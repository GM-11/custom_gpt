# %%
import math

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from custom_dataset import CustomDataset
from decoder import Decoder
from model_training import train_model

# %%
encoder = tiktoken.encoding_for_model("gpt-2")
vocab_size = encoder.n_vocab
print(vocab_size)
# encoder = CustomTokenizer()

# %%
text = open("transcripts.txt", "r", encoding="utf-8").read()
# encoder.build_vocab(text)
token_ids = encoder.encode(text)
data = torch.tensor(token_ids, dtype=torch.long)  # shape: (N,)
vocab_size = encoder.n_vocab
print(data.shape, data.dtype)

# %%
seq_len = 64
dataset = CustomDataset(data, seq_len=seq_len)

# %%
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# %%
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

xb, yb = next(iter(train_loader))
print("xb:", xb.shape, "yb:", yb.shape)  # (B, T) and (B, T)
# %%
model = Decoder(
    vocab_size=vocab_size,
    embedding_dim=256,
    nhead=8,
    num_layers=3,
    dim_feedforward=1024,
    max_len=512,
    dropout=0.1,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AdamW = Adam + correct weight decay behavior
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

# ---- warmup + cosine decay scheduler (step every batch) ----
epochs = 10
steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch
warmup_steps = int(0.05 * total_steps)


def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    vocab_size=vocab_size,
    epochs=epochs,
    device=device,
    grad_clip_norm=1.0,
    scheduler=scheduler,
    val_max_batches=None,
)
# %%
text = "I love you more than life\n"
max_tokens = 500
temperature = 1.5
top_k = 20
print(text, end="")
with torch.no_grad():
    model.eval()

    token_ids = encoder.encode(text)

    for _ in range(max_tokens):
        ctx = token_ids[-seq_len:]
        x = torch.tensor([ctx], dtype=torch.long)

        logits = model(x)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature

        if top_k is not None:
            k = min(top_k, next_logits.size(-1))
            top_vals, top_idx = torch.topk(next_logits, k=k)
            probs = torch.softmax(top_vals, dim=-1)
            next_id = int(top_idx[torch.multinomial(probs, num_samples=1)].item())
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        print(encoder.decode([next_id]), end="")
        token_ids.append(next_id)

    print(encoder.decode(token_ids))
