import math
from typing import Optional

import torch
from torch import nn, optim
from torch.nn.modules import loss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    vocab_size: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (xb, yb) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)  # (B, T, V)
        loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))

        tokens = yb.numel()
        total_tokens += tokens
        total_loss += loss.item() * tokens

    return total_loss / max(1, total_tokens)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: loss._WeightedLoss,
    optimizer: optim.Optimizer,
    vocab_size: int,
    epochs: int = 20,
    device: Optional[torch.device] = None,
    grad_clip_norm: float = 1.0,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    val_max_batches: Optional[int] = None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        num_tokens = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()

                logits = model(xb)  # (B, T, V)
                loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))

                loss.backward()

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                token_count = yb.numel()
                num_tokens += token_count
                total_loss += loss.item() * token_count

                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": loss.item(), "lr": lr})

        train_avg = total_loss / max(1, num_tokens)

        if val_loader is not None:
            val_avg = evaluate_loss(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                vocab_size=vocab_size,
                device=device,
                max_batches=val_max_batches,
            )
            print(
                f"Epoch {epoch + 1}: train loss = {train_avg:.4f} | val loss = {val_avg:.4f} | val ppl = {math.exp(val_avg):.2f}"
            )
        else:
            print(f"Epoch {epoch + 1}: train loss = {train_avg:.4f}")
