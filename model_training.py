from torch import nn, optim
from torch.nn.modules import loss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: loss._WeightedLoss,
    optimizer: optim.Optimizer,
    epochs=20,
):
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        num_samples = 0
        with tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for xb, yb in pbar:
                bs = xb.size(0)
                optimizer.zero_grad()

                logits, _ = model(xb)
                loss = loss_fn(logits, yb)

                loss.backward()

                optimizer.step()

                num_samples += bs

                total_loss += loss.item() * bs

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch + 1}: avg loss = {avg_loss}")
