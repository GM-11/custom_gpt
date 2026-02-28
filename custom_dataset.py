import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, token_stream: torch.Tensor, seq_len: int) -> None:
        if token_stream.dtype != torch.long or token_stream.dim() != 1:
            raise ValueError("token_stream must be a 1D torch.long tensor of token ids")
        self.data = token_stream
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]  # (T,)
        y = self.data[idx + 1 : idx + self.seq_len + 1]  # (T,)
        return x, y
