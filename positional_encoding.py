import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int) -> None:
        super().__init__()

        self.max_len = max_len
        self.embedding_dim = embedding_dim

        positional_embeddings = torch.zeros(max_len, embedding_dim)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        division_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        positional_embeddings[:, 0::2] = torch.sin(position * division_term)
        positional_embeddings[:, 1::2] = torch.cos(position * division_term)

        self.pe = positional_embeddings.unsqueeze(0)
        # self.register_buffer("pe", positional_embeddings.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]
