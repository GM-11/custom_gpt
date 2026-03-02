import math

import torch.nn as nn

from custom_transformer import TransformerBlock
from positional_encoding import PositionalEncoding
from utils import make_causal_mask


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_len: int,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.output_projection.weight = self.token_emb.weight

        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        self.pos_enc = PositionalEncoding(max_len, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    n_heads=nhead,
                    hidden_dim=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(embedding_dim)

        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):

        causal_mask = make_causal_mask(src.size(1)).to(src.device)

        x = self.token_emb(src) * math.sqrt(self.embedding_dim)

        x = x + self.pos_enc(x)

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)

        x = self.ln_final(x)

        logits = self.output_projection(x)

        return logits
