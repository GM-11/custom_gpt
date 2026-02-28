import math

import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding
from utils import create_padding_mask, make_causal_mask


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        self.pos_enc = PositionalEncoding(max_len, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_blocks = nn.TransformerEncoder(
            encoder_layer=decoder_layer, num_layers=num_layers
        )

        self.ln_final = nn.LayerNorm(embedding_dim)

        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):
        padding_mask = create_padding_mask(src, pad_idx=0)

        causal_mask = make_causal_mask(src.size(1)).to(src.device)

        x = self.token_emb(src) * math.sqrt(self.embedding_dim)

        x = x + self.pos_enc(x)

        x = self.dropout(x)

        x = self.transformer_blocks(
            src=x, mask=causal_mask, src_key_padding_mask=padding_mask
        )

        x = self.ln_final(x)

        logits = self.output_projection(x)

        return logits
