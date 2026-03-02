import torch.nn as nn

from swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, hidden_dim: int, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = SwiGLU(embedding_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        residual = x
        x = self.ln1(x)

        attn_output, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)

        x = residual + self.dropout(attn_output)

        residual = x
        x = self.ln2(x)

        x = residual + self.dropout(self.ffn(x))

        return x
