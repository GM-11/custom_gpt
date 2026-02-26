import torch
import torch.nn as nn

from manual_self_attention import ManualSelfAttention


class SelfAttnWithPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, embedding_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)
        self.attn = ManualSelfAttention(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, token_ids: torch.Tensor):
        batch_size, seq_len = token_ids.shape
        positions = (
            torch.arange(seq_len, device=token_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        word_vetors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)
        input_vectors = word_vetors + position_vectors
        attention_out, attention_weights = self.attn(input_vectors)

        last_hidden = attention_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits, attention_weights
