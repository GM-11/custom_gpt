import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualSelfAttention(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()

        self.to_query = nn.Linear(d, d, bias=False)
        self.to_key = nn.Linear(d, d, bias=False)
        self.to_value = nn.Linear(d, d, bias=False)

    def forward(self, x):
        Q = self.to_query(x)
        K = self.to_key(x)
        V = self.to_value(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))

        attn = F.softmax(scores, dim=-1)

        out = attn @ V

        return out, attn
