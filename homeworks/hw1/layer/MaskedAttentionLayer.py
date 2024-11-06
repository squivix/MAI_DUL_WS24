import math

import torch
from torch import nn


class MaskedAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = embed_dim // n_heads
        self.q_weights = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(n_heads)])
        self.k_weights = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(n_heads)])
        self.v_weights = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(n_heads)])

        self.n_heads = n_heads

    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape
        logits = []
        for i in range(self.n_heads):
            q = self.q_weights[i](x)
            k = self.k_weights[i](x)
            v = self.v_weights[i](x)
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(x.device)
            s = (q @ k.permute(0, 2, 1) + mask) / math.sqrt(self.head_dim)
            s = torch.softmax(s, dim=-1)
            logits.append(s @ v)
        return torch.concat(logits, dim=-1)
