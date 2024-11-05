import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedAttentionLayer(nn.Module):
    def __init__(self, embed_dim,  sequence_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.k_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.v_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.d_k = embed_dim
        self.mask = nn.Parameter(torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1),
                                 requires_grad=False)
        nn.init.kaiming_uniform_(self.q_weights)
        nn.init.kaiming_uniform_(self.k_weights)
        nn.init.kaiming_uniform_(self.v_weights)

    def forward(self, x):
        q = x @ self.q_weights.T
        k = x @ self.k_weights.T
        v = x @ self.v_weights.T
        s = (q @ k.permute(0, 2, 1) - (1 - self.mask) * 1e10) / math.sqrt(self.d_k)

        s = torch.softmax(s, dim=1)
        logits = s @ v
        return logits
