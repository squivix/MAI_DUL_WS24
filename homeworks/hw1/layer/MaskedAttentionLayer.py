import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedAttentionLayer(nn.Module):
    def __init__(self, embedding_size, head_dimension, sequence_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_weights = nn.Parameter(torch.Tensor(head_dimension, embedding_size))
        self.k_weights = nn.Parameter(torch.Tensor(head_dimension, embedding_size))
        self.v_weights = nn.Parameter(torch.Tensor(head_dimension, embedding_size))
        self.d_k = head_dimension
        self.mask = nn.Parameter(torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1), requires_grad=False)

    def forward(self, x):
        q = x @ self.q_weights
        k = x @ self.k_weights
        v = x @ self.v_weights
        s = (q @ k.permute(0, 2, 1) - (1 - self.mask) * 10e10) / math.sqrt(self.d_k)
        s = torch.softmax(s, dim=1)
        logits = s @ v
        return logits
