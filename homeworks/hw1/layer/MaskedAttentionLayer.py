import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedAttentionLayer(nn.Module):
    def __init__(self, in_features, d_k, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_weights = nn.Linear(in_features, d_k)
        self.k_weights = nn.Linear(in_features, d_k)
        self.v_weights = nn.Linear(d_k, out_features)
        self.mask = nn.Parameter(torch.triu(torch.ones(d_k, d_k), diagonal=1), requires_grad=False)
        self.d_k = d_k

    def forward(self, x):
        q = self.q_weights.forward(x)
        k = self.k_weights.forward(x)
        s = (q @ k.T - (1 - self.mask) * 10e10) / math.sqrt(self.d_k)
        logits = self.v_weights.forward(torch.softmax(s, dim=0))
        return logits
