import torch
from torch import nn
from torch.nn import functional as F

from layer.MaskedAttentionLayer import MaskedAttentionLayer


class IGPTModel(nn.Module):
    def __init__(self, d_k=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_k = d_k
        self.module = nn.Sequential(
            nn.Flatten(1),
            nn.Unflatten(1, (20 * 20, 1)),
            MaskedAttentionLayer(embedding_size=1, head_dimension=1, sequence_length=400),
            nn.GELU(),
            nn.Linear(800, 800),
            nn.Unflatten(1, (2, 20, 20))
        )

    def loss_function(self, x, logits):
        return F.nll_loss(torch.log_softmax(logits, dim=1), (x[:, 0] == -1).long())

    def forward(self, x):
        return self.module(x)
