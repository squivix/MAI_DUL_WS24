import torch
from torch import nn
from torch.nn import functional as F

from layer.MaskedAttentionLayer import MaskedAttentionLayer


class IGPTModel(nn.Module):
    def __init__(self, d_k=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            nn.Flatten(1),
            MaskedAttentionLayer(20 * 20, d_k, 800),
            nn.GELU(),
            nn.Linear(800, 800),
            nn.Unflatten(1, (2, 20, 20))

        )

    def loss_function(self, x, logits):
        return F.nll_loss(torch.log_softmax(logits, dim=1), (x[:, 0] == -1).long())

    def forward(self, x):
        return self.module(x)
