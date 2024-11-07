import math

import torch
from torch import nn
from torch.nn import functional as F

from layer.MaskedAttentionLayer import MaskedAttentionLayer


class GATModel(nn.Module):
    def __init__(self, vocab_size, context_length=128, n_heads=4, embed_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.module = nn.Sequential(
            MaskedAttentionLayer(embed_dim=embed_dim, n_heads=n_heads),
        )
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def loss_function(self, logits, target):
        logits = logits.permute(0, 2, 1)
        target = target[:, 1:]
        logits = logits[:, :, :-1]
        return F.cross_entropy(logits, target)

    def forward(self, x):
        f1 = self.token_embedding.forward(x) + self.position_embedding(torch.arange(0, x.shape[1], device=x.device))
        f2 = self.module(f1)
        f3 = self.linear.forward(f1 + f2)

        return f3

    def generate(self, batch_size=1, max_sequence_length=100, device="cpu"):
        texts = torch.zeros(size=(batch_size, 1)).to(device).long()
        texts[:, 0] = 0
        with torch.no_grad():
            for i in range(min(self.context_length, max_sequence_length)):
                if i == 0:
                    continue
                next_logits = self.forward(texts)

                next_probs = torch.softmax(next_logits[:, -1, :-1], dim=1)
                next_tokens = torch.multinomial(next_probs, num_samples=1)
                texts = torch.cat((texts, next_tokens), dim=1)

        return texts[:, 1:]
