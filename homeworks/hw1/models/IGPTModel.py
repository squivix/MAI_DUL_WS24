import math

import torch
from torch import nn
from torch.nn import functional as F

from layer.MaskedAttentionLayer import MaskedAttentionLayer


class IGPTModel(nn.Module):
    def __init__(self, sequence_length, embed_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(3, embed_dim)
        # self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(sequence_length, embed_dim), requires_grad=False)

        self.module = nn.Sequential(
            MaskedAttentionLayer(embed_dim=embed_dim, sequence_length=sequence_length),
        )
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),
        )

    def loss_function(self, logits, target):
        target = target[:, 1:]
        logits = logits[:, :, :-1]
        return F.cross_entropy(logits, target)

    def forward(self, x):
        embedding = self.token_embedding.forward(x) + self.positional_encoding[:self.sequence_length, :].to(x.device)  # self.position_embedding(torch.arange(0, self.sequence_length, device=x.device))

        f1 = self.module(embedding)
        f3 = self.linear.forward(f1)

        return f3.permute(0, 2, 1)

    def generate(self, batch_size=1, device="cpu"):
        sequence = 401
        images = torch.zeros(size=(batch_size, sequence)).to(device).long()
        images[:, 0] = 2
        with torch.no_grad():
            for i in range(sequence):
                if i == 0:
                    continue
                next_logits = self.forward(images)

                next_probs = torch.softmax(next_logits[:, :, i], dim=1)
                next_pixels = torch.multinomial(next_probs, num_samples=1).squeeze(1)
                images[:, i] = next_pixels

        return images[:, 1:].reshape(batch_size, 1, 20, 20)

    def _generate_positional_encoding(self, max_seq_len, embed_dim):
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
